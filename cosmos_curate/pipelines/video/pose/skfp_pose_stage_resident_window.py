# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident SparseKeyframePose stage using keyframe-union window reconstruction."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import nvtx  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.pose.pose_utils import build_pose_meta, build_relative_pose
from cosmos_curate.pipelines.video.pose.skfp_resident_common import SkfpResidentClient, summarize_resident_worker_report
from cosmos_curate.pipelines.video.utils.decoder_utils import decode_video_cpu_frame_ids
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Window


class SkfpResidentWindowStage(CuratorStage):
    """Run one union sparse ViPE job per clip, then interpolate each window."""

    def __init__(
        self,
        *,
        skfp_root: str,
        vipe_python: str,
        vipe_adapter_script: str,
        vipe_work_root: str | None = None,
        stride: int = 32,
        min_keyframes: int = 3,
        max_keyframes: int = 0,
        fail_policy: Literal["warn-only", "skip-window"] = "warn-only",
        num_gpus_per_worker: float = 1.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._skfp_root = skfp_root
        self._vipe_python = vipe_python
        self._vipe_adapter_script = vipe_adapter_script
        self._vipe_work_root = vipe_work_root
        self._stride = stride
        self._min_keyframes = min_keyframes
        self._max_keyframes = max_keyframes
        self._fail_policy = fail_policy
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)
        self._client: SkfpResidentClient | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Reserve GPU resources for the resident SKFP/ViPE subprocess."""
        return CuratorStageResource(cpus=1.0, gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Start the external SKFP Python worker once per actor."""
        if self._skfp_root not in sys.path:
            sys.path.insert(0, self._skfp_root)
        self._client = SkfpResidentClient(
            skfp_root=self._skfp_root,
            vipe_python=self._vipe_python,
            vipe_adapter_script=self._vipe_adapter_script,
            vipe_work_root=self._vipe_work_root,
            stride=self._stride,
            min_keyframes=self._min_keyframes,
            max_keyframes=self._max_keyframes,
            verbose=self._verbose,
        )
        self._client.start()

    def destroy(self) -> None:
        """Stop the external SKFP Python worker."""
        if self._client is not None:
            self._client.stop()
            self._client = None

    def _mark_error(self, window: Window, error: str) -> None:
        window.pose_status = "error"
        window.pose_error = error
        window.errors["skfp_pose"] = error

    def _attach_window_dense(self, window: Window, *, clip_uuid: str, dense, quality: dict) -> None:
        target_num_frames = window.end_frame - window.start_frame + 1
        poses_c2w = dense.poses_c2w
        intrinsics = dense.intrinsics
        if poses_c2w.shape[0] != target_num_frames:
            raise ValueError(f"SKFP union dense length {poses_c2w.shape[0]} does not match window length {target_num_frames}")
        relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
        window.pose_intrinsics = intrinsics.astype(np.float32)
        window.pose_c2w = poses_c2w.astype(np.float32)
        window.pose_relative = relative_poses.astype(np.float32)
        window.pose_status = "ok"
        window.pose_error = None
        adapter_quality = dict(quality)
        adapter_quality.update(
            {
                "runner": "skfp_resident_keyframe_union_slice_sparse_then_interpolate",
                "pipeline_run_mode": "resident-window",
                "skfp_stride": self._stride,
                "skfp_min_keyframes": self._min_keyframes,
                "skfp_max_keyframes": self._max_keyframes,
                "slice_source_window": {"start_frame": window.start_frame, "end_frame": window.end_frame},
            }
        )
        window.pose_meta = build_pose_meta(
            clip_uuid=clip_uuid,
            start_frame=window.start_frame,
            end_frame=window.end_frame,
            num_frames=target_num_frames,
            status="ok",
            translation_scale_factor=scale_factor,
            total_path_length_before_norm=total_path,
            quality={"adapter_quality": adapter_quality},
            num_interpolated_frames=0,
        )
        window.pose_meta["method"] = "skfp_vipe"
        window.pose_meta["label_source"] = "skfp_vipe"

    def _process_clip(self, clip_uuid: str, clip) -> None:
        if not clip.windows:
            return
        if self._client is None:
            self.stage_setup()
        assert self._client is not None
        if self._skfp_root not in sys.path:
            sys.path.insert(0, self._skfp_root)

        from sparse_keyframe_pose.interpolation import interpolate_dense_poses  # noqa: PLC0415
        from sparse_keyframe_pose.resident_stage import (  # noqa: PLC0415
            KeyframeUnionSparseJob,
            WindowSparseSelection,
            load_worker_sparse_pose,
            slice_union_sparse_pose,
            write_sparse_resident_frame_dir_input,
        )
        from sparse_keyframe_pose.keyframes import select_uniform_keyframes  # noqa: PLC0415

        clip_data = clip.encoded_data.resolve()
        if clip_data is None:
            raise ValueError("clip has no encoded_data")
        clip_meta = clip.extract_metadata() or {}
        staged_num_frames = max(window.end_frame for window in clip.windows) + 1
        fps = float(clip_meta.get("framerate") or 0.0)
        if fps <= 0:
            raise ValueError("clip metadata has invalid framerate")
        width = int(clip_meta.get("width") or 0)
        height = int(clip_meta.get("height") or 0)
        selections: list[WindowSparseSelection] = []
        union_frames: set[int] = set()
        for window in clip.windows:
            target_num_frames = int(window.end_frame) - int(window.start_frame) + 1
            local_keyframes = select_uniform_keyframes(
                num_frames=target_num_frames,
                stride=self._stride,
                min_keyframes=self._min_keyframes,
                max_keyframes=self._max_keyframes,
            ).astype(np.int64)
            source_frames = (local_keyframes + int(window.start_frame)).astype(np.int64)
            selections.append(
                WindowSparseSelection(
                    start_frame=int(window.start_frame),
                    end_frame=int(window.end_frame),
                    local_keyframe_indices=local_keyframes,
                    source_frame_indices=source_frames,
                )
            )
            union_frames.update(int(x) for x in source_frames.tolist())
        union_keyframes = np.asarray(sorted(union_frames), dtype=np.int32)
        frames_rgb = decode_video_cpu_frame_ids(clip_data, union_keyframes.astype(np.int32), num_threads=2)

        sample_id = f"{clip_uuid}_keyframe_union"
        with tempfile.TemporaryDirectory(prefix="cosmos_skfp_union_") as tmp:
            tmp_path = Path(tmp)
            sparse_dir = tmp_path / "input" / sample_id
            worker_out_root = tmp_path / "worker_outputs"
            write_sparse_resident_frame_dir_input(
                clip_dir=sparse_dir,
                frames_rgb=frames_rgb,
                keyframe_indices=union_keyframes,
                source_frame_indices=union_keyframes,
                window_id=sample_id,
                fps=fps,
                width=width,
                height=height,
                num_raw_frames=staged_num_frames,
                source_label=str(clip.source_video),
            )
            sparse_job = KeyframeUnionSparseJob(
                sample_id=sample_id,
                sparse_dir=sparse_dir,
                worker_out_root=worker_out_root,
                union_keyframe_indices=union_keyframes,
                windows=tuple(selections),
            )
            report = self._client.run_stage_jobs(jobs=[sparse_job.job_payload], label="keyframe_union")
            method_dir = sparse_job.method_dir
            worker_job = (report.get("jobs") or [{}])[0] if isinstance(report, dict) else {}
            union_sparse = load_worker_sparse_pose(
                method_dir=method_dir,
                keyframe_indices=union_keyframes,
                estimator_name="vipe",
                worker_report=worker_job if isinstance(worker_job, dict) else {},
            )
            for window, selection in zip(clip.windows, sparse_job.windows, strict=True):
                source_frames = selection.source_frame_indices
                local_keyframes = selection.local_keyframe_indices
                sparse = slice_union_sparse_pose(union_sparse, selection)
                pose_meta = {
                    "method": "sparse_keyframe_pose",
                    "base_estimator": "vipe",
                    "keyframe_strategy": f"uniform_stride_{self._stride}",
                    "keyframe_stride": self._stride,
                    "keyframe_indices": local_keyframes.astype(int).tolist(),
                    "num_keyframes": int(len(local_keyframes)),
                    "num_dense_frames": int(window.end_frame - window.start_frame + 1),
                    "dense_pose_source": "resident_union_sparse_slice_interpolation",
                    "estimator_meta": sparse.estimator_meta,
                }
                dense = interpolate_dense_poses(sparse, num_frames=window.end_frame - window.start_frame + 1, pose_meta=pose_meta)
                quality = {
                    "skfp_meta": dense.pose_meta,
                    "resident_worker_summary": summarize_resident_worker_report(report),
                    **({"resident_worker_debug": report} if os.environ.get("SKFP_REPORT_LEVEL", "summary").strip().lower() == "debug" else {}),
                    "num_keyframes": int(len(local_keyframes)),
                    "keyframe_indices": local_keyframes.astype(int).tolist(),
                    "union_source_frame_indices": source_frames.astype(int).tolist(),
                }
                self._attach_window_dense(window, clip_uuid=clip_uuid, dense=dense, quality=quality)

    @nvtx.annotate("SkfpResidentWindowStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Attach benchmark-aligned keyframe-union SKFP pose arrays to windows."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process(len(task.video.clips)):
                for clip in task.video.clips:
                    try:
                        self._process_clip(str(clip.uuid), clip)
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                        error = str(exc)
                        logger.exception(f"Resident SKFP union pose failed for clip {clip.uuid}: {error}")
                        for window in clip.windows:
                            self._mark_error(window, error)
                            if self._fail_policy == "skip-window":
                                window.caption_status = "skipped"
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
