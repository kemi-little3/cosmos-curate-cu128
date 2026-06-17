# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ViPE camera pose estimation stage."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from typing import Literal

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.pose.pose_utils import (
    align_pose_length,
    build_pose_meta,
    build_relative_pose,
    load_vipe_adapter_output,
    sample_id_from_window,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Window

DEFAULT_VIPE_ADAPTER = "/data1/hexuming/pose_estimation/adapter/scripts/run_adapter.py"


class VipePoseStage(CuratorStage):
    """Run ViPE on each caption window and attach camera pose arrays."""

    def __init__(
        self,
        vipe_python: str,
        adapter_script: str = DEFAULT_VIPE_ADAPTER,
        fail_policy: Literal["warn-only", "skip-window"] = "warn-only",
        num_gpus_per_worker: float = 1.0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._vipe_python = vipe_python
        self._adapter_script = adapter_script
        self._fail_policy = fail_policy
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)

    @property
    def resources(self) -> CuratorStageResource:
        """Reserve a GPU so the ViPE subprocess inherits CUDA visibility."""
        return CuratorStageResource(cpus=1.0, gpus=self._num_gpus_per_worker)

    def _mark_error(self, window: Window, error: str) -> None:
        window.pose_status = "error"
        window.pose_error = error
        window.errors["vipe_pose"] = error

    def _stage_window_input(self, window: Window, clip_dir: pathlib.Path) -> None:
        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "window has no mp4 bytes"
            raise ValueError(msg)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
        clip_mp4 = clip_dir / "clip.mp4"
        clip_mp4.write_bytes(mp4_data.tobytes())

        num_frames = window.end_frame - window.start_frame + 1
        frame_indices = np.arange(num_frames, dtype=np.int32)
        timestamps = np.arange(num_frames, dtype=np.float64)
        np.save(clip_dir / "frame_indices.npy", frame_indices)
        np.save(clip_dir / "timestamps.npy", timestamps)
        width, height, fps = self._probe_video(clip_mp4)
        timestamps = np.arange(num_frames, dtype=np.float64) / fps if fps > 0 else timestamps
        np.save(clip_dir / "timestamps.npy", timestamps)
        meta = {
            "source_mp4": str(clip_mp4),
            "original_resolution": [width, height],
            "original_fps": fps,
            "duration_s": None,
            "num_raw_frames": num_frames,
            "sampled_fps": fps,
            "num_sampled": num_frames,
        }
        with (clip_dir / "meta.json").open("w") as f:
            json.dump(meta, f)

    def _probe_video(self, clip_mp4: pathlib.Path) -> tuple[int, int, float]:
        try:
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate",
                    "-of",
                    "json",
                    str(clip_mp4),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(out.stdout)["streams"][0]
            num, den = stream.get("avg_frame_rate", "0/1").split("/")
            fps = float(num) / float(den) if float(den) else 0.0
            return int(stream["width"]), int(stream["height"]), fps
        except Exception:  # noqa: BLE001
            logger.warning(f"Could not probe video metadata for {clip_mp4}; using zeros in pose adapter meta")
            return 0, 0, 0.0

    def _run_adapter(self, clip_dir: pathlib.Path, out_root: pathlib.Path, sample_id: str) -> pathlib.Path:
        cmd = [
            self._vipe_python,
            self._adapter_script,
            "--method",
            "vipe",
            "--clip-dir",
            str(clip_dir),
            "--out-root",
            str(out_root),
            "--clip-uuid",
            sample_id,
        ]
        if self._verbose:
            logger.info(f"Running ViPE pose adapter: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return out_root / sample_id / "vipe"

    def _process_window(self, window: Window, clip_uuid: str) -> None:
        sample_id = sample_id_from_window(clip_uuid, window.start_frame, window.end_frame)
        target_num_frames = window.end_frame - window.start_frame + 1
        with tempfile.TemporaryDirectory(prefix="cosmos_vipe_pose_") as tmp:
            tmp_path = pathlib.Path(tmp)
            clip_dir = tmp_path / "input" / sample_id
            out_root = tmp_path / "output"
            self._stage_window_input(window, clip_dir)
            vipe_out = self._run_adapter(clip_dir, out_root, sample_id)
            adapter_data = load_vipe_adapter_output(vipe_out)

        intrinsics, poses_c2w, num_interpolated = align_pose_length(
            adapter_data["intrinsics"],
            adapter_data["poses_c2w"],
            target_num_frames,
        )
        relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
        quality = adapter_data.get("quality") or {}
        status = "degraded" if num_interpolated > 0 or quality.get("status") == "degraded" else "ok"
        window.pose_intrinsics = intrinsics.astype(np.float32)
        window.pose_c2w = poses_c2w.astype(np.float32)
        window.pose_relative = relative_poses.astype(np.float32)
        window.pose_status = status
        window.pose_error = None
        window.pose_meta = build_pose_meta(
            clip_uuid=clip_uuid,
            start_frame=window.start_frame,
            end_frame=window.end_frame,
            num_frames=target_num_frames,
            status=status,
            translation_scale_factor=scale_factor,
            total_path_length_before_norm=total_path,
            quality=quality,
            num_interpolated_frames=num_interpolated,
        )

    @nvtx.annotate("VipePoseStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Attach ViPE pose arrays to every clip window."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process(len(task.video.clips)):
                for clip in task.video.clips:
                    for window in clip.windows:
                        try:
                            self._process_window(window, str(clip.uuid))
                        except (OSError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                            error = str(exc)
                            logger.exception(
                                f"ViPE pose failed for clip {clip.uuid} window "
                                f"{window.start_frame}_{window.end_frame}: {error}"
                            )
                            self._mark_error(window, error)
                            if self._fail_policy == "skip-window":
                                window.caption_status = "skipped"
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
