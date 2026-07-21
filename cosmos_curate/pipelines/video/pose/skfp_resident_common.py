# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for resident SparseKeyframePose pipeline stages."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import threading
from typing import Any

import numpy as np
from loguru import logger

from cosmos_curate.pipelines.video.pose.pose_utils import align_pose_length, build_pose_meta, build_relative_pose

RESPONSE_PREFIX = "__SKFP_RPC__ "


def load_skfp_dense_output(out_dir: pathlib.Path) -> dict[str, Any]:
    """Load dense pose files emitted by SparseKeyframePose."""
    out_dir = pathlib.Path(out_dir)
    dense_dir = out_dir / "dense"
    if not dense_dir.exists():
        dense_dir = out_dir

    meta_path = dense_dir / "pose_meta.json"
    quality: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open() as fp:
            skfp_meta = json.load(fp)
        quality = {
            "skfp_meta": skfp_meta,
            "warnings": skfp_meta.get("warnings", []),
        }

    poses_path = dense_dir / "poses_c2w.npy"
    if not poses_path.exists():
        poses_path = dense_dir / "poses.npy"

    return {
        "intrinsics": np.load(dense_dir / "intrinsics.npy").astype(np.float32),
        "poses_c2w": np.load(poses_path).astype(np.float64),
        "relative_poses": (
            np.load(dense_dir / "relative_poses.npy").astype(np.float64)
            if (dense_dir / "relative_poses.npy").exists()
            else None
        ),
        "quality": quality,
    }


def attach_skfp_output_to_window(
    window: Any,
    *,
    clip_uuid: str,
    skfp_out: pathlib.Path,
    quality_extra: dict[str, Any] | None = None,
) -> None:
    """Load a SKFP output directory and attach arrays/meta to a window."""
    target_num_frames = window.end_frame - window.start_frame + 1
    skfp_data = load_skfp_dense_output(skfp_out)
    intrinsics, poses_c2w, num_interpolated = align_pose_length(
        skfp_data["intrinsics"],
        skfp_data["poses_c2w"],
        target_num_frames,
    )
    relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
    quality = skfp_data.get("quality") or {}
    if quality_extra:
        quality = {**quality, **quality_extra}
    quality.setdefault("base_estimator", "vipe")
    quality.setdefault("runner", "skfp_resident_window")
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
    window.pose_meta["method"] = "skfp_vipe"
    window.pose_meta["label_source"] = "skfp_vipe"


def attach_skfp_clip_output_slice_to_window(
    window: Any,
    *,
    clip_uuid: str,
    skfp_out: pathlib.Path,
    quality_extra: dict[str, Any] | None = None,
) -> None:
    """Load a full-clip SKFP output, slice it by window bounds, and attach arrays/meta."""
    skfp_data = load_skfp_dense_output(skfp_out)
    start = window.start_frame
    end = window.end_frame + 1

    intrinsics = skfp_data["intrinsics"][start:end]
    poses_c2w = skfp_data["poses_c2w"][start:end]
    target_num_frames = window.end_frame - window.start_frame + 1
    if poses_c2w.shape[0] != target_num_frames:
        msg = f"SKFP clip-once output too short for window {window.start_frame}_{window.end_frame}"
        raise ValueError(msg)

    relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
    quality = dict(skfp_data.get("quality") or {})
    if quality_extra:
        quality = {**quality, **quality_extra}
    quality.setdefault("base_estimator", "vipe")
    quality["runner"] = "skfp_resident_clip_once_slice"
    quality["slice_source_window"] = {"start_frame": window.start_frame, "end_frame": window.end_frame}
    status = "degraded" if quality.get("status") == "degraded" else "ok"

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
        num_interpolated_frames=0,
    )
    window.pose_meta["method"] = "skfp_vipe"
    window.pose_meta["label_source"] = "skfp_vipe"


def _pick_timing(timings: dict[str, Any], key: str) -> float | None:
    value = timings.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_resident_worker_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a compact production-safe resident worker report."""
    summary: dict[str, Any] = {}
    for key in (
        "mode",
        "job_count",
        "planned_job_count",
        "worker_startup_s",
        "worker_total_s",
        "jobs_total_s",
        "stage_jobs_total_s",
        "stage_residual_s",
    ):
        if key in report:
            summary[key] = report[key]

    jobs: list[dict[str, Any]] = []
    for job in report.get("jobs", []) or []:
        if not isinstance(job, dict):
            continue
        timings = job.get("timings") if isinstance(job.get("timings"), dict) else {}
        item: dict[str, Any] = {
            "clip_uuid": job.get("clip_uuid"),
            "input_mode": job.get("input_mode"),
            "job_total_s": job.get("job_total_s"),
            "pipeline_run_s": job.get("pipeline_run_s") or _pick_timing(timings, "pipeline_run_s"),
            "artifact_convert_s": job.get("artifact_convert_s") or _pick_timing(timings, "artifact_convert_s"),
            "status": job.get("status") or ("error" if job.get("error") else "ok"),
        }
        if job.get("error"):
            item["error"] = job.get("error")
        jobs.append({key: value for key, value in item.items() if value is not None})
    summary["jobs"] = jobs
    return summary


class SkfpResidentClient:
    """JSONL stdin/stdout client for the external resident SKFP worker."""

    def __init__(
        self,
        *,
        skfp_root: str,
        vipe_python: str,
        vipe_adapter_script: str,
        skfp_python: str | None = None,
        vipe_work_root: str | None = None,
        stride: int = 32,
        min_keyframes: int = 3,
        max_keyframes: int = 0,
        verbose: bool = False,
    ) -> None:
        self._skfp_python = skfp_python or sys.executable
        self._skfp_root = skfp_root
        self._vipe_python = vipe_python
        self._vipe_adapter_script = vipe_adapter_script
        self._vipe_work_root = vipe_work_root
        self._stride = stride
        self._min_keyframes = min_keyframes
        self._max_keyframes = max_keyframes
        self._verbose = verbose
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        worker_script = pathlib.Path(__file__).with_name("skfp_resident_rpc_worker.py")
        cmd = [
            self._skfp_python,
            str(worker_script),
            "--skfp-root",
            self._skfp_root,
            "--vipe-python",
            self._vipe_python,
            "--vipe-adapter-script",
            self._vipe_adapter_script,
            "--stride",
            str(self._stride),
            "--min-keyframes",
            str(self._min_keyframes),
            "--max-keyframes",
            str(self._max_keyframes),
        ]
        if self._vipe_work_root:
            cmd.extend(["--vipe-work-root", self._vipe_work_root])
        if self._verbose:
            logger.info(f"Starting resident SKFP worker: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(target=self._forward_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read_response()
        if ready.get("status") != "ready":
            msg = f"resident SKFP worker failed to start: {ready}"
            raise RuntimeError(msg)

    def run_stage_jobs(self, *, jobs: list[dict[str, Any]], label: str = "skfp") -> dict[str, Any]:
        """Run sparse-keyframe jobs through the ViPE resident stage-worker."""
        if self._proc is None:
            self.start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        request: dict[str, Any] = {
            "command": "stage_jobs",
            "label": label,
            "jobs": jobs,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            msg = f"resident SKFP stage jobs failed: {response}"
            raise RuntimeError(msg)
        report = response.get("report")
        if not isinstance(report, dict):
            msg = f"resident SKFP stage jobs returned invalid report: {response}"
            raise RuntimeError(msg)
        return report

    def run_job(
        self,
        *,
        clip_dir: pathlib.Path,
        out_root: pathlib.Path,
        clip_uuid: str,
        num_frames: int | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> pathlib.Path:
        if self._proc is None:
            self.start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        request: dict[str, Any] = {
            "clip_dir": str(clip_dir),
            "out_root": str(out_root),
            "clip_uuid": clip_uuid,
        }
        if num_frames is not None:
            request["num_frames"] = int(num_frames)
        if fps is not None:
            request["fps"] = float(fps)
        if width is not None:
            request["width"] = int(width)
        if height is not None:
            request["height"] = int(height)
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            msg = f"resident SKFP job failed: {response}"
            raise RuntimeError(msg)
        output_dir = response.get("output_dir")
        if output_dir:
            return pathlib.Path(str(output_dir))
        return out_root / clip_uuid / "skfp"

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                proc.stdin.flush()
                self._read_response()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to gracefully stop resident SKFP worker")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
            self._proc = None

    def _read_response(self) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            msg = "resident SKFP worker is not running"
            raise RuntimeError(msg)
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                code = self._proc.poll()
                msg = f"resident SKFP worker exited before response, returncode={code}"
                raise RuntimeError(msg)
            line = line.rstrip("\n")
            if line.startswith(RESPONSE_PREFIX):
                return json.loads(line[len(RESPONSE_PREFIX) :])
            logger.info(f"[resident-skfp] {line}")

    def _forward_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            logger.info(f"[resident-skfp] {line.rstrip()}")
