# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for resident ViPE pipeline stages."""

from __future__ import annotations

import json
import pathlib
import subprocess
import threading
from typing import Any

import numpy as np
from loguru import logger

from cosmos_curate.pipelines.video.pose.pose_utils import (
    align_pose_length,
    build_pose_meta,
    build_relative_pose,
    load_vipe_adapter_output,
)
from cosmos_curate.pipelines.video.utils.data_model import Clip, Window

RESPONSE_PREFIX = "__VIPE_RPC__ "


def probe_video(clip_mp4: pathlib.Path) -> tuple[int, int, float]:
    """Return width, height, fps for a staged mp4."""
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


def stage_window_input(window: Window, clip_dir: pathlib.Path) -> None:
    """Write a single window's mp4 bytes and adapter metadata to clip_dir."""
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
    width, height, fps = probe_video(clip_mp4)
    timestamps = np.arange(num_frames, dtype=np.float64) / fps if fps > 0 else np.arange(num_frames, dtype=np.float64)
    np.save(clip_dir / "frame_indices.npy", frame_indices)
    np.save(clip_dir / "timestamps.npy", timestamps)
    meta = {
        "source_mp4": str(clip_mp4),
        "original_resolution": [width, height],
        "original_fps": fps,
        "duration_s": num_frames / fps if fps > 0 else None,
        "num_raw_frames": num_frames,
        "sampled_fps": fps,
        "num_sampled": num_frames,
    }
    with (clip_dir / "meta.json").open("w") as f:
        json.dump(meta, f)


def stage_clip_input(clip: Clip, clip_dir: pathlib.Path) -> tuple[int, float]:
    """Write a full transcoded clip and adapter metadata to clip_dir."""
    mp4_data = clip.encoded_data.resolve()
    if mp4_data is None:
        msg = "clip has no encoded_data"
        raise ValueError(msg)
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(mp4_data.tobytes())

    if not clip.windows:
        msg = "clip has no windows"
        raise ValueError(msg)
    num_frames = max(window.end_frame for window in clip.windows) + 1
    frame_indices = np.arange(num_frames, dtype=np.int32)
    width, height, fps = probe_video(clip_mp4)
    timestamps = np.arange(num_frames, dtype=np.float64) / fps if fps > 0 else np.arange(num_frames, dtype=np.float64)
    np.save(clip_dir / "frame_indices.npy", frame_indices)
    np.save(clip_dir / "timestamps.npy", timestamps)
    meta = {
        "source_mp4": str(clip_mp4),
        "original_resolution": [width, height],
        "original_fps": fps,
        "duration_s": num_frames / fps if fps > 0 else None,
        "num_raw_frames": num_frames,
        "sampled_fps": fps,
        "num_sampled": num_frames,
    }
    with (clip_dir / "meta.json").open("w") as f:
        json.dump(meta, f)
    return num_frames, fps


def attach_adapter_output_to_window(
    window: Window,
    *,
    clip_uuid: str,
    vipe_out: pathlib.Path,
    quality_extra: dict[str, Any] | None = None,
) -> None:
    """Load an adapter output directory and attach arrays/meta to a window."""
    target_num_frames = window.end_frame - window.start_frame + 1
    adapter_data = load_vipe_adapter_output(vipe_out)
    intrinsics, poses_c2w, num_interpolated = align_pose_length(
        adapter_data["intrinsics"],
        adapter_data["poses_c2w"],
        target_num_frames,
    )
    relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
    quality = adapter_data.get("quality") or {}
    if quality_extra:
        quality = {**quality, **quality_extra}
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


def attach_clip_output_slice_to_window(
    window: Window,
    *,
    clip_uuid: str,
    clip_adapter_data: dict[str, Any],
) -> None:
    """Slice full-clip adapter arrays by window frame bounds and attach them."""
    start = window.start_frame
    end = window.end_frame + 1
    quality = dict(clip_adapter_data.get("quality") or {})
    quality["runner"] = "resident_clip_once_slice"
    quality["slice_source_window"] = {"start_frame": window.start_frame, "end_frame": window.end_frame}

    intrinsics = clip_adapter_data["intrinsics"][start:end]
    poses_c2w = clip_adapter_data["poses_c2w"][start:end]
    if poses_c2w.shape[0] != window.end_frame - window.start_frame + 1:
        msg = f"clip-once output too short for window {window.start_frame}_{window.end_frame}"
        raise ValueError(msg)

    relative_poses, scale_factor, total_path = build_relative_pose(poses_c2w)
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
        num_frames=window.end_frame - window.start_frame + 1,
        status=status,
        translation_scale_factor=scale_factor,
        total_path_length_before_norm=total_path,
        quality=quality,
        num_interpolated_frames=0,
    )


class ResidentVipeClient:
    """JSONL stdin/stdout client for the external resident ViPE worker."""

    def __init__(
        self,
        *,
        vipe_python: str,
        adapter_script: str,
        vipe_repo: str,
        verbose: bool = False,
    ) -> None:
        self._vipe_python = vipe_python
        self._adapter_script = adapter_script
        self._vipe_repo = vipe_repo
        self._verbose = verbose
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        worker_script = pathlib.Path(__file__).with_name("vipe_resident_rpc_worker.py")
        cmd = [
            self._vipe_python,
            str(worker_script),
            "--adapter-script",
            self._adapter_script,
            "--vipe-repo",
            self._vipe_repo,
        ]
        if self._verbose:
            logger.info(f"Starting resident ViPE worker: {' '.join(cmd)}")
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
        if ready.get("status") not in {"ready"}:
            msg = f"resident ViPE worker failed to start: {ready}"
            raise RuntimeError(msg)

    def run_job(self, *, clip_dir: pathlib.Path, out_root: pathlib.Path, clip_uuid: str) -> pathlib.Path:
        if self._proc is None:
            self.start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        request = {
            "clip_dir": str(clip_dir),
            "out_root": str(out_root),
            "clip_uuid": clip_uuid,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            msg = f"resident ViPE job failed: {response}"
            raise RuntimeError(msg)
        return out_root / clip_uuid / "vipe"

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
            logger.warning("Failed to gracefully stop resident ViPE worker")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
            self._proc = None

    def _read_response(self) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            msg = "resident ViPE worker is not running"
            raise RuntimeError(msg)
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                code = self._proc.poll()
                msg = f"resident ViPE worker exited before response, returncode={code}"
                raise RuntimeError(msg)
            line = line.rstrip("\n")
            if line.startswith(RESPONSE_PREFIX):
                return json.loads(line[len(RESPONSE_PREFIX) :])
            logger.info(f"[resident-vipe] {line}")

    def _forward_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            logger.info(f"[resident-vipe] {line.rstrip()}")

