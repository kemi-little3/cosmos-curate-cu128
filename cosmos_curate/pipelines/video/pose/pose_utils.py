# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for per-window camera pose estimation outputs."""

from __future__ import annotations

import json
import pathlib
import uuid
from typing import Any

import numpy as np
import numpy.typing as npt

VIPE_TO_GT_AXIS_TRANSFORM_NAME = "x=x,y=z,z=-y"
VIPE_TO_GT_AXIS_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def sample_id_from_window(clip_uuid: uuid.UUID | str, start_frame: int, end_frame: int) -> str:
    """Return the stable Cosmos-Predict sample id for a clip window."""
    return f"{clip_uuid}_{start_frame}_{end_frame}"


def invert_se3_batch(poses: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert a batch of SE(3) matrices using the rigid transform inverse."""
    out = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    rot_inv = np.swapaxes(rot, 1, 2)
    out[:, :3, :3] = rot_inv
    out[:, :3, 3] = -np.einsum("tij,tj->ti", rot_inv, trans)
    return out


def relative_to_first(poses_c2w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert absolute c2w poses to poses relative to the first frame."""
    if poses_c2w.shape[0] == 0:
        return poses_c2w.copy()
    first_w2c = invert_se3_batch(poses_c2w[:1])[0]
    rel = np.einsum("ij,tjk->tik", first_w2c, poses_c2w)
    rel[0] = np.eye(4, dtype=np.float64)
    return rel


def apply_world_axis_transform(
    poses_c2w: npt.NDArray[np.float64],
    axis_transform: npt.NDArray[np.float64] = VIPE_TO_GT_AXIS_TRANSFORM,
) -> npt.NDArray[np.float64]:
    """Apply a fixed basis transform to relative c2w poses.

    Relative pose rotations are conjugated so the first-frame identity remains
    identity in the canonical label space. Translations are expressed in the new
    basis with the same axis transform.
    """
    out = poses_c2w.copy()
    out[:, :3, :3] = np.einsum("ij,tjk,kl->til", axis_transform, poses_c2w[:, :3, :3], axis_transform.T)
    out[:, :3, 3] = poses_c2w[:, :3, 3] @ axis_transform.T
    return out


def normalize_translation(
    relative_poses: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Normalize relative-pose translations by max translation norm.

    Returns:
        (normalized_poses, scale_factor, total_path_length_before_norm)

    """
    out = relative_poses.copy()
    if out.shape[0] == 0:
        return out, 1.0, 0.0
    translations = out[:, :3, 3]
    norms = np.linalg.norm(translations, axis=1)
    scale_factor = float(norms.max()) if norms.size else 1.0
    deltas = np.diff(translations, axis=0)
    total_path = float(np.linalg.norm(deltas, axis=1).sum()) if deltas.size else 0.0
    if scale_factor > 0:
        out[:, :3, 3] = translations / scale_factor
    else:
        scale_factor = 1.0
    return out, scale_factor, total_path


def build_relative_pose(
    poses_c2w: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Build canonical relative poses from absolute ViPE OpenCV c2w poses.

    Translations are NOT normalized; the max-translation-norm (scale_factor) is
    returned for reference and stored in pose_meta, but the poses themselves retain
    their original ViPE-estimated scale so downstream consumers can use speed info.
    """
    relative = relative_to_first(poses_c2w)
    canonical = apply_world_axis_transform(relative)
    if canonical.shape[0] == 0:
        return canonical, 1.0, 0.0
    translations = canonical[:, :3, 3]
    norms = np.linalg.norm(translations, axis=1)
    scale_factor = float(norms.max()) if norms.size else 1.0
    if scale_factor <= 0:
        scale_factor = 1.0
    deltas = np.diff(translations, axis=0)
    total_path = float(np.linalg.norm(deltas, axis=1).sum()) if deltas.size else 0.0
    return canonical, scale_factor, total_path


def load_vipe_adapter_output(out_dir: pathlib.Path) -> dict[str, Any]:
    """Load unified adapter files emitted by the external ViPE adapter."""
    out_dir = pathlib.Path(out_dir)
    quality_path = out_dir / "quality.json"
    quality: dict[str, Any] = {}
    if quality_path.exists():
        with quality_path.open() as f:
            quality = json.load(f)
    poses_path = out_dir / "poses_c2w.npy"
    if not poses_path.exists():
        poses_path = out_dir / "poses.npy"
    rel_path = out_dir / "relative_poses.npy"
    return {
        "intrinsics": np.load(out_dir / "intrinsics.npy").astype(np.float32),
        "poses_c2w": np.load(poses_path).astype(np.float64),
        "relative_poses": np.load(rel_path).astype(np.float64) if rel_path.exists() else None,
        "quality": quality,
    }


def align_pose_length(
    intrinsics: npt.NDArray[np.float32],
    poses_c2w: npt.NDArray[np.float64],
    target_num_frames: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64], int]:
    """Align pose arrays to target frame count with nearest-neighbor indexing.

    This is a conservative fallback for adapter outputs that are not already the
    same length as the window. The external adapter should normally emit the
    target length because it uses the staged window frame indices.
    """
    src_len = poses_c2w.shape[0]
    if src_len == target_num_frames:
        return intrinsics, poses_c2w, 0
    if src_len == 0:
        msg = "Cannot align empty pose output"
        raise ValueError(msg)
    src_idx = np.linspace(0, src_len - 1, target_num_frames)
    nearest = np.rint(src_idx).astype(np.int64)
    return intrinsics[nearest].astype(np.float32), poses_c2w[nearest].astype(np.float64), target_num_frames


def build_pose_meta(
    *,
    clip_uuid: str,
    start_frame: int,
    end_frame: int,
    num_frames: int,
    status: str,
    translation_scale_factor: float,
    total_path_length_before_norm: float,
    quality: dict[str, Any] | None = None,
    num_interpolated_frames: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the JSON metadata sidecar for a per-window pose sample."""
    warnings = []
    if quality:
        warnings = list(quality.get("warnings", []))
    data: dict[str, Any] = {
        "method": "vipe",
        "status": status,
        "pose_convention": "opencv_c2w",
        "label_source": "vipe",
        "camera_coordinate": "opencv",
        "canonical_coordinate": "gt_relative_c2w",
        "axis_transform": VIPE_TO_GT_AXIS_TRANSFORM_NAME,
        "axis_transform_matrix": VIPE_TO_GT_AXIS_TRANSFORM.astype(int).tolist(),
        "axis_transform_application": {
            "rotation": "R_new = R_axis @ R_old @ R_axis.T",
            "translation": "t_new = R_axis @ t_old",
        },
        "translation_normalization": "none",
        "metric_scale_used": False,
        "confidence": "pseudo",
        "intrinsics_format": "fx_fy_cx_cy",
        "num_frames": num_frames,
        "source_window": {
            "clip_uuid": clip_uuid,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "outputs": {
            "intrinsics": "intrinsics.npy",
            "poses": "poses.npy",
            "relative_poses": "relative_poses.npy",
        },
        "relative_pose_rule": {
            "first_frame_identity": True,
            "translation_normalized": False,
            "translation_normalization": "none",
            "translation_scale_factor": translation_scale_factor,
            "total_path_length_before_norm": total_path_length_before_norm,
            "axis_transform": VIPE_TO_GT_AXIS_TRANSFORM_NAME,
        },
        "quality": {
            "status": status,
            "warnings": warnings,
            "num_interpolated_frames": num_interpolated_frames,
        },
    }
    if quality:
        data["quality"]["adapter_quality"] = quality
    if error:
        data["error"] = error
    return data
