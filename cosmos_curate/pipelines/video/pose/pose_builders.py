# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage builders for camera pose estimation."""

from __future__ import annotations

from typing import Literal

import attrs

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.pose.skfp_pose_stage_resident_clip import SkfpResidentClipStage
from cosmos_curate.pipelines.video.pose.skfp_pose_stage_resident_window import SkfpResidentWindowStage
from cosmos_curate.pipelines.video.pose.vipe_pose_stage import DEFAULT_VIPE_ADAPTER
from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_clip import VipeResidentClipStage
from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_window import VipeResidentWindowStage

VipeRunMode = Literal["resident-window", "resident-clip"]
SkfpRunMode = Literal["resident-window", "resident-clip"]


@attrs.define(frozen=True)
class VipePoseConfig:
    """Configuration for ViPE per-window pose estimation."""

    vipe_python: str
    adapter_script: str = DEFAULT_VIPE_ADAPTER
    fail_policy: Literal["warn-only", "skip-window"] = "warn-only"
    run_mode: VipeRunMode = "resident-clip"
    vipe_repo: str | None = None
    num_gpus_per_worker: float = 1.0
    num_workers_per_node: int = 0
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class SkfpPoseConfig:
    """Configuration for resident SparseKeyframePose ViPE-keyframe estimation."""

    skfp_root: str
    vipe_python: str
    vipe_adapter_script: str
    vipe_work_root: str | None = None
    run_mode: SkfpRunMode = "resident-window"
    stride: int = 32
    min_keyframes: int = 3
    max_keyframes: int = 0
    fail_policy: Literal["warn-only", "skip-window"] = "warn-only"
    num_gpus_per_worker: float = 1.0
    num_workers_per_node: int = 0
    verbose: bool = False
    perf_profile: bool = False


def build_skfp_pose_stages(config: SkfpPoseConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build resident SKFP pose stages.

    Stage 1 wires configuration only; resident worker runtime lands in later
    phases. Both modes reserve resources exactly like the ViPE resident stages.
    """
    stage_kwargs = {
        "skfp_root": config.skfp_root,
        "vipe_python": config.vipe_python,
        "vipe_adapter_script": config.vipe_adapter_script,
        "vipe_work_root": config.vipe_work_root,
        "stride": config.stride,
        "min_keyframes": config.min_keyframes,
        "max_keyframes": config.max_keyframes,
        "fail_policy": config.fail_policy,
        "num_gpus_per_worker": config.num_gpus_per_worker,
        "verbose": config.verbose,
        "log_stats": config.perf_profile,
    }
    if config.run_mode == "resident-window":
        stage = SkfpResidentWindowStage(**stage_kwargs)
    elif config.run_mode == "resident-clip":
        stage = SkfpResidentClipStage(**stage_kwargs)
    else:
        msg = f"Unsupported SKFP run mode: {config.run_mode}"
        raise ValueError(msg)
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]


def build_vipe_pose_stages(config: VipePoseConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build the resident ViPE pose stage."""
    stage_kwargs = {
        "vipe_python": config.vipe_python,
        "adapter_script": config.adapter_script,
        "vipe_repo": config.vipe_repo,
        "fail_policy": config.fail_policy,
        "num_gpus_per_worker": config.num_gpus_per_worker,
        "verbose": config.verbose,
        "log_stats": config.perf_profile,
    }
    if config.run_mode == "resident-window":
        stage = VipeResidentWindowStage(**stage_kwargs)
    elif config.run_mode == "resident-clip":
        stage = VipeResidentClipStage(**stage_kwargs)
    else:
        msg = f"Unsupported ViPE run mode: {config.run_mode}"
        raise ValueError(msg)
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]
