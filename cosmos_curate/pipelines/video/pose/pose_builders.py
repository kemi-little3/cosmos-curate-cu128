# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage builders for camera pose estimation."""

from __future__ import annotations

from typing import Literal

import attrs

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.pose.vipe_pose_stage import DEFAULT_VIPE_ADAPTER, VipePoseStage
from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_clip import VipeResidentClipStage
from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_window import VipeResidentWindowStage

VipeRunMode = Literal["subprocess-window", "resident-window", "resident-clip"]


@attrs.define(frozen=True)
class VipePoseConfig:
    """Configuration for ViPE per-window pose estimation."""

    vipe_python: str
    adapter_script: str = DEFAULT_VIPE_ADAPTER
    fail_policy: Literal["warn-only", "skip-window"] = "warn-only"
    run_mode: VipeRunMode = "subprocess-window"
    vipe_repo: str | None = None
    num_gpus_per_worker: float = 1.0
    num_workers_per_node: int = 0
    verbose: bool = False
    perf_profile: bool = False


def build_vipe_pose_stages(config: VipePoseConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build ViPE pose stages.

    If num_workers_per_node > 0, a fixed worker count is used, bypassing the
    xenna autoscaler's speed-estimation gate (which stalls when individual ViPE
    tasks take longer than autoscale_speed_estimation_window_duration_s).
    """
    if config.run_mode == "subprocess-window":
        stage = VipePoseStage(
            vipe_python=config.vipe_python,
            adapter_script=config.adapter_script,
            fail_policy=config.fail_policy,
            num_gpus_per_worker=config.num_gpus_per_worker,
            verbose=config.verbose,
            log_stats=config.perf_profile,
        )
    elif config.run_mode == "resident-window":
        stage = VipeResidentWindowStage(
            vipe_python=config.vipe_python,
            adapter_script=config.adapter_script,
            vipe_repo=config.vipe_repo,
            fail_policy=config.fail_policy,
            num_gpus_per_worker=config.num_gpus_per_worker,
            verbose=config.verbose,
            log_stats=config.perf_profile,
        )
    elif config.run_mode == "resident-clip":
        stage = VipeResidentClipStage(
            vipe_python=config.vipe_python,
            adapter_script=config.adapter_script,
            vipe_repo=config.vipe_repo,
            fail_policy=config.fail_policy,
            num_gpus_per_worker=config.num_gpus_per_worker,
            verbose=config.verbose,
            log_stats=config.perf_profile,
        )
    else:
        msg = f"Unsupported ViPE run mode: {config.run_mode}"
        raise ValueError(msg)
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]
