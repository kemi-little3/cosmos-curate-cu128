# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident ViPE stage that runs once per clip and slices results per window."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
from typing import Literal

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.pose.pose_utils import load_vipe_adapter_output
from cosmos_curate.pipelines.video.pose.vipe_pose_stage import DEFAULT_VIPE_ADAPTER
from cosmos_curate.pipelines.video.pose.vipe_resident_common import (
    ResidentVipeClient,
    attach_clip_output_slice_to_window,
    stage_clip_input,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Window


class VipeResidentClipStage(CuratorStage):
    """Keep a ViPE process alive, run each transcoded clip once, and slice windows."""

    def __init__(
        self,
        vipe_python: str,
        adapter_script: str = DEFAULT_VIPE_ADAPTER,
        vipe_repo: str | None = None,
        fail_policy: Literal["warn-only", "skip-window"] = "warn-only",
        num_gpus_per_worker: float = 1.0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._vipe_python = vipe_python
        self._adapter_script = adapter_script
        self._vipe_repo = vipe_repo or str(pathlib.Path(adapter_script).resolve().parents[2] / "repo" / "vipe")
        self._fail_policy = fail_policy
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)
        self._client: ResidentVipeClient | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Reserve GPU resources for the resident ViPE subprocess."""
        return CuratorStageResource(cpus=1.0, gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Start the external ViPE Python worker once per actor."""
        self._client = ResidentVipeClient(
            vipe_python=self._vipe_python,
            adapter_script=self._adapter_script,
            vipe_repo=self._vipe_repo,
            verbose=self._verbose,
        )
        self._client.start()

    def destroy(self) -> None:
        """Stop the external ViPE Python worker."""
        if self._client is not None:
            self._client.stop()
            self._client = None

    def _mark_error(self, window: Window, error: str) -> None:
        window.pose_status = "error"
        window.pose_error = error
        window.errors["vipe_pose"] = error

    def _process_clip(self, clip_uuid: str, clip) -> None:
        if self._client is None:
            self.stage_setup()
        assert self._client is not None
        sample_id = f"{clip_uuid}_full_clip"
        with tempfile.TemporaryDirectory(prefix="cosmos_vipe_pose_clip_") as tmp:
            tmp_path = pathlib.Path(tmp)
            clip_dir = tmp_path / "input" / sample_id
            out_root = tmp_path / "output"
            stage_clip_input(clip, clip_dir)
            vipe_out = self._client.run_job(clip_dir=clip_dir, out_root=out_root, clip_uuid=sample_id)
            clip_adapter_data = load_vipe_adapter_output(vipe_out)

        for window in clip.windows:
            attach_clip_output_slice_to_window(
                window,
                clip_uuid=clip_uuid,
                clip_adapter_data=clip_adapter_data,
            )

    @nvtx.annotate("VipeResidentClipStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Attach clip-once resident ViPE pose slices to every clip window."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process(len(task.video.clips)):
                for clip in task.video.clips:
                    try:
                        self._process_clip(str(clip.uuid), clip)
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                        error = str(exc)
                        logger.exception(f"Resident ViPE clip pose failed for clip {clip.uuid}: {error}")
                        for window in clip.windows:
                            self._mark_error(window, error)
                            if self._fail_policy == "skip-window":
                                window.caption_status = "skipped"
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
