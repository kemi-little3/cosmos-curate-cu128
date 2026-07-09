# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident ViPE stage that runs one ViPE job per window."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
from typing import Literal

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.pose.vipe_pose_stage import DEFAULT_VIPE_ADAPTER
from cosmos_curate.pipelines.video.pose.vipe_resident_common import (
    ResidentVipeClient,
    attach_adapter_output_to_window,
    stage_window_input,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Window


class VipeResidentWindowStage(CuratorStage):
    """Keep a ViPE process alive, but run each window as an independent job."""

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

    def _process_window(self, window: Window, clip_uuid: str) -> None:
        if self._client is None:
            self.stage_setup()
        assert self._client is not None
        sample_id = f"{clip_uuid}_{window.start_frame}_{window.end_frame}"
        with tempfile.TemporaryDirectory(prefix="cosmos_vipe_pose_") as tmp:
            tmp_path = pathlib.Path(tmp)
            clip_dir = tmp_path / "input" / sample_id
            out_root = tmp_path / "output"
            stage_window_input(window, clip_dir)
            vipe_out = self._client.run_job(clip_dir=clip_dir, out_root=out_root, clip_uuid=sample_id)
            attach_adapter_output_to_window(
                window,
                clip_uuid=clip_uuid,
                vipe_out=vipe_out,
                quality_extra={"pipeline_run_mode": "resident-window"},
            )

    @nvtx.annotate("VipeResidentWindowStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Attach per-window resident ViPE pose arrays to every clip window."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process(len(task.video.clips)):
                for clip in task.video.clips:
                    for window in clip.windows:
                        try:
                            self._process_window(window, str(clip.uuid))
                        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                            error = str(exc)
                            logger.exception(
                                f"Resident ViPE window pose failed for clip {clip.uuid} window "
                                f"{window.start_frame}_{window.end_frame}: {error}"
                            )
                            self._mark_error(window, error)
                            if self._fail_policy == "skip-window":
                                window.caption_status = "skipped"
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
