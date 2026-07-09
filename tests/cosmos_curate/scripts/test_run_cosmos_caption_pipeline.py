# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the caption-input pipeline helper script."""

from pathlib import Path
from types import SimpleNamespace

from scripts.run_cosmos_caption_pipeline import build_payload, copy_stage_inputs


def test_build_payload_uses_low_resource_stage_save_configuration(tmp_path: Path) -> None:
    """The helper should request only the caption-stage inputs through the lightest pipeline path."""
    args = SimpleNamespace(
        input_video_path="/tmp/samples",
        limit=0,
        execution_mode="BATCH",
        motion_filter="disable",
        openai_model_name="Qwen3.6-27B",
        openai_caption_retries=1,
        openai_retry_delay_seconds=1,
        stage_name="OpenAICaptionStage",
        api_caption_batch_size=1,
    )

    payload = build_payload(args, tmp_path / "outputs" / "run_01")

    assert payload["pipeline"] == "split"
    request_args = payload["args"]
    assert request_args["limit"] == 0
    assert request_args["execution_mode"] == "BATCH"
    assert request_args["splitting_algorithm"] == "fixed-stride"
    assert request_args["fixed_stride_split_duration"] == 10
    assert request_args["motion_filter"] == "disable"
    assert request_args["captioning_algorithm"] == "openai"
    assert request_args["stage_save"] == ["OpenAICaptionStage"]
    assert request_args["api_caption_batch_size"] == 1
    assert request_args["api_caption_num_workers_per_node"] == 1


def test_copy_stage_inputs_replaces_existing_destination(tmp_path: Path) -> None:
    """Existing copied inputs should be replaced with the latest saved tasks."""
    stage_output_dir = tmp_path / "outputs" / "tasks" / "OpenAICaptionStage"
    stage_output_dir.mkdir(parents=True)
    (stage_output_dir / "new.pkl").write_bytes(b"new")

    stage_input_dir = tmp_path / "inputs" / "tasks" / "OpenAICaptionStage"
    stage_input_dir.mkdir(parents=True)
    (stage_input_dir / "old.pkl").write_bytes(b"old")

    copy_stage_inputs(stage_output_dir, stage_input_dir)

    assert (stage_input_dir / "new.pkl").read_bytes() == b"new"
    assert not (stage_input_dir / "old.pkl").exists()
