"""Smoke tests for the Data Engine integration path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosmos_curate.core.cf import data_engine_callback
from cosmos_curate.core.cf.data_engine_adapter import build_kemi_workflow_shell_args, parse_data_engine_request


def test_parse_data_engine_request_smoke() -> None:
    raw = {
        "pipeline": "pack_dataset_tars",
        "pipeline_id": "pipeline-123",
        "pipeline_task_id": "task-456",
        "args": {
            "source_uris": [
                "s3://data-lake/videos/a.mp4",
                "s3://data-lake/videos/b.mp4",
            ],
            "output_uri": "s3://data-lake/datasets/out-001",
            "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
            "target_dataset_id": "dataset-789",
            "tar_count": 1,
            "probe_manifest_only": True,
        },
    }

    request = parse_data_engine_request(raw)

    assert request.pipeline == "pack_dataset_tars"
    assert request.pipeline_id == "pipeline-123"
    assert request.pipeline_task_id == "task-456"
    assert request.args.source_uris == [
        "s3://data-lake/videos/a.mp4",
        "s3://data-lake/videos/b.mp4",
    ]
    assert request.args.output_uri == "s3://data-lake/datasets/out-001"
    assert request.args.callback_url == "http://127.0.0.1:19090/datasets/process/callback"
    assert request.args.target_dataset_id == "dataset-789"
    assert request.args.tar_count == 1
    assert request.args.probe_manifest_only is True


def test_parse_data_engine_request_defaults_probe_manifest_only_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_ENGINE_PROBE_MANIFEST_ONLY_DEFAULT", raising=False)
    raw = {
        "pipeline": "pack_dataset_tars",
        "pipeline_id": "pipeline-123",
        "pipeline_task_id": "task-456",
        "args": {
            "source_uris": ["s3://data-lake/videos/a.mp4"],
            "output_uri": "s3://data-lake/datasets/out-001",
            "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
            "target_dataset_id": "dataset-789",
        },
    }

    request = parse_data_engine_request(raw)

    assert request.args.probe_manifest_only is True


def test_parse_data_engine_request_can_disable_default_probe_manifest_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ENGINE_PROBE_MANIFEST_ONLY_DEFAULT", "0")
    raw = {
        "pipeline": "pack_dataset_tars",
        "pipeline_id": "pipeline-123",
        "pipeline_task_id": "task-456",
        "args": {
            "source_uris": ["s3://data-lake/videos/a.mp4"],
            "output_uri": "s3://data-lake/datasets/out-001",
            "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
            "target_dataset_id": "dataset-789",
        },
    }

    request = parse_data_engine_request(raw)

    assert request.args.probe_manifest_only is False


def test_build_callback_payload_smoke() -> None:
    payload = data_engine_callback.build_callback_payload(
        pipeline_task_id="task-456",
        pipeline_id="pipeline-123",
        target_dataset_id="dataset-789",
        succeeded=True,
        output_uri="s3://data-lake/datasets/out-001",
        total_videos=2,
        success_videos=1,
    )

    assert payload == {
        "code": 0,
        "message": "success",
        "data": {
            "pipeline_task_id": "task-456",
            "pipeline_id": "pipeline-123",
            "target_dataset_id": "dataset-789",
            "succeeded": True,
            "output_uri": "s3://data-lake/datasets/out-001",
            "total_videos": 2,
            "success_videos": 1,
        },
    }


def test_build_invalid_request_response_includes_error_type() -> None:
    payload = data_engine_callback.build_invalid_request_response(
        {
            "pipeline": "pack_dataset_tars",
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task-456",
            "args": {
                "target_dataset_id": "dataset-789",
            },
        },
        "source_uris is empty",
    )

    assert payload == {
        "code": -1,
        "message": "invalid request",
        "data": {
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task-456",
            "target_dataset_id": "dataset-789",
            "error": "source_uris is empty",
            "error_type": "invalid_request",
        },
    }


def test_build_kemi_workflow_shell_args_writes_pipeline_input(tmp_path: Path) -> None:
    request = parse_data_engine_request(
        {
            "pipeline": "pack_dataset_tars",
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task/456",
            "args": {
                "source_uris": [
                    "https://data-engine-test.tos/videos/a.mp4",
                    "https://data-engine-test.tos/videos/b.mp4",
                ],
                "output_uri": "s3://data-lake/datasets/out-001",
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "target_dataset_id": "dataset-789",
                "tar_count": 1,
            },
        },
    )

    args = build_kemi_workflow_shell_args(request, workspace_prefix=tmp_path)

    assert args.OUTPUT_PREFIX == "task_456"
    assert args.BATCH_SIZE == "400"
    assert args.RUN_SHARD == "1"
    assert args.SHARD_OUTPUT_DATASET_PATH == "s3://data-lake/datasets/out-001"
    assert args.SHARD_TARGET_TAR_COUNT == "1"
    assert args.GENERATE_T5_EMBEDDINGS == "0"

    input_path = Path(args.RAW_INPUT_VIDEO_LIST_JSON)
    assert input_path == tmp_path / "input" / "data_engine" / "task_456.json"
    assert json.loads(input_path.read_text(encoding="utf-8")) == request.args.source_uris


def test_send_callback_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        recorded["url"] = url
        recorded["json"] = json
        recorded["timeout"] = timeout
        return _FakeResponse()

    log_events: list[tuple[str, str, dict[str, object]]] = []

    def fake_log_event(level: str, event: str, request: object, **fields: object) -> None:
        log_events.append((level, event, fields))

    monkeypatch.setattr(data_engine_callback.requests, "post", fake_post)
    monkeypatch.setattr(data_engine_callback, "log_data_engine_event", fake_log_event)

    request = parse_data_engine_request(
        {
            "pipeline": "pack_dataset_tars",
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task-456",
            "args": {
                "source_uris": ["s3://data-lake/videos/a.mp4"],
                "output_uri": "s3://data-lake/datasets/out-001",
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "target_dataset_id": "dataset-789",
                "tar_count": 1,
            },
        },
    )

    data_engine_callback.send_callback(request, succeeded=True)

    assert recorded["url"] == "http://127.0.0.1:19090/datasets/process/callback"
    assert recorded["timeout"] == 60
    assert recorded["json"] == {
        "code": 0,
        "message": "success",
        "data": {
            "pipeline_task_id": "task-456",
            "pipeline_id": "pipeline-123",
            "target_dataset_id": "dataset-789",
            "succeeded": True,
            "output_uri": "s3://data-lake/datasets/out-001",
            "total_videos": 1,
            "success_videos": 1,
        },
    }
    assert log_events == [
        (
            "info",
            "data_engine_callback_sent",
            {
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "callback_succeeded": True,
                "callback_payload_code": 0,
                "callback_status_code": None,
                "total_videos": 1,
                "success_videos": 1,
            },
        )
    ]


def test_send_callback_logs_failure_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    log_events: list[tuple[str, str, dict[str, object]]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> object:
        raise RuntimeError("callback down")

    def fake_log_event(level: str, event: str, request: object, **fields: object) -> None:
        log_events.append((level, event, fields))

    monkeypatch.setattr(data_engine_callback.requests, "post", fake_post)
    monkeypatch.setattr(data_engine_callback, "log_data_engine_event", fake_log_event)

    request = parse_data_engine_request(
        {
            "pipeline": "pack_dataset_tars",
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task-456",
            "args": {
                "source_uris": ["s3://data-lake/videos/a.mp4", "s3://data-lake/videos/b.mp4"],
                "output_uri": "s3://data-lake/datasets/out-001",
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "target_dataset_id": "dataset-789",
                "tar_count": 1,
            },
        },
    )

    with pytest.raises(RuntimeError, match="callback down"):
        data_engine_callback.send_callback(request, succeeded=False, total_videos=2, success_videos=1)

    assert log_events == [
        (
            "error",
            "data_engine_callback_failed",
            {
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "callback_succeeded": False,
                "callback_payload_code": -1,
                "total_videos": 2,
                "success_videos": 1,
                "error": "callback down",
                "exception_type": "RuntimeError",
            },
        )
    ]
