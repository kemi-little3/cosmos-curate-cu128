"""Smoke tests for the Data Engine integration path."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cosmos_curate.core.cf import data_engine_callback, data_engine_packager
from cosmos_curate.core.cf.data_engine_adapter import parse_data_engine_request


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


def test_build_callback_payload_smoke() -> None:
    payload = data_engine_callback.build_callback_payload(
        pipeline_task_id="task-456",
        target_dataset_id="dataset-789",
        succeeded=True,
        output_uri="s3://data-lake/datasets/out-001",
    )

    assert payload == {
        "pipeline_task_id": "task-456",
        "target_dataset_id": "dataset-789",
        "succeeded": True,
        "output_uri": "s3://data-lake/datasets/out-001",
    }


def test_pack_source_uris_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOutputClient:
        def __init__(self) -> None:
            self.writes: list[tuple[str, bytes]] = []

        def object_exists(self, _dest: Any) -> bool:
            return False

        def upload_bytes(self, dest: Any, data: bytes) -> None:
            self.writes.append((str(dest), data))

    class _FakeInputClient:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def download_object_as_bytes(self, _uri: Any) -> bytes:
            return self._payload

    fake_output_client = _FakeOutputClient()
    payloads = {
        "s3://data-lake/videos/a.mp4": b"video-a",
        "s3://data-lake/videos/b.mp4": b"video-b",
    }

    def fake_get_storage_client(target_path: str, **_kwargs: Any) -> Any:
        if target_path.startswith("s3://data-lake/datasets/out-001"):
            return fake_output_client
        if target_path in payloads:
            return _FakeInputClient(payloads[target_path])
        raise AssertionError(f"unexpected storage target: {target_path}")

    monkeypatch.setattr(data_engine_packager, "get_storage_client", fake_get_storage_client)
    monkeypatch.setattr(data_engine_packager, "read_bytes", lambda uri, client=None: client.download_object_as_bytes(uri))

    request = parse_data_engine_request(
        {
            "pipeline": "pack_dataset_tars",
            "pipeline_id": "pipeline-123",
            "pipeline_task_id": "task-456",
            "args": {
                "source_uris": list(payloads.keys()),
                "output_uri": "s3://data-lake/datasets/out-001",
                "callback_url": "http://127.0.0.1:19090/datasets/process/callback",
                "target_dataset_id": "dataset-789",
            },
        },
    )

    data_engine_packager.pack_source_uris(request)

    written_names = [name for name, _ in fake_output_client.writes]
    assert written_names == [
        "s3://data-lake/datasets/out-001/batch-000001.tar",
        "s3://data-lake/datasets/out-001/batch-000001.json",
    ]

    tar_bytes = fake_output_client.writes[0][1]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
        assert sorted(tar.getnames()) == ["000000_a.mp4", "000001_b.mp4"]
        assert tar.extractfile("000000_a.mp4").read() == b"video-a"
        assert tar.extractfile("000001_b.mp4").read() == b"video-b"

    manifest = json.loads(fake_output_client.writes[1][1].decode("utf-8"))
    assert manifest["tar_file"] == "batch-000001.tar"
    assert manifest["file_count"] == 2
    assert [entry["source_uri"] for entry in manifest["files"]] == list(payloads.keys())


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

    monkeypatch.setattr(data_engine_callback.requests, "post", fake_post)

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
            },
        },
    )

    data_engine_callback.send_callback(request, succeeded=True)

    assert recorded["url"] == "http://127.0.0.1:19090/datasets/process/callback"
    assert recorded["timeout"] == 60
    assert recorded["json"] == {
        "pipeline_task_id": "task-456",
        "target_dataset_id": "dataset-789",
        "succeeded": True,
        "output_uri": "s3://data-lake/datasets/out-001",
    }
