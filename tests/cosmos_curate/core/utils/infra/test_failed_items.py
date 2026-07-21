import json

import pytest

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curate.core.utils.infra import failed_items
from cosmos_curate.core.utils.infra.failed_items import item_failure_wrapper


class _Item(PipelineTask):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _PartialFailStage(CuratorStage):
    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        if len(tasks) > 1:
            raise RuntimeError("batch failed")
        item = tasks[0]
        if getattr(item, "session_id", "") == "bad":
            raise ValueError("bad item")
        return tasks


class _AllFailStage(CuratorStage):
    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        if len(tasks) > 1:
            raise RuntimeError("batch failed")
        raise ValueError("item failed")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_item_failure_wrapper_retries_items_and_records_failures(tmp_path, monkeypatch):
    failed_items_jsonl = tmp_path / "failed_items.jsonl"
    monkeypatch.setenv("FAILED_ITEMS_JSONL", str(failed_items_jsonl))
    monkeypatch.setattr(failed_items, "get_logging_client", lambda enable_loki=True: None)

    stage = item_failure_wrapper(_PartialFailStage())
    outputs = stage.process_data([_Item("good-1"), _Item("bad"), _Item("good-2")])

    assert [item.session_id for item in outputs or []] == ["good-1", "good-2"]
    records = _read_jsonl(failed_items_jsonl)
    assert len(records) == 1
    assert records[0]["stage"] == "_PartialFailStage"
    assert records[0]["item_id"] == "bad"
    assert records[0]["exception_type"] == "ValueError"


def test_item_failure_wrapper_reraises_when_every_item_fails(tmp_path, monkeypatch):
    failed_items_jsonl = tmp_path / "failed_items.jsonl"
    monkeypatch.setenv("FAILED_ITEMS_JSONL", str(failed_items_jsonl))
    monkeypatch.setattr(failed_items, "get_logging_client", lambda enable_loki=True: None)

    stage = item_failure_wrapper(_AllFailStage())

    with pytest.raises(RuntimeError, match="batch failed"):
        stage.process_data([_Item("bad-1"), _Item("bad-2")])

    assert len(_read_jsonl(failed_items_jsonl)) == 2


def test_item_failure_wrapper_skips_single_failed_item(tmp_path, monkeypatch):
    failed_items_jsonl = tmp_path / "failed_items.jsonl"
    monkeypatch.setenv("FAILED_ITEMS_JSONL", str(failed_items_jsonl))
    monkeypatch.setattr(failed_items, "get_logging_client", lambda enable_loki=True: None)

    stage = item_failure_wrapper(_AllFailStage())

    assert stage.process_data([_Item("bad")]) == []
    assert _read_jsonl(failed_items_jsonl)[0]["item_id"] == "bad"
