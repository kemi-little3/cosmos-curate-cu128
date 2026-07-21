# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for best-effort item-level failure recording."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curate.core.utils.infra.logging_sdk import get_logging_client

FAILED_ITEMS_JSONL_ENV = "FAILED_ITEMS_JSONL"


def _short_text(value: object, limit: int = 2048) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _env_fields() -> dict[str, object]:
    fields: dict[str, object] = {}
    env_map = {
        "pipeline_id": "DATA_ENGINE_PIPELINE_ID",
        "pipeline_task_id": "DATA_ENGINE_PIPELINE_TASK_ID",
        "target_dataset_id": "DATA_ENGINE_TARGET_DATASET_ID",
        "output_uri": "DATA_ENGINE_OUTPUT_URI",
        "source_count": "DATA_ENGINE_SOURCE_COUNT",
    }
    for field, env_name in env_map.items():
        value = os.environ.get(env_name)
        if value:
            fields[field] = value
    return fields


def _task_fields(task: object) -> dict[str, object]:
    fields: dict[str, object] = {"item_type": task.__class__.__name__}

    session_id = getattr(task, "session_id", None)
    if session_id:
        fields["item_id"] = _short_text(session_id)

    videos = getattr(task, "videos", None)
    if isinstance(videos, list) and videos:
        video = videos[0]
        uri = getattr(video, "uri", None)
        input_video = getattr(video, "input_video", None)
        if uri:
            fields["video_uri"] = _short_text(uri)
        elif input_video:
            fields["video_uri"] = _short_text(input_video)
    elif hasattr(task, "video"):
        try:
            video = getattr(task, "video")
            input_video = getattr(video, "input_video", None)
            if input_video:
                fields["video_uri"] = _short_text(input_video)
        except Exception:
            pass

    if "item_id" not in fields:
        fields["item_id"] = _short_text(repr(task), limit=512)
    return fields


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _log_data_engine_item_failed(record: dict[str, object]) -> None:
    if "pipeline_task_id" not in record:
        return
    client = get_logging_client(enable_loki=True)
    if client is None:
        return
    try:
        client.warning("data_engine_item_failed", **record)
    except Exception:
        return


def record_failed_item(
    item: object,
    *,
    stage: str,
    exc: BaseException,
    action: str = "skipped",
    extra: dict[str, object] | None = None,
) -> None:
    """Record an item-level failure without interrupting the caller."""

    record: dict[str, object] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "action": action,
        "error": _short_text(exc),
        "exception_type": type(exc).__name__,
        **_env_fields(),
        **_task_fields(item),
    }
    if extra:
        record.update(extra)

    path_raw = os.environ.get(FAILED_ITEMS_JSONL_ENV)
    if path_raw:
        try:
            _append_jsonl(Path(path_raw), record)
        except Exception as write_exc:  # noqa: BLE001
            logger.warning(f"Failed to write failed item record to {path_raw}: {write_exc}")

    _log_data_engine_item_failed(record)
    logger.warning(
        "Skipped failed item at stage {}: item_id={}, error={}",
        stage,
        record.get("item_id"),
        record["error"],
    )


def _make_item_fallback_stage_class(stage_cls: type[CuratorStage]) -> type[CuratorStage]:
    base_name = stage_cls.__name__

    class _ItemFallbackStage(stage_cls):  # type: ignore[valid-type, misc]
        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask] | None:
            try:
                return super().process_data(tasks)  # type: ignore[no-any-return]
            except Exception as batch_exc:
                outputs: list[PipelineTask] = []
                successes = 0
                failures = 0
                for task in tasks:
                    try:
                        result = super().process_data([task])
                    except Exception as item_exc:  # noqa: BLE001
                        failures += 1
                        record_failed_item(task, stage=base_name, exc=item_exc)
                        continue
                    successes += 1
                    if result:
                        outputs.extend(result)

                if successes == 0 and len(tasks) > 1:
                    raise batch_exc
                logger.warning(
                    "Stage {} recovered from batch failure with item fallback: successes={}, failures={}",
                    base_name,
                    successes,
                    failures,
                )
                return outputs

    _ItemFallbackStage.__name__ = base_name
    _ItemFallbackStage.__qualname__ = base_name
    return _ItemFallbackStage


def item_failure_wrapper(stage: CuratorStage) -> CuratorStage:
    """Wrap a stage so batch failures fall back to per-item processing."""

    stage.__class__ = _make_item_fallback_stage_class(stage.__class__)
    return stage
