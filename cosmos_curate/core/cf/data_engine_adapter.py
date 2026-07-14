# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for adapting Data Engine requests to the existing Cosmos pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True, slots=True)
class DataEngineArgs:
    """The subset of Data Engine args that the adapter cares about."""

    source_uris: list[str]
    output_uri: str
    callback_url: str
    target_dataset_id: str
    tar_count: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class DataEngineRequest:
    """Normalized Data Engine request payload."""

    pipeline: str
    pipeline_id: str
    pipeline_task_id: str
    args: DataEngineArgs


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "data-engine-task"


def _require_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Missing or invalid required field: {key}"
        raise ValueError(msg)
    return value


def _require_list_of_str(raw: Mapping[str, object], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = f"Missing or invalid required list field: {key}"
        raise ValueError(msg)
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            msg = f"Invalid item at {key}[{idx}]"
            raise ValueError(msg)
        out.append(item)
    if not out:
        msg = f"Field {key} must not be empty"
        raise ValueError(msg)
    return out


def _optional_positive_int(raw: Mapping[str, object], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        msg = f"Invalid integer field: {key}"
        raise ValueError(msg)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid integer field: {key}"
        raise ValueError(msg) from exc
    if parsed < 1:
        msg = f"Field {key} must be positive"
        raise ValueError(msg)
    return parsed


def parse_data_engine_request(raw: Mapping[str, object]) -> DataEngineRequest:
    """Validate and normalize a Data Engine request body."""

    args_raw = raw.get("args")
    if not isinstance(args_raw, Mapping):
        msg = "Missing or invalid required field: args"
        raise ValueError(msg)

    return DataEngineRequest(
        pipeline=_require_str(raw, "pipeline"),
        pipeline_id=_require_str(raw, "pipeline_id"),
        pipeline_task_id=_require_str(raw, "pipeline_task_id"),
        args=DataEngineArgs(
            source_uris=_require_list_of_str(args_raw, "source_uris"),
            output_uri=_require_str(args_raw, "output_uri"),
            callback_url=_require_str(args_raw, "callback_url"),
            target_dataset_id=_require_str(args_raw, "target_dataset_id"),
            tar_count=_optional_positive_int(args_raw, "tar_count", 1),
        ),
    )


def write_data_engine_source_list(request: DataEngineRequest, input_dir: Path) -> Path:
    """Write source URIs in the JSON-list format used by the Kemi workflow."""

    input_dir.mkdir(parents=True, exist_ok=True)
    output_path = input_dir / f"{_safe_path_component(request.pipeline_task_id)}.json"
    output_path.write_text(
        json.dumps(request.args.source_uris, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_kemi_workflow_shell_args(
    request: DataEngineRequest,
    *,
    workspace_prefix: Path,
) -> argparse.Namespace:
    """Convert a Data Engine request into args accepted by kemi-workflow-shell."""

    input_json_path = write_data_engine_source_list(
        request,
        workspace_prefix / "input" / "data_engine",
    )

    return argparse.Namespace(
        OUTPUT_PREFIX=_safe_path_component(request.pipeline_task_id),
        RAW_INPUT_VIDEO_LIST_JSON=str(input_json_path),
        BATCH_SIZE="0",
        RUN_SHARD="1",
        SHARD_OUTPUT_DATASET_PATH=request.args.output_uri,
        SHARD_TARGET_TAR_COUNT=str(request.args.tar_count),
        GENERATE_T5_EMBEDDINGS="0",
    )
