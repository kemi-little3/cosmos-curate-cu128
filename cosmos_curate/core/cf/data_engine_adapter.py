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

import dataclasses
from collections.abc import Mapping, Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class DataEngineArgs:
    """The subset of Data Engine args that the adapter cares about."""

    source_uris: list[str]
    output_uri: str
    callback_url: str
    target_dataset_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class DataEngineRequest:
    """Normalized Data Engine request payload."""

    pipeline: str
    pipeline_id: str
    pipeline_task_id: str
    args: DataEngineArgs


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
        ),
    )
