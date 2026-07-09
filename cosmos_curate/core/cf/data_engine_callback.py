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
"""Callback helpers for the Data Engine integration."""

from __future__ import annotations

import json

import requests
from loguru import logger

from cosmos_curate.core.cf.data_engine_adapter import DataEngineRequest


def build_callback_payload(
    *,
    pipeline_task_id: str,
    target_dataset_id: str,
    succeeded: bool,
    output_uri: str,
) -> dict[str, object]:
    """Build the callback payload required by Data Engine."""

    return {
        "pipeline_task_id": pipeline_task_id,
        "target_dataset_id": target_dataset_id,
        "succeeded": succeeded,
        "output_uri": output_uri,
    }


def send_callback(request: DataEngineRequest, *, succeeded: bool) -> None:
    """Send the terminal callback for a Data Engine request."""

    payload = build_callback_payload(
        pipeline_task_id=request.pipeline_task_id,
        target_dataset_id=request.args.target_dataset_id,
        succeeded=succeeded,
        output_uri=request.args.output_uri,
    )
    logger.info("Sending Data Engine callback to %s: %s", request.args.callback_url, json.dumps(payload))
    response = requests.post(request.args.callback_url, json=payload, timeout=60)
    response.raise_for_status()
