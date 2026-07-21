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
from collections.abc import Mapping
from typing import Any

import requests
from loguru import logger

from cosmos_curate.core.cf.data_engine_adapter import DataEngineRequest
from cosmos_curate.core.utils.infra.logging_sdk import get_logging_client


def log_data_engine_event(level: str, event: str, request: DataEngineRequest, **fields: object) -> None:
    """Write a structured Data Engine business event to the logging SDK."""

    client = get_logging_client(enable_loki=True)
    if client is None:
        return

    payload = {
        "pipeline_id": request.pipeline_id,
        "pipeline_task_id": request.pipeline_task_id,
        "target_dataset_id": request.args.target_dataset_id,
        "output_uri": request.args.output_uri,
        "source_count": len(request.args.source_uris),
        **fields,
    }

    try:
        log_fn: Any = getattr(client, level, None) or client.info
        log_fn(event, **payload)
    except Exception:
        return


def build_callback_payload(
    *,
    pipeline_task_id: str,
    pipeline_id: str,
    target_dataset_id: str,
    succeeded: bool,
    output_uri: str,
    total_videos: int | None = None,
    success_videos: int | None = None,
) -> dict[str, object]:
    """Build the callback payload required by Data Engine."""

    data: dict[str, object] = {
        "pipeline_task_id": pipeline_task_id,
        "pipeline_id": pipeline_id,
        "target_dataset_id": target_dataset_id,
        "succeeded": succeeded,
        "output_uri": output_uri,
    }
    if total_videos is not None:
        data["total_videos"] = total_videos
    if success_videos is not None:
        data["success_videos"] = success_videos

    return {
        "code": 0 if succeeded else -1,
        "message": "success" if succeeded else "failed",
        "data": data,
    }


def build_accepted_response(request: DataEngineRequest) -> dict[str, object]:
    """Build the synchronous accepted response required by Data Engine."""

    return {
        "code": 0,
        "message": "accepted",
        "data": {
            "pipeline_id": request.pipeline_id,
            "pipeline_task_id": request.pipeline_task_id,
            "target_dataset_id": request.args.target_dataset_id,
            "accepted": True,
        },
    }


def build_invalid_request_response(raw: Mapping[str, object], error: str) -> dict[str, object]:
    """Build a synchronous invalid-request response for malformed Data Engine requests."""

    args_raw = raw.get("args")
    args = args_raw if isinstance(args_raw, Mapping) else {}

    def optional_str(source: Mapping[str, object], key: str) -> str:
        value = source.get(key)
        return value if isinstance(value, str) else ""

    return {
        "code": -1,
        "message": "invalid request",
        "data": {
            "pipeline_id": optional_str(raw, "pipeline_id"),
            "pipeline_task_id": optional_str(raw, "pipeline_task_id"),
            "target_dataset_id": optional_str(args, "target_dataset_id"),
            "error": error,
            "error_type": "invalid_request",
        },
    }


def send_callback(
    request: DataEngineRequest,
    *,
    succeeded: bool,
    total_videos: int | None = None,
    success_videos: int | None = None,
) -> None:
    """Send the terminal callback for a Data Engine request."""

    if total_videos is None:
        total_videos = len(request.args.source_uris)
    if success_videos is None:
        success_videos = total_videos if succeeded else 0

    payload = build_callback_payload(
        pipeline_task_id=request.pipeline_task_id,
        pipeline_id=request.pipeline_id,
        target_dataset_id=request.args.target_dataset_id,
        succeeded=succeeded,
        output_uri=request.args.output_uri,
        total_videos=total_videos,
        success_videos=success_videos,
    )
    logger.info("Sending Data Engine callback to {}: {}", request.args.callback_url, json.dumps(payload))
    try:
        response = requests.post(request.args.callback_url, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        log_data_engine_event(
            "error",
            "data_engine_callback_failed",
            request,
            callback_url=request.args.callback_url,
            callback_succeeded=succeeded,
            callback_payload_code=payload["code"],
            total_videos=total_videos,
            success_videos=success_videos,
            error=str(exc),
            exception_type=type(exc).__name__,
        )
        raise

    log_data_engine_event(
        "info",
        "data_engine_callback_sent",
        request,
        callback_url=request.args.callback_url,
        callback_succeeded=succeeded,
        callback_payload_code=payload["code"],
        callback_status_code=getattr(response, "status_code", None),
        total_videos=total_videos,
        success_videos=success_videos,
    )
