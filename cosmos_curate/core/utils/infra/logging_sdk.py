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
"""Helpers for constructing the project logging SDK client."""

import os

from log_sdk import LogConfig, get_client

DEFAULT_APP = "darwinmind_data_engine"
DEFAULT_SERVICE_NAME = "cosmos_curate"
DEFAULT_LOKI_URL = "http://192.168.9.132:3100/loki/api/v1/push"
_ENV_ENABLE_LOKI = "COSMOS_CURATE_LOG_ENABLE_LOKI"
_ENV_LOKI_URL = "COSMOS_CURATE_LOG_LOKI_URL"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in _TRUE_VALUES


def _resolve_enable_loki(enable_loki: bool | None) -> bool:
    if enable_loki is not None:
        return enable_loki
    raw = os.environ.get(_ENV_ENABLE_LOKI)
    if raw is None:
        return False
    return _parse_bool(raw)


def _resolve_loki_url(loki_url: str | None) -> str | None:
    if loki_url is not None:
        return loki_url
    return os.environ.get(_ENV_LOKI_URL, DEFAULT_LOKI_URL)


def get_logging_client(*, enable_loki: bool | None = None, loki_url: str | None = None) -> object:
    """Return the shared project logging client.

    Args:
        enable_loki: Whether to push logs to Loki. ``None`` means read from
            ``COSMOS_CURATE_LOG_ENABLE_LOKI`` and default to ``False``.
        loki_url: Optional Loki push URL. ``None`` means read from
            ``COSMOS_CURATE_LOG_LOKI_URL`` and otherwise use the documented
            default.
    """
    resolved_enable_loki = _resolve_enable_loki(enable_loki)
    resolved_loki_url = _resolve_loki_url(loki_url) if resolved_enable_loki else None
    return get_client(
        LogConfig(
            app=DEFAULT_APP,
            service_name=DEFAULT_SERVICE_NAME,
            enable_loki=resolved_enable_loki,
            loki_url=resolved_loki_url,
        )
    )
