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
"""Tests for the project logging SDK helper."""

from cosmos_curate.core.utils.infra.logging_sdk import DEFAULT_APP, DEFAULT_SERVICE_NAME, get_logging_client


def test_get_logging_client_defaults_to_stdout_only() -> None:
    client = get_logging_client()
    assert client._logger.logger.name == f"{DEFAULT_APP}.{DEFAULT_SERVICE_NAME}"  # noqa: SLF001
    assert len(client._logger.logger.handlers) == 2  # noqa: SLF001
    assert {handler.__class__.__name__ for handler in client._logger.logger.handlers} == {"StreamHandler", "LokiHandler"}  # noqa: SLF001


def test_get_logging_client_can_enable_loki(monkeypatch) -> None:
    monkeypatch.setenv("COSMOS_CURATE_LOG_ENABLE_LOKI", "1")
    client = get_logging_client()
    assert len(client._logger.logger.handlers) == 2  # noqa: SLF001
    assert {handler.__class__.__name__ for handler in client._logger.logger.handlers} == {"StreamHandler", "LokiHandler"}  # noqa: SLF001
