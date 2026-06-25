# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from cosmos_curate.core.utils.infra import ray_cluster_utils


def test_init_or_connect_uses_ray_address_env(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:9101")

    with (
        patch("cosmos_curate.core.utils.infra.ray_cluster_utils.ray.init") as mock_init,
        patch("cosmos_curate.core.utils.infra.ray_cluster_utils.ray.util.register_serializer"),
    ):
        ray_cluster_utils.init_or_connect_to_cluster()

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["address"] == "127.0.0.1:9101"
    assert mock_init.call_args.kwargs["ignore_reinit_error"] is True


def test_init_or_connect_keeps_local_default_without_ray_address(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    with (
        patch("cosmos_curate.core.utils.infra.ray_cluster_utils.ray.init") as mock_init,
        patch("cosmos_curate.core.utils.infra.ray_cluster_utils.ray.util.register_serializer"),
    ):
        ray_cluster_utils.init_or_connect_to_cluster()

    mock_init.assert_called_once()
    assert "address" not in mock_init.call_args.kwargs
