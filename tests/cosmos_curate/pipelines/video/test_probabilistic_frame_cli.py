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
"""Tests for probabilistic frame-window CLI helpers."""

import argparse

import pytest

from cosmos_curate.pipelines.video.splitting_pipeline import (
    _build_weighted_frame_window_config,
    _parse_frame_number_output_subdirs,
    _setup_parser,
)


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "frame_number": None,
        "frame_number_choices": None,
        "frame_number_weights": None,
        "frame_number_random_seed": 0,
        "frame_number_output_subdirs": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_weighted_frame_config_parses_valid_values() -> None:
    """Valid comma-separated choices and weights become a weighted config."""
    config = _build_weighted_frame_window_config(
        _args(
            frame_number_choices="81,321,641,961",
            frame_number_weights="12,3,1.5,1",
            frame_number_random_seed=1234,
        )
    )
    assert config is not None
    assert config.choices == (81, 321, 641, 961)
    assert config.weights == (12.0, 3.0, 1.5, 1.0)
    assert config.random_seed == 1234


def test_weighted_frame_config_rejects_fixed_frame_number_mix() -> None:
    """Weighted mode is mutually exclusive with the legacy fixed frame count."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_weighted_frame_window_config(
            _args(
                frame_number=128,
                frame_number_choices="81,321",
                frame_number_weights="1,1",
            )
        )


def test_weighted_frame_config_rejects_length_mismatch() -> None:
    """Choices and weights must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        _build_weighted_frame_window_config(
            _args(
                frame_number_choices="81,321",
                frame_number_weights="1",
            )
        )


def test_output_subdirs_parse_to_mapping() -> None:
    """Output subdirectories are mapped by matching frame-count choice."""
    mapping = _parse_frame_number_output_subdirs(
        choices=(81, 321),
        raw_subdirs="frames_81,frames_321",
    )
    assert mapping == {81: "frames_81", 321: "frames_321"}


def test_output_subdirs_reject_parent_escape() -> None:
    """Output subdirectories cannot escape the dataset root."""
    with pytest.raises(ValueError, match="relative path component"):
        _parse_frame_number_output_subdirs(choices=(81,), raw_subdirs="../bad")


def test_api_caption_worker_cli_defaults_and_overrides() -> None:
    """API caption worker controls expose low-CPU explicit actor defaults."""
    parser = argparse.ArgumentParser()
    _setup_parser(parser)

    defaults = parser.parse_args([])
    assert defaults.api_caption_num_workers_per_node == 4
    assert defaults.api_caption_cpus_per_worker == 0.25

    parsed = parser.parse_args(
        [
            "--api-caption-num-workers-per-node",
            "8",
            "--api-caption-cpus-per-worker",
            "0.5",
        ]
    )
    assert parsed.api_caption_num_workers_per_node == 8
    assert parsed.api_caption_cpus_per_worker == 0.5
