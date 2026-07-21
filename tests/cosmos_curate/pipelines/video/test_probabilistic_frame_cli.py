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


def test_vipe_run_mode_cli_defaults_to_resident_clip_only() -> None:
    """ViPE pipeline mode is fixed to the tested clip-once resident strategy."""
    parser = argparse.ArgumentParser()
    _setup_parser(parser)

    defaults = parser.parse_args([])
    assert defaults.vipe_run_mode == "resident-clip"

    parsed = parser.parse_args(["--vipe-run-mode", "resident-clip"])
    assert parsed.vipe_run_mode == "resident-clip"

    for old_mode in ["subprocess-window", "resident-window"]:
        try:
            parser.parse_args(["--vipe-run-mode", old_mode])
        except SystemExit:
            pass
        else:  # pragma: no cover - keeps the assertion readable if argparse changes.
            raise AssertionError(f"legacy ViPE mode should be rejected: {old_mode}")


def test_skfp_pose_cli_defaults_and_overrides() -> None:
    """SKFP pose controls expose only resident window/clip modes."""
    parser = argparse.ArgumentParser()
    _setup_parser(parser)

    defaults = parser.parse_args([])
    assert defaults.enable_skfp_pose is False
    assert defaults.skfp_run_mode == "resident-window"
    assert defaults.skfp_stride == 32
    assert defaults.skfp_min_keyframes == 3
    assert defaults.skfp_max_keyframes == 64
    assert defaults.skfp_fail_policy == "warn-only"
    assert defaults.skfp_vipe_work_root == "/mlp-01/duanmengxuan/data_pipeline/tmp/skfp_pipeline_vipe_runs"

    for mode in ["resident-window", "resident-clip"]:
        parsed = parser.parse_args(
            [
                "--enable-skfp-pose",
                "--skfp-run-mode",
                mode,
                "--skfp-root",
                "/tmp/skfp",
                "--skfp-vipe-python",
                "python",
                "--skfp-vipe-adapter-script",
                "adapter.py",
                "--skfp-stride",
                "16",
                "--skfp-min-keyframes",
                "4",
                "--skfp-max-keyframes",
                "32",
                "--skfp-fail-policy",
                "skip-window",
                "--skfp-gpus-per-worker",
                "0.5",
                "--skfp-num-workers",
                "2",
            ]
        )
        assert parsed.enable_skfp_pose is True
        assert parsed.skfp_run_mode == mode
        assert parsed.skfp_root == "/tmp/skfp"
        assert parsed.skfp_vipe_python == "python"
        assert parsed.skfp_vipe_adapter_script == "adapter.py"
        assert parsed.skfp_stride == 16
        assert parsed.skfp_min_keyframes == 4
        assert parsed.skfp_max_keyframes == 32
        assert parsed.skfp_fail_policy == "skip-window"
        assert parsed.skfp_gpus_per_worker == 0.5
        assert parsed.skfp_num_workers == 2


def test_vipe_and_skfp_pose_cli_flags_are_mutually_exclusive() -> None:
    from cosmos_curate.pipelines.video.splitting_pipeline import _assemble_stages

    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    args = parser.parse_args(
        [
            "--input-video-path",
            "/tmp/input",
            "--output-clip-path",
            "/tmp/output",
            "--enable-vipe-pose",
            "--vipe-python",
            "python",
            "--enable-skfp-pose",
            "--skfp-root",
            "/tmp/skfp",
            "--skfp-vipe-python",
            "python",
            "--skfp-vipe-adapter-script",
            "adapter.py",
        ]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _assemble_stages(args)
