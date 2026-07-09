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
"""Tests for video input source preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cosmos_curate.pipelines.video.read_write.input_sources import build_input_data


def _make_args(**kwargs: object) -> argparse.Namespace:
    base = {
        "input_video_path": "/tmp/input",
        "output_clip_path": "/tmp/output",
        "splitting_algorithm": "fixed-stride",
        "input_video_list_json_path": None,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_build_input_data_from_json_list(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    video_path = input_root / "clip.mp4"
    video_path.write_bytes(b"x")
    input_list = tmp_path / "input_list.json"
    input_list.write_text(json.dumps([str(video_path)]), encoding="utf-8")

    args = _make_args(
        input_video_path=str(input_root),
        output_clip_path=str(output_root),
        input_video_list_json_path=str(input_list),
    )

    tasks, input_videos_relative, num_processed, num_selected = build_input_data(
        args,
        multi_cam=False,
        primary_camera_keyword="front",
        input_s3_profile_name="default",
        input_video_list_s3_profile_name="default",
        output_s3_profile_name="default",
        limit=0,
        verbose=False,
    )

    assert len(tasks) == 1
    assert input_videos_relative == [str(video_path)]
    assert num_processed == 0
    assert num_selected == 1


def test_build_input_data_rejects_nested_input_and_output(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = input_root / "nested_output"
    input_root.mkdir()

    args = _make_args(
        input_video_path=str(input_root),
        output_clip_path=str(output_root),
    )

    try:
        build_input_data(
            args,
            multi_cam=False,
            primary_camera_keyword="front",
            input_s3_profile_name="default",
            input_video_list_s3_profile_name="default",
            output_s3_profile_name="default",
            limit=0,
            verbose=False,
        )
    except ValueError as exc:
        assert "Do not make input and output paths nested" in str(exc)
    else:
        raise AssertionError("expected nested path validation to fail")
