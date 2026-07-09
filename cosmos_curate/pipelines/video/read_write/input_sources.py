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
"""Input source preparation for video pipelines.

This module holds the logic that used to live in shell wrappers:
argument validation, input path normalization, and raw input extraction.
"""

from __future__ import annotations

import argparse

from loguru import logger

from cosmos_curate.core.utils.storage.storage_utils import create_path, is_path_nested, verify_path
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask
from cosmos_curate.pipelines.video.utils.video_pipe_input import (
    extract_multi_cam_split_tasks,
    extract_single_cam_split_tasks,
    format_session_videos_tree,
)

MULTICAM_VIDEO_EXTENSIONS: set[str] = {".mp4"}


def build_input_data(
    args: argparse.Namespace,
    *,
    multi_cam: bool,
    primary_camera_keyword: str,
    input_s3_profile_name: str,
    input_video_list_s3_profile_name: str,
    output_s3_profile_name: str,
    limit: int,
    verbose: bool,
) -> tuple[list[SplitPipeTask], list[str], int, int]:
    """Build input data for the video pipeline.

    This is intentionally kept close to the old splitting-pipeline behavior so the
    surrounding pipeline wiring can move without changing input semantics.
    """
    verify_path(args.input_video_path)
    verify_path(args.output_clip_path, level=1)
    create_path(args.output_clip_path)
    if is_path_nested(args.input_video_path, args.output_clip_path):
        raise ValueError("Do not make input and output paths nested")

    if multi_cam and args.splitting_algorithm != "fixed-stride":
        raise ValueError("Multi-cam only supports fixed-stride splitting; set --splitting-algorithm fixed-stride")

    if multi_cam:
        input_tasks = extract_multi_cam_split_tasks(
            sessions_prefix=args.input_video_path,
            primary_camera_keyword=primary_camera_keyword,
            video_extensions=MULTICAM_VIDEO_EXTENSIONS,
            input_s3_profile_name=input_s3_profile_name,
            limit=limit,
            verbose=verbose,
        )

        if verbose:
            logger.info(format_session_videos_tree(input_tasks, args.input_video_path, limit=3))

        input_videos_relative: list[str] = []
        num_processed = 0
        num_input_videos_selected = len(input_tasks)
        logger.info(f"About to process {len(input_tasks)} multi-cam session tasks ...")
        return input_tasks, input_videos_relative, num_processed, num_input_videos_selected

    input_videos, input_videos_relative, num_processed = extract_single_cam_split_tasks(
        input_path=args.input_video_path,
        input_video_list_json_path=args.input_video_list_json_path,
        output_path=args.output_clip_path,
        output_video_path=ClipWriterStage.get_output_path_processed_videos(args.output_clip_path),
        output_clip_chunk_path=ClipWriterStage.get_output_path_processed_clip_chunks(args.output_clip_path),
        input_s3_profile_name=input_s3_profile_name,
        input_video_list_s3_profile_name=input_video_list_s3_profile_name,
        output_s3_profile_name=output_s3_profile_name,
        limit=limit,
        verbose=verbose,
    )
    input_tasks = [SplitPipeTask(videos=[video], session_id=str(video.input_video)) for video in input_videos]

    if len(input_videos) == 0:
        logger.warning(
            "About to process 0 raw videos - all inputs were already processed. "
            f"Remove the output directory {ClipWriterStage.get_output_path_processed_videos(args.output_clip_path)}"
            " to reprocess.",
        )
    else:
        logger.info(f"About to process {len(input_videos)} raw videos ...")

    if verbose:
        logger.debug("\n".join(str(x.input_video) for x in input_videos))
    num_input_videos_selected = len(input_videos)
    return input_tasks, input_videos_relative, num_processed, num_input_videos_selected
