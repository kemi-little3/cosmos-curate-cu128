#!/usr/bin/env python3
"""Stage-like builders for reusable video pipeline input and output I/O.

These classes are intentionally lightweight wrappers around the existing input
helpers and sharding pipeline. They provide named, reusable entrypoints for
external platform integration without duplicating the underlying pipeline logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from cosmos_curate.pipelines.video.read_write.input_preparation_builders import (
    build_duration_balanced_input_shards,
    build_input_json_from_video_dir,
    build_input_shards_from_video_list_json,
)
from cosmos_curate.pipelines.video.sharding_pipeline import nvcf_run_shard


@dataclass(frozen=True)
class VideoInputPreparationConfig:
    """Configuration for preparing worker input JSON shards."""

    container_input_video_path: str
    output_json: str
    num_shards: int
    shard_output_prefix: str
    shard_list_json: str
    input_video_path: str | None = None
    input_video_list_json_path: str | None = None
    include_dir: str | None = None
    summary_json: str | None = None
    limit: int | None = None
    keep_original_order: bool = True
    input_video_list_s3_profile_name: str = "default"


@dataclass(frozen=True)
class VideoInputPreparationResult:
    """Artifacts produced by :class:`VideoInputPreparationStage`."""

    input_json: str
    shard_list_json: str
    shard_jsons: list[str]
    num_videos: int


class VideoInputPreparationStage:
    """Prepare raw video inputs for distributed video pipeline workers."""

    stage_name = "video_input_preparation"

    def __init__(self, config: VideoInputPreparationConfig) -> None:
        self.config = config

    def run(self) -> VideoInputPreparationResult:
        if not self.config.input_video_path and not self.config.input_video_list_json_path:
            raise ValueError("input_video_path or input_video_list_json_path must be set")

        if self.config.input_video_list_json_path:
            if not self.config.input_video_path:
                # Remote video URLs can only use count sharding, so input_video_path is not needed.
                input_video_path = "."
            else:
                input_video_path = self.config.input_video_path
            shard_jsons, _strategy = build_input_shards_from_video_list_json(
                input_video_list_json_path=self.config.input_video_list_json_path,
                materialized_input_json=self.config.output_json,
                output_prefix=self.config.shard_output_prefix,
                num_shards=self.config.num_shards,
                output_json=self.config.shard_list_json,
                input_video_path=input_video_path,
                container_input_video_path=self.config.container_input_video_path,
                input_video_list_s3_profile_name=self.config.input_video_list_s3_profile_name,
                keep_original_order=self.config.keep_original_order,
                limit=self.config.limit,
            )
            videos = Path(self.config.output_json).read_text(encoding="utf-8")
            num_videos = len(json.loads(videos))
        else:
            videos = build_input_json_from_video_dir(
                input_video_path=self.config.input_video_path or "",
                container_input_video_path=self.config.container_input_video_path,
                output_json=self.config.output_json,
                include_dir=self.config.include_dir,
                summary_json=self.config.summary_json,
                limit=self.config.limit,
            )
            shard_jsons = build_duration_balanced_input_shards(
                input_json=self.config.output_json,
                output_prefix=self.config.shard_output_prefix,
                num_shards=self.config.num_shards,
                output_json=self.config.shard_list_json,
                input_video_path=self.config.input_video_path or "",
                container_input_video_path=self.config.container_input_video_path,
                keep_original_order=self.config.keep_original_order,
            )
            num_videos = len(videos)

        return VideoInputPreparationResult(
            input_json=str(Path(self.config.output_json).expanduser()),
            shard_list_json=str(Path(self.config.shard_list_json).expanduser()),
            shard_jsons=shard_jsons,
            num_videos=num_videos,
        )


@dataclass(frozen=True)
class VideoOutputShardingConfig:
    """Configuration for packing split-pipeline outputs into webdataset shards."""

    input_clip_path: str
    output_dataset_path: str
    captioning_algorithm: str = "openai"
    annotation_version: str = "v0"
    shard_input_mode: str = "window-package"
    target_tar_size_mb: int = 500
    min_clips_per_tar: int = 1
    max_tars_per_part: int = 100
    drop_small_shards: bool = False
    generate_t5_embeddings: bool = False
    input_semantic_dedup_path: str | None = None
    input_semantic_dedup_s3_profile_name: str = "default"
    verbose: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class VideoOutputShardingResult:
    """Artifacts produced by :class:`VideoOutputShardingStage`."""

    output_dataset_path: str
    shard_input_mode: str


class VideoOutputShardingStage:
    """Package processed clips/windows into sharded webdataset tar files."""

    stage_name = "video_output_sharding"

    def __init__(self, config: VideoOutputShardingConfig) -> None:
        self.config = config

    def build_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            input_clip_path=self.config.input_clip_path,
            output_dataset_path=self.config.output_dataset_path,
            captioning_algorithm=self.config.captioning_algorithm,
            annotation_version=self.config.annotation_version,
            shard_input_mode=self.config.shard_input_mode,
            target_tar_size_mb=self.config.target_tar_size_mb,
            min_clips_per_tar=self.config.min_clips_per_tar,
            max_tars_per_part=self.config.max_tars_per_part,
            drop_small_shards=self.config.drop_small_shards,
            generate_t5_embeddings=self.config.generate_t5_embeddings,
            input_semantic_dedup_path=self.config.input_semantic_dedup_path,
            input_semantic_dedup_s3_profile_name=self.config.input_semantic_dedup_s3_profile_name,
            verbose=self.config.verbose,
            dry_run=self.config.dry_run,
        )

    def run(self) -> VideoOutputShardingResult:
        nvcf_run_shard(self.build_args())
        return VideoOutputShardingResult(
            output_dataset_path=self.config.output_dataset_path,
            shard_input_mode=self.config.shard_input_mode,
        )
