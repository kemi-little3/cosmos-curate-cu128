#!/usr/bin/env python3
"""Reusable input preparation entrypoints for video pipelines.

This module is intentionally shaped like a builder layer: callers pass host and
container input roots, and it prepares the JSON artifacts consumed by the worker
pipeline. The implementation stays in the neighboring read_write helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from cosmos_curate.core.utils.storage.storage_utils import get_storage_client, read_json_file
from cosmos_curate.pipelines.video.read_write import input_json_generator
from cosmos_curate.pipelines.video.read_write import input_shard_list_builder


def build_input_json_from_video_dir(
    *,
    input_video_path: Path | str,
    container_input_video_path: str,
    output_json: Path | str,
    include_dir: Path | str | None = None,
    summary_json: Path | str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Scan a host video directory and write a container-path JSON list."""
    host_root = Path(input_video_path).expanduser().resolve()
    if not host_root.is_dir():
        raise NotADirectoryError(f"input video path is not a directory: {host_root}")

    scan_root = host_root / include_dir if include_dir else host_root
    scan_root = scan_root.expanduser().resolve()
    if not scan_root.is_dir():
        raise NotADirectoryError(f"include directory is not a directory: {scan_root}")

    videos = input_json_generator.collect_video_paths(
        host_root=host_root,
        scan_roots=[scan_root],
        container_root=container_input_video_path,
        video_exts=input_json_generator.normalize_exts(None),
        excluded_dirs=set(),
        limit=limit,
    )

    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(videos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if summary_json:
        summary = input_json_generator.build_summary(
            input_json=output_path,
            host_root=host_root,
            container_root=container_input_video_path,
            videos=videos,
        )
        summary_path = Path(summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return videos


def build_duration_balanced_input_shards(
    *,
    input_json: Path | str,
    output_prefix: Path | str,
    num_shards: int,
    output_json: Path | str,
    input_video_path: Path | str,
    container_input_video_path: str,
    keep_original_order: bool = True,
) -> list[str]:
    """Write duration-balanced shard JSON files and the shard-list JSON."""
    return input_shard_list_builder.shard_and_write_paths(
        input_json=Path(input_json),
        output_prefix=Path(output_prefix),
        num_shards=num_shards,
        output_json=Path(output_json),
        input_video_path=Path(input_video_path),
        container_input_video_path=container_input_video_path,
        keep_original_order=keep_original_order,
    )



def materialize_input_json(
    *,
    input_video_list_json_path: str,
    output_json: Path | str,
    input_video_list_s3_profile_name: str = "default",
    limit: int | None = None,
) -> list[str]:
    """Read a local or remote input JSON and write a local materialized copy."""
    client = get_storage_client(input_video_list_json_path, profile_name=input_video_list_s3_profile_name)
    data = read_json_file(input_video_list_json_path, client)
    if not isinstance(data, list):
        raise ValueError(f"input video list JSON must contain a list: {input_video_list_json_path}")
    videos = [str(item) for item in data]
    if limit is not None:
        videos = videos[:limit]

    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(videos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return videos


def can_duration_shard_video_list(videos: list[str], container_input_video_path: str) -> bool:
    """Return True when all listed videos can be mapped to local host paths for duration reads."""
    container_root = container_input_video_path.rstrip("/")
    return all(video == container_root or video.startswith(f"{container_root}/") for video in videos)


def build_input_shards_from_video_list_json(
    *,
    input_video_list_json_path: str,
    materialized_input_json: Path | str,
    output_prefix: Path | str,
    num_shards: int,
    output_json: Path | str,
    input_video_path: Path | str,
    container_input_video_path: str,
    input_video_list_s3_profile_name: str = "default",
    keep_original_order: bool = True,
    limit: int | None = None,
) -> tuple[list[str], str]:
    """Prepare shard-list JSON from an existing local or remote video-list JSON.

    Returns:
        A pair of ``(shard_json_paths, strategy)`` where strategy is either
        ``"duration"`` or ``"count"``.
    """
    videos = materialize_input_json(
        input_video_list_json_path=input_video_list_json_path,
        output_json=materialized_input_json,
        input_video_list_s3_profile_name=input_video_list_s3_profile_name,
        limit=limit,
    )
    if can_duration_shard_video_list(videos, container_input_video_path):
        paths = build_duration_balanced_input_shards(
            input_json=materialized_input_json,
            output_prefix=output_prefix,
            num_shards=num_shards,
            output_json=output_json,
            input_video_path=input_video_path,
            container_input_video_path=container_input_video_path,
            keep_original_order=keep_original_order,
        )
        return paths, "duration"

    paths = input_shard_list_builder.count_shard_and_write_paths(
        input_json=Path(materialized_input_json),
        output_prefix=Path(output_prefix),
        num_shards=num_shards,
        output_json=Path(output_json),
    )
    return paths, "count"
