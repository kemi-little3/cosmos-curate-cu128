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
"""Ray pipelines.

Which:
  - Download splitted & annotated video clips
  - Optionally filter clips based on semantic dedup results
  - Generate T5 embedding for captions
  - Pack clips into webdataset
"""

import argparse
import collections
import pathlib
import time
from typing import Any
from collections.abc import Generator, Iterable

from loguru import logger

from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.core.utils.config import args_utils
from cosmos_curate.core.utils.dataset import dimensions, webdataset_utils
from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curate.core.utils.storage.storage_utils import (
    create_path,
    get_directories_relative,
    get_files_relative,
    get_full_path,
    get_storage_client,
    read_json_file,
    verify_path,
)
from cosmos_curate.pipelines.common_pipeline_settings import (
    composite_from_namespace,
    composite_profiling_scope,
    sync_common_from_namespace,
)
from cosmos_curate.pipelines.pipeline_args import add_common_args
from cosmos_curate.pipelines.video.captioning.captioning_stages import T5StageForShard
from cosmos_curate.pipelines.video.read_write.download_stages import DownloadPackUpload
from cosmos_curate.pipelines.video.read_write.summary_writers import write_shard_summary
from cosmos_curate.pipelines.video.shard_pipeline_settings import (
    MIN_CLIPS_PER_TAR_DEFAULT,
    ShardPipelineSettings,
    add_shard_args,
)
from cosmos_curate.pipelines.video.utils.data_model import (
    ClipSample,
    ShardPipeTask,
)
from cosmos_curate.pipelines.video.utils.video_pipe_input import (
    extract_shard_tasks,
    filter_shard_tasks_by_semantic_dedup,
)

FLAT_SHARD_DATASETS_DIR = "datasets"


def _parse_window_dimensions(item: dict[str, Any]) -> tuple[int, int]:
    raw_resolution = item.get("resolution")
    if isinstance(raw_resolution, str) and raw_resolution:
        normalized = raw_resolution.lower().replace("x", "*")
        if "*" in normalized:
            width_raw, height_raw = normalized.split("*", 1)
            return int(width_raw), int(height_raw)

    width = item.get("width")
    height = item.get("height")
    if width is None or height is None:
        msg = f"Window package item {item.get('id')!r} is missing resolution/width/height"
        raise ValueError(msg)
    return int(width), int(height)


def _parse_window_frame_range(item: dict[str, Any]) -> tuple[int | None, int | None]:
    item_id = str(item.get("id", ""))
    parts = item_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return int(parts[1]), int(parts[2])
    return None, None


def _parse_window_frame_num(item: dict[str, Any]) -> int:
    raw_frame_num = item.get("frame_num")
    if raw_frame_num is not None:
        return int(raw_frame_num)
    start_frame, end_frame = _parse_window_frame_range(item)
    if start_frame is not None and end_frame is not None:
        return end_frame - start_frame + 1
    msg = f"Window package item {item.get('id')!r} is missing frame_num"
    raise ValueError(msg)


def _window_video_num_bytes(video_path: StoragePrefix | pathlib.Path, item: dict[str, Any]) -> int:
    raw_num_bytes = item.get("num_bytes")
    if raw_num_bytes is not None:
        return int(raw_num_bytes)
    if isinstance(video_path, pathlib.Path):
        if not video_path.is_file():
            msg = f"Missing window video file: {video_path}"
            raise FileNotFoundError(msg)
        return video_path.stat().st_size
    return 0


def extract_window_package_samples(
    input_path: str,
    input_s3_profile_name: str = "default",
    *,
    caption_field: str = "openai_caption",
) -> list[ClipSample]:
    """Extract per-window shard samples from package_output.json.

    The package directory is expected to contain package_output.json with each item pointing to a
    per-window mp4 via its relative video_path field.
    """
    client_input = get_storage_client(input_path, profile_name=input_s3_profile_name)
    package_output_path = get_full_path(input_path, "package_output.json")
    package_items = read_json_file(package_output_path, client_input)
    if not isinstance(package_items, list):
        msg = f"Expected {package_output_path} to contain a JSON list"
        raise ValueError(msg)

    samples: list[ClipSample] = []
    for item in package_items:
        if not isinstance(item, dict):
            msg = f"Window package item must be an object, got {type(item).__name__}"
            raise ValueError(msg)
        item_id = str(item.get("id") or "")
        video_rel_path = str(item.get("video_path") or "")
        if not item_id or not video_rel_path:
            msg = f"Window package item is missing id/video_path: {item!r}"
            raise ValueError(msg)

        video_path = get_full_path(input_path, video_rel_path)
        width, height = _parse_window_dimensions(item)
        frame_num = _parse_window_frame_num(item)
        fps = float(item.get("fps") or 16)
        num_bytes = _window_video_num_bytes(video_path, item)
        caption = str(item.get("caption") or "")
        start_frame, end_frame = _parse_window_frame_range(item)
        window_metadata: dict[str, Any] = {caption_field: caption}
        if start_frame is not None and end_frame is not None:
            window_metadata["start_frame"] = start_frame
            window_metadata["end_frame"] = end_frame

        clip_metadata = dict(item)
        clip_metadata.update(
            {
                "span_uuid": item_id,
                "clip_location": str(video_path),
                "width": width,
                "height": height,
                "framerate": fps,
                "num_frames": frame_num,
                "num_bytes": num_bytes,
                "valid": True,
                "has_caption": bool(caption),
                "windows": [window_metadata],
            },
        )

        samples.append(
            ClipSample(
                uuid=item_id,
                width=width,
                height=height,
                num_frames=frame_num,
                framerate=fps,
                num_bytes=num_bytes,
                clip_location=video_path,
                clip_metadata=clip_metadata,
            ),
        )
    return samples


def _group_samples_by_bin(
    samples: Iterable[ClipSample],
) -> dict[dimensions.ResolutionAspectRatioFrames | None, list[ClipSample]]:
    out: dict[dimensions.ResolutionAspectRatioFrames | None, list[ClipSample]] = collections.defaultdict(list)
    bin_spec = dimensions.ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    count = 0
    for sample in samples:
        framerate = getattr(sample, "framerate", None)
        if framerate and framerate > 0:
            lbin = bin_spec.find_appropriate_bin(
                dimensions.Dimensions(sample.width, sample.height), float(sample.num_frames) / sample.framerate
            )
        else:
            logger.warning(f"Invalid framerate={framerate} for sample; assigning to None bin.")
            lbin = None
        out[lbin].append(sample)
        count += 1
    logger.info(f"Found {count} total samples in {len(out)} bins.")
    return out


def _group_samples_by_size(
    samples: list[ClipSample],
    target_size_bytes: int,
    *,
    drop_small_shards: bool,
    min_clips_per_tar: int = MIN_CLIPS_PER_TAR_DEFAULT,
) -> Generator[list[ClipSample], None, None]:
    current_size = 0
    out: list[ClipSample] = []

    for sample in samples:
        if not out:
            out.append(sample)
            current_size = sample.num_bytes
        elif current_size + sample.num_bytes > target_size_bytes:
            yield out
            out = [sample]
            current_size = sample.num_bytes
        else:
            out.append(sample)
            current_size += sample.num_bytes

    if out and not (drop_small_shards and len(out) < min_clips_per_tar):
        yield out


def _group_samples_by_count(
    samples: list[ClipSample],
    target_tar_count: int,
    *,
    drop_small_shards: bool,
    min_clips_per_tar: int = MIN_CLIPS_PER_TAR_DEFAULT,
) -> Generator[list[ClipSample], None, None]:
    if target_tar_count < 1:
        msg = "target_tar_count must be positive"
        raise ValueError(msg)
    if not samples:
        return

    total_size = sum(max(0, sample.num_bytes) for sample in samples)
    target_size_bytes = max(1, (total_size + target_tar_count - 1) // target_tar_count)
    emitted = 0
    current_size = 0
    out: list[ClipSample] = []

    for idx, sample in enumerate(samples):
        remaining_samples = len(samples) - idx
        remaining_bins = target_tar_count - emitted
        must_leave_samples = remaining_bins > 1 and remaining_samples <= remaining_bins
        should_split = bool(out) and emitted + 1 < target_tar_count and (
            current_size + sample.num_bytes > target_size_bytes or must_leave_samples
        )
        if should_split:
            if not (drop_small_shards and len(out) < min_clips_per_tar):
                emitted += 1
                yield out
            out = [sample]
            current_size = sample.num_bytes
        else:
            out.append(sample)
            current_size += sample.num_bytes

    if out and not (drop_small_shards and len(out) < min_clips_per_tar):
        yield out


def _allocate_target_tar_counts(
    grouped_by_bin: dict[dimensions.ResolutionAspectRatioFrames | None, list[ClipSample]],
    target_tar_count: int | None,
) -> dict[dimensions.ResolutionAspectRatioFrames, int]:
    if target_tar_count is None:
        return {}

    valid_bins = [(lbin, bin_samples) for lbin, bin_samples in grouped_by_bin.items() if lbin is not None and bin_samples]
    if not valid_bins:
        return {}

    if target_tar_count < len(valid_bins):
        logger.warning(
            f"target_tar_count={target_tar_count} is smaller than the number of non-empty bins={len(valid_bins)}; "
            "emitting at least one tar per bin."
        )

    total_target_count = max(target_tar_count, len(valid_bins))
    remaining = total_target_count - len(valid_bins)
    allocations = {lbin: 1 for lbin, _ in valid_bins}
    if remaining == 0:
        return allocations

    bin_sizes = {
        lbin: sum(max(0, sample.num_bytes) for sample in bin_samples)
        for lbin, bin_samples in valid_bins
    }
    total_size = sum(bin_sizes.values())
    if total_size <= 0:
        for idx in range(remaining):
            allocations[valid_bins[idx % len(valid_bins)][0]] += 1
        return allocations

    fractional_remainders: list[tuple[float, dimensions.ResolutionAspectRatioFrames]] = []
    assigned_extra = 0
    for lbin, _ in valid_bins:
        exact_extra = remaining * bin_sizes[lbin] / total_size
        extra = int(exact_extra)
        allocations[lbin] += extra
        assigned_extra += extra
        fractional_remainders.append((exact_extra - extra, lbin))

    for _, lbin in sorted(fractional_remainders, key=lambda item: item[0], reverse=True)[: remaining - assigned_extra]:
        allocations[lbin] += 1
    return allocations


def _next_flat_tar_num(output_path: StoragePrefix | pathlib.Path, client_output: StorageClient | None) -> int:
    existing_files = get_files_relative(str(output_path), client_output)
    existing_tar_nums = []
    for item in existing_files:
        path = pathlib.PurePosixPath(item)
        if path.parent != pathlib.PurePosixPath(".") or path.suffix != ".tar" or not path.stem.isdigit():
            continue
        existing_tar_nums.append(int(path.stem))
    return max(existing_tar_nums, default=-1) + 1


def _group_samples_into_flat_tasks(  # noqa: PLR0913
    samples: Iterable[ClipSample],
    *,
    drop_small_shards: bool,
    target_tar_size_bytes: int,
    target_tar_count: int | None,
    min_clips_per_tar: int,
    output_path: str,
    output_s3_profile_name: str,
) -> tuple[list[ShardPipeTask], list[StoragePrefix | pathlib.Path], int]:
    tasks: list[ShardPipeTask] = []
    valid_samples: list[ClipSample] = []
    num_dropped_samples = 0
    grouped_by_bin = _group_samples_by_bin(samples)
    for lbin, binned_samples in grouped_by_bin.items():
        if lbin is None:
            logger.warning(f"Found {len(binned_samples)} samples which do not correspond to a lbin. Ignoring them ...")
            num_dropped_samples += len(binned_samples)
            continue
        valid_samples.extend(binned_samples)

    flat_root = get_full_path(output_path)
    flat_dataset_path = get_full_path(output_path, FLAT_SHARD_DATASETS_DIR)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)
    starting_tar_num = _next_flat_tar_num(flat_dataset_path, client_output)
    logger.info(f"Writing flat shard output under {flat_dataset_path}; starting tar number: {starting_tar_num}")

    grouped_samples = (
        _group_samples_by_count(
            valid_samples,
            target_tar_count,
            drop_small_shards=drop_small_shards,
            min_clips_per_tar=min_clips_per_tar,
        )
        if target_tar_count is not None
        else _group_samples_by_size(
            valid_samples,
            target_tar_size_bytes,
            drop_small_shards=drop_small_shards,
            min_clips_per_tar=min_clips_per_tar,
        )
    )
    for tar_idx, tar_samples in enumerate(grouped_samples, start=starting_tar_num):
        tar_name = webdataset_utils.make_tar_path_str(tar_idx)
        tasks.append(
            ShardPipeTask(
                str(flat_root),
                tar_idx,
                tar_samples,
                get_full_path(flat_dataset_path, tar_name),
                get_full_path(flat_root, "metas", tar_name),
                get_full_path(flat_root, "t5_xxl", tar_name),
                key_count=0,
                write_auxiliary_tars=False,
            ),
        )
    return tasks, [flat_root], num_dropped_samples


def _group_samples_into_tasks(  # noqa: PLR0913
    samples: Iterable[ClipSample],
    *,
    drop_small_shards: bool,
    max_tars_per_part: int,
    target_tar_size_bytes: int,
    target_tar_count: int | None,
    min_clips_per_tar: int,
    output_path: str,
    output_s3_profile_name: str,
    output_layout: str = "binned",
) -> tuple[list[ShardPipeTask], list[StoragePrefix | pathlib.Path], int]:
    if output_layout == "flat":
        tasks, all_bins, num_dropped_samples = _group_samples_into_flat_tasks(
            samples,
            drop_small_shards=drop_small_shards,
            target_tar_size_bytes=target_tar_size_bytes,
            target_tar_count=target_tar_count,
            min_clips_per_tar=min_clips_per_tar,
            output_path=output_path,
            output_s3_profile_name=output_s3_profile_name,
        )
        logger.info(f"Created {len(tasks)} flat shard tasks:")
        for task in tasks:
            logger.info(f"tar={task.part_num} output={task.output_tar_video}, samples={len(task.samples)}")
        return tasks, all_bins, num_dropped_samples

    tasks: list[ShardPipeTask] = []
    all_bins: list[StoragePrefix | pathlib.Path] = []
    num_dropped_samples: int = 0
    grouped_by_bin = _group_samples_by_bin(samples)
    target_tar_counts_by_bin = _allocate_target_tar_counts(grouped_by_bin, target_tar_count)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)
    for lbin, binned_samples in grouped_by_bin.items():
        sample_count = len(binned_samples)
        if lbin is None:
            logger.warning(f"Found {sample_count} samples which do not correspond to a lbin. Ignoring them ...")
            num_dropped_samples += sample_count
            continue

        path_for_bin = get_full_path(output_path, lbin.to_path_string())
        logger.info(f"Inspecting bin {lbin} with {sample_count} samples at {path_for_bin}.")
        all_bins.append(path_for_bin)

        path_for_video = get_full_path(path_for_bin, "video")
        part_dirs = get_directories_relative(str(path_for_video), client_output)

        logger.info(f"Current parts under {part_dirs}:")
        for part_dir in part_dirs:
            logger.info(part_dir)

        starting_part_num = (
            max(
                [webdataset_utils.get_part_num_from_path_str(str(x)) for x in part_dirs],
                default=-1,
            )
            + 1
        )
        logger.info(f"Starting part number: {starting_part_num}")

        bin_target_tar_count = target_tar_counts_by_bin.get(lbin)
        grouped_samples = (
            _group_samples_by_count(
                binned_samples,
                bin_target_tar_count,
                drop_small_shards=drop_small_shards,
                min_clips_per_tar=min_clips_per_tar,
            )
            if bin_target_tar_count is not None
            else _group_samples_by_size(
                binned_samples,
                target_tar_size_bytes,
                drop_small_shards=drop_small_shards,
                min_clips_per_tar=min_clips_per_tar,
            )
        )
        for part_idx, tar_group in enumerate(
            grouping.split_by_chunk_size(
                grouped_samples,
                max_tars_per_part,
            ),
        ):
            part_num = starting_part_num + part_idx
            for tar_idx, tar_samples in enumerate(tar_group):
                path_for_tar = (
                    webdataset_utils.make_part_path_str(part_num) + "/" + webdataset_utils.make_tar_path_str(tar_idx)
                )
                output_object_video = get_full_path(path_for_video, path_for_tar)
                output_object_metas = get_full_path(path_for_bin, "metas", path_for_tar)
                output_object_t5_xxl = get_full_path(path_for_bin, "t5_xxl", path_for_tar)
                tasks.append(
                    ShardPipeTask(
                        str(path_for_bin),
                        part_num,
                        tar_samples,
                        output_object_video,
                        output_object_metas,
                        output_object_t5_xxl,
                        key_count=0,
                    ),
                )
    logger.info(f"Created {len(tasks)} tasks in {len(all_bins)} shards:")
    for task in tasks:
        logger.info(f"part={task.part_num} output={task.output_tar_video}, samples={len(task.samples)}")
    return tasks, all_bins, num_dropped_samples


def shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline with profiling and tracing.

    Public entry point that wraps ``_shard()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: Command line arguments.

    """
    settings = composite_from_namespace(ShardPipelineSettings, args)
    with composite_profiling_scope(settings) as profiling_ns:
        _shard(settings, profiling_ns)


def _shard(settings: ShardPipelineSettings, profiling_args_ns: argparse.Namespace) -> None:
    """Run the shard pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        settings: Validated shard pipeline settings.
        profiling_args_ns: Flat CLI namespace for profiling (same field names as *settings*)

    """
    start_time = time.time()
    # validate input arguments
    output_dataset_path = (
        str(get_full_path(settings.output_dataset_path, settings.annotation_version))
        if settings.shard_output_layout == "binned"
        else settings.output_dataset_path
    )
    verify_path(settings.input_clip_path)
    verify_path(settings.output_dataset_path, level=1)
    create_path(settings.output_dataset_path)

    # get input samples
    if settings.shard_input_mode == "window-package":
        samples = extract_window_package_samples(
            settings.input_clip_path,
            settings.common.input_s3_profile_name,
            caption_field=f"{settings.captioning_algorithm}_caption",
        )
        logger.info(f"Found {len(samples)} window package samples under input path {settings.input_clip_path}.")
    else:
        samples = extract_shard_tasks(
            settings.input_clip_path,
            output_dataset_path,
            settings.common.input_s3_profile_name,
            settings.common.output_s3_profile_name,
            settings.annotation_version,
            verbose=settings.common.verbose,
        )
        logger.info(f"Found {len(samples)} clip samples under input path {settings.input_clip_path}.")

    if settings.input_semantic_dedup_path is not None:
        samples = filter_shard_tasks_by_semantic_dedup(
            samples,
            settings.input_semantic_dedup_path,
            settings.input_semantic_dedup_s3_profile_name,
            settings.semantic_dedup_epsilon,
            verbose=settings.common.verbose,
        )
        logger.info(f"After semantic deduplication, {len(samples)} samples remain.")

    # Convert target tar size from MB to bytes
    target_tar_size_bytes = settings.target_tar_size_mb * 1024 * 1024

    tasks, all_bins, num_dropped_samples = _group_samples_into_tasks(
        samples,
        drop_small_shards=settings.drop_small_shards,
        max_tars_per_part=settings.max_tars_per_part,
        target_tar_size_bytes=target_tar_size_bytes,
        target_tar_count=settings.target_tar_count,
        min_clips_per_tar=settings.min_clips_per_tar,
        output_path=output_dataset_path,
        output_s3_profile_name=settings.common.output_s3_profile_name,
        output_layout=settings.shard_output_layout,
    )
    logger.info(f"Dropped {num_dropped_samples} samples during sharding process.")
    if len(tasks) == 0:
        logger.warning("No tasks to process. Exiting ...")
        return

    stages: list[CuratorStage | CuratorStageSpec] = []
    if settings.generate_t5_embeddings:
        stages.append(
            T5StageForShard(
                caption_fields=[f"{settings.captioning_algorithm}_caption"],
                verbose=settings.common.verbose,
                log_stats=settings.common.perf_profile,
            ),
        )
    stages.append(
        CuratorStageSpec(
            DownloadPackUpload(
                input_path=settings.input_clip_path,
                output_path=output_dataset_path,
                input_s3_profile_name=settings.common.input_s3_profile_name,
                output_s3_profile_name=settings.common.output_s3_profile_name,
                verbose=settings.common.verbose,
                log_stats=settings.common.perf_profile,
            ),
            num_workers_per_node=8,
        ),
    )

    output_packets: list[ShardPipeTask] = run_pipeline(
        tasks,
        stages,
        args=profiling_args_ns,
    )
    sync_common_from_namespace(settings, profiling_args_ns)
    if settings.common.perf_profile:
        total_object_size = 0
        for packet in output_packets:
            total_object_size += packet.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    if settings.shard_output_layout == "binned":
        write_shard_summary(
            output_dataset_path,
            settings.output_dataset_path,
            settings.common.output_s3_profile_name,
            all_bins,
            settings.max_tars_per_part,
            output_packets,
            perf_profile=settings.common.perf_profile,
        )

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Embedding-Shard-Webdataset pipeline completed in {elapsed_time:.2f} minutes")


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    """Set up the parser for the shard pipeline.

    Registers shard-only flags then shared flags (same order as settings construction in :func:`shard`).

    Args:
        parser: The parser to add arguments to.

    """
    add_shard_args(parser)
    add_common_args(parser)


def nvcf_run_shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    cli_run_shard(args)


def cli_run_shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline in CLI mode.

    Args:
        args: Command line arguments.

    """
    shard(args)


def add_shard_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add shard command to the CLI parser.

    Args:
        subparsers: Subparsers object to add the command to.

    Returns:
        The configured parser for the shard command.

    """
    parser = subparsers.add_parser(
        "shard",
        help="Shard clips into webdatasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=cli_run_shard)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
