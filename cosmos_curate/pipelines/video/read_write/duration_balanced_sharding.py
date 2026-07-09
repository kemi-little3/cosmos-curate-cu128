#!/usr/bin/env python3
"""Split an input video JSON into duration-balanced shard JSON files."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from cosmos_curate.pipelines.video.read_write.input_duration_summary import duration_for, fmt
except ModuleNotFoundError:  # Allow running this file directly during local smoke tests.
    from input_duration_summary import duration_for, fmt


DEFAULT_CONTAINER_INPUT_VIDEO_PATH = "/config/ogame_videos"
DEFAULT_INPUT_VIDEO_PATH = "/vepfs-mlp-data-dl-01/linyuxi/datasets"
DEFAULT_DURATION_TOLERANCE_SECONDS = 3600.0


@dataclass(frozen=True)
class VideoItem:
    container_path: str
    host_path: Path
    duration: float
    original_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an input JSON into duration-balanced shard JSON files."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input JSON containing container-visible video paths.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "Output prefix. For example /tmp/input_shard writes "
            "/tmp/input_shard_0.json and /tmp/input_shard_1.json."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=2,
        help="Number of shard JSON files to write. Defaults to 2.",
    )
    parser.add_argument(
        "--input-video-path",
        default=DEFAULT_INPUT_VIDEO_PATH,
        help=f"Host video root. Defaults to {DEFAULT_INPUT_VIDEO_PATH}.",
    )
    parser.add_argument(
        "--container-input-video-path",
        default=DEFAULT_CONTAINER_INPUT_VIDEO_PATH,
        help=(
            "Container video root used in the input JSON. "
            f"Defaults to {DEFAULT_CONTAINER_INPUT_VIDEO_PATH}."
        ),
    )
    parser.add_argument(
        "--keep-original-order",
        action="store_true",
        help="Keep original JSON order inside each shard after balancing.",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_SECONDS,
        help=(
            "When shard durations are within this many seconds of the shortest "
            "shard, prefer the shard with fewer files. "
            f"Defaults to {DEFAULT_DURATION_TOLERANCE_SECONDS:.0f}."
        ),
    )
    parser.add_argument(
        "--fail-on-missing-duration",
        action="store_true",
        help="Exit with an error if any listed video duration cannot be read.",
    )
    return parser.parse_args()


def container_to_host_path(
    container_path: str, container_root: str, host_root: Path
) -> Path:
    container_root = container_root.rstrip("/")
    if container_path == container_root:
        relative_path = ""
    elif container_path.startswith(f"{container_root}/"):
        relative_path = container_path[len(container_root) + 1 :]
    else:
        raise ValueError(
            f"path is not under container root {container_root}: {container_path}"
        )
    return host_root / relative_path


def load_video_items(
    input_json: Path,
    host_root: Path,
    container_root: str,
    fail_on_missing_duration: bool,
) -> tuple[list[VideoItem], list[tuple[str, str]]]:
    container_paths = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(container_paths, list):
        raise ValueError(f"input JSON must be a list: {input_json}")

    items = []
    failed = []
    for index, container_path in enumerate(container_paths):
        if not isinstance(container_path, str):
            failed.append((str(container_path), "not a string"))
            continue

        try:
            host_path = container_to_host_path(container_path, container_root, host_root)
        except ValueError as exc:
            failed.append((container_path, str(exc)))
            continue

        try:
            duration = duration_for(str(host_path))
        except OSError as exc:
            duration = None
            failed.append((container_path, str(exc)))

        if duration is None:
            failed.append((container_path, "duration unavailable"))
            if fail_on_missing_duration:
                continue
            duration = 0.0

        items.append(VideoItem(container_path, host_path, float(duration), index))

    return items, failed


def greedy_shard(
    items: list[VideoItem],
    num_shards: int,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
) -> list[list[VideoItem]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if duration_tolerance_seconds < 0:
        raise ValueError("duration_tolerance_seconds must be non-negative")

    shards = [[] for _ in range(num_shards)]
    totals = [0.0 for _ in range(num_shards)]

    for item in sorted(items, key=lambda value: (-value.duration, value.original_index)):
        min_total = min(totals)
        candidates = [
            index
            for index in range(num_shards)
            if totals[index] <= min_total + duration_tolerance_seconds
        ]
        shard_index = min(
            candidates,
            key=lambda index: (len(shards[index]), totals[index], index),
        )
        shards[shard_index].append(item)
        totals[shard_index] += item.duration

    return shards


def write_shards(
    shards: list[list[VideoItem]], output_prefix: Path, keep_original_order: bool
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        if keep_original_order:
            shard = sorted(shard, key=lambda item: item.original_index)
        output_path = output_prefix.with_name(f"{output_prefix.name}_{index}.json")
        paths = [item.container_path for item in shard]
        output_path.write_text(
            json.dumps(paths, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def print_shard_summary(shards: list[list[VideoItem]]) -> None:
    for index, shard in enumerate(shards):
        total = sum(item.duration for item in shard)
        print(
            f"shard={index}\tfiles={len(shard)}\t"
            f"duration_seconds={total:.3f}\tduration={fmt(total)}"
        )


def main() -> int:
    args = parse_args()
    started_at = datetime.now()
    started = time.time()
    print(f"started_at={started_at.isoformat(timespec='seconds')}")

    if args.num_shards <= 0:
        print("num-shards must be positive")
        return 2

    input_json = Path(args.input_json).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser()
    host_root = Path(args.input_video_path).expanduser().resolve()

    items, failed = load_video_items(
        input_json=input_json,
        host_root=host_root,
        container_root=args.container_input_video_path,
        fail_on_missing_duration=args.fail_on_missing_duration,
    )

    if failed:
        print(f"failed_durations={len(failed)}")
        for container_path, reason in failed[:20]:
            print(f"FAILED\t{reason}\t{container_path}")
        if args.fail_on_missing_duration:
            return 1

    shards = greedy_shard(
        items,
        args.num_shards,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
    )
    write_shards(shards, output_prefix, args.keep_original_order)

    print(f"input_json={input_json}")
    print(f"host_input_root={host_root}")
    print(f"container_input_root={args.container_input_video_path.rstrip('/')}")
    print(f"input_videos={len(items)}")
    print(f"duration_tolerance_seconds={args.duration_tolerance_seconds:.3f}")
    print_shard_summary(shards)

    ended_at = datetime.now()
    elapsed = time.time() - started
    print(f"ended_at={ended_at.isoformat(timespec='seconds')}")
    print(f"elapsed_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
