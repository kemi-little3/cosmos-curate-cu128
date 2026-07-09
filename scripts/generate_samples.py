#!/usr/bin/env python3
"""Sample MP4 videos from each directory that directly contains videos."""

import argparse
import os
import random
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path("/mlp-01/linyuxi/datasets")
DEFAULT_OUTPUT = Path("/mlp-01/lihaoyue/caption/data/samples")
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SEED = 20260703


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan a dataset tree, and for each directory that directly contains "
            "MP4 files, sample up to N videos and copy them into a flat output directory."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Root dataset directory to scan.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Flat output directory for sampled MP4s.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum number of MP4s to sample from each directory.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without copying files.",
    )
    return parser.parse_args()


def iter_video_dirs(root: Path) -> list[tuple[Path, list[Path]]]:
    video_dirs: list[tuple[Path, list[Path]]] = []
    for current_dir, _, _ in os.walk(root):
        directory = Path(current_dir)
        videos = sorted(
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() == ".mp4"
        )
        if videos:
            video_dirs.append((directory, videos))
    return sorted(video_dirs, key=lambda item: str(item[0]))


def sample_videos(
    video_dirs: list[tuple[Path, list[Path]]],
    sample_size: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []
    for _, videos in video_dirs:
        if len(videos) <= sample_size:
            chosen = videos
        else:
            chosen = sorted(rng.sample(videos, sample_size))
        selected.extend(chosen)
    return selected


def execute_copy(selected: list[Path], output_dir: Path, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overwritten = 0
    for src in selected:
        dest = output_dir / src.name
        if dest.exists():
            overwritten += 1
        if dry_run:
            print(f"DRY-RUN {src} -> {dest}")
            continue
        shutil.copy2(src, dest)
        print(f"COPIED {src} -> {dest}")
    print(
        f"Done. sampled_files={len(selected)} output_dir={output_dir} "
        f"overwritten_files={overwritten} dry_run={dry_run}"
    )


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError(f"--sample-size must be positive, got {args.sample_size}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.source}")
    if not args.source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {args.source}")

    video_dirs = iter_video_dirs(args.source)
    print(f"Found {len(video_dirs)} directories that directly contain MP4 files under {args.source}")
    selected = sample_videos(video_dirs, args.sample_size, args.seed)
    execute_copy(selected, args.output, args.dry_run)


if __name__ == "__main__":
    main()
