#!/usr/bin/env python3
"""Generate a JSON list of container-visible input video paths.

The script scans INPUT_VIDEO_PATH on the host, keeps video files, converts each
path to the matching CONTAINER_INPUT_VIDEO_PATH path, and writes the list to
INPUT_VIDEO_LIST_JSON_PATH.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from cosmos_curate.pipelines.video.read_write.input_duration_summary import duration_for, fmt
except ModuleNotFoundError:  # Allow running this file directly during local smoke tests.
    from input_duration_summary import duration_for, fmt


DEFAULT_CONTAINER_INPUT_VIDEO_PATH = "/config/ogame_videos"


DEFAULT_VIDEO_EXTS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".mpeg",
    ".mpg",
)


def env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an input JSON file from a host video directory."
    )
    parser.add_argument(
        "--input-video-path",
        default=env_or_default("INPUT_VIDEO_PATH"),
        help="Host video root to scan. Defaults to INPUT_VIDEO_PATH.",
    )
    parser.add_argument(
        "--container-input-video-path",
        default=env_or_default(
            "CONTAINER_INPUT_VIDEO_PATH", DEFAULT_CONTAINER_INPUT_VIDEO_PATH
        ),
        help=(
            "Container video root to write into JSON. "
            "Defaults to CONTAINER_INPUT_VIDEO_PATH, or "
            f"{DEFAULT_CONTAINER_INPUT_VIDEO_PATH} if unset."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=env_or_default("INPUT_VIDEO_LIST_JSON_PATH"),
        help="Output JSON path. Defaults to INPUT_VIDEO_LIST_JSON_PATH.",
    )
    parser.add_argument(
        "--summary-json",
        default=env_or_default("INPUT_VIDEO_SUMMARY_JSON_PATH"),
        help=(
            "Optional summary JSON path. When set, writes input roots, "
            "video count, total duration, and failed duration paths."
        ),
    )
    parser.add_argument(
        "--ext",
        action="append",
        dest="exts",
        help=(
            "Video extension to include, such as .mp4. "
            "Can be repeated. Defaults to common video extensions."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only write the first N videos after sorting. Useful for tests.",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help=(
            "Directory to scan under the input root. Can be repeated. "
            "Use either a path relative to the input root, such as 10/snowrunner, "
            "or an absolute path. Output paths still stay relative to the input root."
        ),
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help=(
            "Directory to skip while scanning. Can be repeated. "
            "Use either a path relative to the input root, such as 5/Charades, "
            "or an absolute path."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and examples without writing the JSON file.",
    )
    return parser.parse_args()


def normalize_exts(exts: list[str] | None) -> set[str]:
    values = exts if exts else list(DEFAULT_VIDEO_EXTS)
    normalized = set()
    for ext in values:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def normalize_dirs(host_root: Path, dirs: list[str]) -> set[Path]:
    normalized = set()
    for directory in dirs:
        directory = directory.strip()
        if not directory:
            continue

        path = Path(directory).expanduser()
        if not path.is_absolute():
            path = host_root / path
        normalized.add(path.resolve())
    return normalized


def container_to_host_path(container_path: str, container_root: str, host_root: Path) -> Path:
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


def build_summary(
    input_json: Path,
    host_root: Path,
    container_root: str,
    videos: list[str],
    duration_reader=duration_for,
) -> dict:
    total_duration = 0.0
    failed = []

    for video in videos:
        try:
            host_path = container_to_host_path(video, container_root, host_root)
            duration = duration_reader(host_path)
        except Exception as exc:  # Keep summary generation from hiding the input JSON.
            failed.append({"path": video, "reason": str(exc)})
            continue

        if duration is None:
            failed.append({"path": video, "reason": "duration unavailable"})
            continue

        total_duration += float(duration)

    return {
        "input_json_path": str(input_json),
        "input_video_path": str(host_root),
        "container_input_video_path": container_root.rstrip("/"),
        "total_videos": len(videos),
        "total_duration_seconds": round(total_duration, 3),
        "total_duration": fmt(total_duration),
        "failed_duration_count": len(failed),
        "failed_duration_paths": failed,
    }


def collect_video_paths(
    host_root: Path,
    scan_roots: list[Path],
    container_root: str,
    video_exts: set[str],
    excluded_dirs: set[Path],
    limit: int | None,
) -> list[str]:
    container_root = container_root.rstrip("/")
    videos = []

    for scan_root in scan_roots:
        if not scan_root.is_dir():
            print(f"include directory is not a directory, skipping: {scan_root}")
            continue
        for current_root, dirs, files in os.walk(scan_root):
            current_root_path = Path(current_root).resolve()
            dirs.sort()
            files.sort()

            dirs[:] = [
                dirname
                for dirname in dirs
                if current_root_path / dirname not in excluded_dirs
            ]

            for filename in files:
                path = current_root_path / filename
                if path.suffix.lower() not in video_exts:
                    continue

                relative_path = path.relative_to(host_root).as_posix()
                videos.append(f"{container_root}/{relative_path}")

                if limit is not None and len(videos) >= limit:
                    return videos

    return videos


def main() -> int:
    args = parse_args()

    missing = [
        name
        for name, value in (
            ("INPUT_VIDEO_PATH or --input-video-path", args.input_video_path),
            (
                "CONTAINER_INPUT_VIDEO_PATH or --container-input-video-path",
                args.container_input_video_path,
            ),
            ("INPUT_VIDEO_LIST_JSON_PATH or --output-json", args.output_json),
        )
        if not value
    ]
    if missing:
        for name in missing:
            print(f"missing required value: {name}")
        return 2

    host_root = Path(args.input_video_path).expanduser().resolve()
    if not host_root.is_dir():
        print(f"input video path is not a directory: {host_root}")
        return 2

    output_json = Path(args.output_json).expanduser()
    summary_json = Path(args.summary_json).expanduser() if args.summary_json else None
    video_exts = normalize_exts(args.exts)
    included_dirs = normalize_dirs(host_root, args.include_dir)
    excluded_dirs = normalize_dirs(host_root, args.exclude_dir)
    scan_roots = sorted(included_dirs) if included_dirs else [host_root]
    videos = collect_video_paths(
        host_root=host_root,
        scan_roots=scan_roots,
        container_root=args.container_input_video_path,
        video_exts=video_exts,
        excluded_dirs=excluded_dirs,
        limit=args.limit,
    )

    print(f"host input root: {host_root}")
    print(f"container input root: {args.container_input_video_path.rstrip('/')}")
    if included_dirs:
        print("included directories:")
        for directory in sorted(included_dirs):
            print(f"  {directory}")
    if excluded_dirs:
        print("excluded directories:")
        for directory in sorted(excluded_dirs):
            print(f"  {directory}")
    print(f"matched videos: {len(videos)}")
    if videos:
        print("examples:")
        for video in videos[:5]:
            print(f"  {video}")

    if args.dry_run:
        print("dry run: JSON file was not written")
        if summary_json:
            print("dry run: summary JSON file was not written")
        return 0

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote JSON: {output_json}")

    if summary_json:
        summary = build_summary(
            input_json=output_json,
            host_root=host_root,
            container_root=args.container_input_video_path,
            videos=videos,
        )
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"wrote summary JSON: {summary_json}")
        print(
            "summary: "
            f"total_videos={summary['total_videos']} "
            f"total_duration={summary['total_duration']} "
            f"failed_duration_count={summary['failed_duration_count']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
