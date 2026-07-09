#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SHARD_SCRIPT = Path(__file__).with_name("duration_balanced_sharding.py")
DEFAULT_INPUT_JSON = Path("/vepfs-mlp-data-dl-01/linyuxi/input/input_ogame_remaining_3h_plus_1.json")
DEFAULT_OUTPUT_PREFIX = Path("/vepfs-mlp-data-dl-01/linyuxi/input/input_ogame_remaining_3h_plus_1_shard")
DEFAULT_OUTPUT_JSON = Path("/vepfs-mlp-data-dl-01/linyuxi/input/input_ogame_remaining_3h_plus_1_paths.json")
DEFAULT_INPUT_VIDEO_PATH = Path("/vepfs-mlp-data-dl-01/linyuxi/datasets")
DEFAULT_CONTAINER_INPUT_VIDEO_PATH = "/config/ogame_videos"


def write_paths_json(*, output_prefix: Path | str, num_shards: int, output_json: Path | str) -> list[str]:
    prefix = Path(output_prefix).expanduser()
    target = Path(output_json).expanduser()
    paths = [str(Path(f"{prefix}_{index}.json").expanduser().resolve()) for index in range(num_shards)]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def verify_shards(*, output_prefix: Path | str, num_shards: int) -> list[Path]:
    prefix = Path(output_prefix).expanduser()
    paths = [Path(f"{prefix}_{index}.json").expanduser() for index in range(num_shards)]
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing shard json: " + ", ".join(str(path) for path in missing))
    return paths


def write_count_shards(*, input_json: Path | str, output_prefix: Path | str, num_shards: int) -> list[Path]:
    source = Path(input_json).expanduser()
    prefix = Path(output_prefix).expanduser()
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"input JSON must be a list: {source}")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    base_size, remainder = divmod(len(items), num_shards)
    start = 0
    for index in range(num_shards):
        size = base_size + (1 if index < remainder else 0)
        shard_items = items[start : start + size]
        start += size
        output_path = prefix.with_name(f"{prefix.name}_{index}.json")
        output_path.write_text(json.dumps(shard_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shard_paths.append(output_path)
    return shard_paths


def count_shard_and_write_paths(
    *,
    input_json: Path,
    output_prefix: Path,
    num_shards: int,
    output_json: Path,
) -> list[str]:
    print("STEP=count_shard_input_json")
    shard_paths = write_count_shards(input_json=input_json, output_prefix=output_prefix, num_shards=num_shards)
    for shard_json in shard_paths:
        print(f"SHARD_JSON\t{shard_json}")
    print("STEP=json_paths")
    paths = write_paths_json(output_prefix=output_prefix, num_shards=num_shards, output_json=output_json)
    print(f"wrote_json={output_json}")
    print(f"num_paths={len(paths)}")
    print("DONE")
    print(f"input_json={input_json}")
    print(f"shard_prefix={output_prefix}")
    print(f"num_shards={num_shards}")
    print(f"paths_json={output_json}")
    return paths


def run_shard_script(
    *,
    shard_script: Path,
    input_json: Path,
    output_prefix: Path,
    num_shards: int,
    input_video_path: Path,
    container_input_video_path: str,
    keep_original_order: bool,
) -> None:
    cmd = [
        sys.executable,
        str(shard_script),
        "--input-json",
        str(input_json),
        "--output-prefix",
        str(output_prefix),
        "--num-shards",
        str(num_shards),
        "--input-video-path",
        str(input_video_path),
        "--container-input-video-path",
        container_input_video_path,
    ]
    if keep_original_order:
        cmd.append("--keep-original-order")
    subprocess.run(cmd, check=True)


def shard_and_write_paths(
    *,
    input_json: Path,
    output_prefix: Path,
    num_shards: int,
    output_json: Path,
    input_video_path: Path = DEFAULT_INPUT_VIDEO_PATH,
    container_input_video_path: str = DEFAULT_CONTAINER_INPUT_VIDEO_PATH,
    keep_original_order: bool = True,
    shard_script: Path = DEFAULT_SHARD_SCRIPT,
) -> list[str]:
    print("STEP=shard_input_json_by_duration")
    run_shard_script(
        shard_script=shard_script,
        input_json=input_json,
        output_prefix=output_prefix,
        num_shards=num_shards,
        input_video_path=input_video_path,
        container_input_video_path=container_input_video_path,
        keep_original_order=keep_original_order,
    )

    print("STEP=verify_shards")
    for shard_json in verify_shards(output_prefix=output_prefix, num_shards=num_shards):
        print(f"SHARD_JSON\t{shard_json}")

    print("STEP=json_paths")
    paths = write_paths_json(output_prefix=output_prefix, num_shards=num_shards, output_json=output_json)
    print(f"wrote_json={output_json}")
    print(f"num_paths={len(paths)}")
    print("DONE")
    print(f"input_json={input_json}")
    print(f"shard_prefix={output_prefix}")
    print(f"num_shards={num_shards}")
    print(f"paths_json={output_json}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard an input JSON by duration and write shard JSON paths.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--input-video-path", type=Path, default=DEFAULT_INPUT_VIDEO_PATH)
    parser.add_argument("--container-input-video-path", default=DEFAULT_CONTAINER_INPUT_VIDEO_PATH)
    parser.add_argument("--shard-script", type=Path, default=DEFAULT_SHARD_SCRIPT)
    parser.add_argument("--keep-original-order", dest="keep_original_order", action="store_true", default=True)
    parser.add_argument("--no-keep-original-order", dest="keep_original_order", action="store_false")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")

    shard_and_write_paths(
        input_json=args.input_json,
        output_prefix=args.output_prefix,
        num_shards=args.num_shards,
        output_json=args.output_json,
        input_video_path=args.input_video_path,
        container_input_video_path=args.container_input_video_path,
        keep_original_order=args.keep_original_order,
        shard_script=args.shard_script,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
