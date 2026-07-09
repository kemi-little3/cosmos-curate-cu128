#!/usr/bin/env python3
import argparse
import json
import os
import struct
from collections import defaultdict
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".m4v", ".mkv", ".webm"}


def parse_mp4_duration(path):
    size = os.path.getsize(path)

    def read_box_header(f, end):
        pos = f.tell()
        if pos + 8 > end:
            return None
        header = f.read(8)
        if len(header) != 8:
            return None
        box_size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if box_size == 1:
            large = f.read(8)
            if len(large) != 8:
                return None
            box_size = struct.unpack(">Q", large)[0]
            header_size = 16
        elif box_size == 0:
            box_size = end - pos
        if box_size < header_size:
            return None
        return pos, box_size, box_type, header_size

    def walk(f, end):
        while f.tell() + 8 <= end:
            header = read_box_header(f, end)
            if header is None:
                return None
            start, box_size, box_type, header_size = header
            data_start = start + header_size
            data_end = min(start + box_size, end)
            if box_type == b"mvhd":
                f.seek(data_start)
                data = f.read(min(32, data_end - data_start))
                if len(data) < 20:
                    return None
                version = data[0]
                if version == 0 and len(data) >= 20:
                    timescale = struct.unpack(">I", data[12:16])[0]
                    duration = struct.unpack(">I", data[16:20])[0]
                elif version == 1 and len(data) >= 32:
                    timescale = struct.unpack(">I", data[20:24])[0]
                    duration = struct.unpack(">Q", data[24:32])[0]
                else:
                    return None
                return duration / timescale if timescale else None
            if box_type in {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}:
                f.seek(data_start)
                found = walk(f, data_end)
                if found is not None:
                    return found
            f.seek(data_end)
        return None

    with open(path, "rb") as f:
        return walk(f, size)


def parse_avi_duration(path):
    with open(path, "rb") as f:
        data = f.read(1024 * 1024)
    idx = data.find(b"avih")
    if idx < 0 or idx + 8 + 56 > len(data):
        return None
    chunk_size = struct.unpack_from("<I", data, idx + 4)[0]
    if chunk_size < 24:
        return None
    start = idx + 8
    microsec_per_frame = struct.unpack_from("<I", data, start)[0]
    total_frames = struct.unpack_from("<I", data, start + 16)[0]
    if microsec_per_frame and total_frames:
        return microsec_per_frame * total_frames / 1_000_000.0
    return None


def duration_for(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".mov", ".m4v"}:
        return parse_mp4_duration(path)
    if ext == ".avi":
        return parse_avi_duration(path)
    return None


def fmt(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def group_name(root, path):
    rel = os.path.relpath(path, root)
    return rel.split(os.sep, 1)[0] if os.sep in rel else "."


def container_to_host_path(container_path, container_root, host_root):
    container_root = container_root.rstrip("/")
    if container_path == container_root:
        rel = ""
    elif container_path.startswith(f"{container_root}/"):
        rel = container_path[len(container_root) + 1:]
    else:
        raise ValueError(f"path is not under container root {container_root}: {container_path}")
    return str(Path(host_root) / rel)


def iter_directory_videos(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            path = os.path.join(dirpath, name)
            yield path, group_name(root, path)


def iter_json_videos(input_json, host_root, container_root):
    paths = json.loads(Path(input_json).read_text(encoding="utf-8"))
    if not isinstance(paths, list):
        raise ValueError(f"input JSON must be a list: {input_json}")

    for container_path in paths:
        if not isinstance(container_path, str):
            yield str(container_path), "INVALID", "not a string"
            continue
        try:
            host_path = container_to_host_path(container_path, container_root, host_root)
        except ValueError as exc:
            yield container_path, "INVALID", str(exc)
            continue
        yield host_path, group_name(host_root, host_path), None


def summarize(entries):
    totals = defaultdict(float)
    counts = defaultdict(int)
    ext_counts = defaultdict(lambda: defaultdict(int))
    failed = defaultdict(list)

    for entry in entries:
        if len(entry) == 2:
            path, group = entry
            error = None
        else:
            path, group, error = entry

        ext = os.path.splitext(path)[1].lower()
        counts[group] += 1
        ext_counts[group][ext] += 1

        if error is not None:
            failed[group].append(f"{path} ({error})")
            continue

        dur = duration_for(path)
        if dur is None:
            failed[group].append(path)
        else:
            totals[group] += dur

    grand_total = 0.0
    for group in sorted(counts):
        total = totals[group]
        grand_total += total
        ext_text = ", ".join(f"{k}:{v}" for k, v in sorted(ext_counts[group].items()))
        print(f"{group}\tfiles={counts[group]}\tduration_seconds={total:.3f}\tduration={fmt(total)}\texts={ext_text}")
    print(f"TOTAL\tduration_seconds={grand_total:.3f}\tduration={fmt(grand_total)}")

    for group, paths in failed.items():
        print(f"FAILED\t{group}\t{len(paths)}")
        for path in paths[:20]:
            print(f"FAILED_PATH\t{path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize video durations.")
    parser.add_argument("root", nargs="?", default=".", help="Directory to scan.")
    parser.add_argument("--input-json", help="JSON list of container video paths.")
    parser.add_argument(
        "--host-root",
        default="/vepfs-mlp-data-dl-01/linyuxi/datasets",
        help="Host root used to map JSON container paths.",
    )
    parser.add_argument(
        "--container-root",
        default="/config/ogame_videos",
        help="Container root used in JSON paths.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_json:
        summarize(iter_json_videos(args.input_json, args.host_root, args.container_root))
    else:
        summarize(iter_directory_videos(args.root))


if __name__ == "__main__":
    main()
