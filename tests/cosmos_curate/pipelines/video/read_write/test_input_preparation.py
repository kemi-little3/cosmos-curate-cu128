from __future__ import annotations

import json
from pathlib import Path

from cosmos_curate.pipelines.video.read_write import input_json_generator
from cosmos_curate.pipelines.video.read_write import input_shard_list_builder


def test_generate_input_json_and_duration_balanced_shards(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").write_bytes(b"not a real mp4")
    (video_root / "b.mp4").write_bytes(b"not a real mp4")
    (video_root / "note.txt").write_text("skip", encoding="utf-8")

    input_json = tmp_path / "input.json"
    videos = input_json_generator.collect_video_paths(
        host_root=video_root,
        scan_roots=[video_root],
        container_root="/config/ogame_videos",
        video_exts={".mp4"},
        excluded_dirs=set(),
        limit=None,
    )
    input_json.write_text(json.dumps(videos), encoding="utf-8")

    shard_prefix = tmp_path / "input_shard"
    paths_json = tmp_path / "input_paths.json"
    paths = input_shard_list_builder.shard_and_write_paths(
        input_json=input_json,
        output_prefix=shard_prefix,
        num_shards=2,
        output_json=paths_json,
        input_video_path=video_root,
        container_input_video_path="/config/ogame_videos",
    )

    assert videos == [
        "/config/ogame_videos/a.mp4",
        "/config/ogame_videos/b.mp4",
    ]
    assert paths == [
        str((tmp_path / "input_shard_0.json").resolve()),
        str((tmp_path / "input_shard_1.json").resolve()),
    ]
    assert json.loads(paths_json.read_text(encoding="utf-8")) == paths
