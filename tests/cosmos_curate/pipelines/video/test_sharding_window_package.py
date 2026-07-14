import json
from pathlib import Path

from cosmos_curate.pipelines.video.sharding_pipeline import _group_samples_by_count, extract_window_package_samples
from cosmos_curate.pipelines.video.utils.data_model import ClipSample


def test_extract_window_package_samples_reads_package_output(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    video_dir = package_dir / "frames_81" / "video"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "clip_a_0_80.mp4"
    video_path.write_bytes(b"fake-video")
    (package_dir / "package_output.json").write_text(
        json.dumps(
            [
                {
                    "id": "clip_a_0_80",
                    "caption": "caption a",
                    "video_path": "frames_81/video/clip_a_0_80.mp4",
                    "frame_num": 81,
                    "resolution": "640*352",
                    "fps": 16,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = extract_window_package_samples(str(package_dir))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.uuid == "clip_a_0_80"
    assert sample.width == 640
    assert sample.height == 352
    assert sample.num_frames == 81
    assert sample.framerate == 16
    assert sample.num_bytes == len(b"fake-video")
    assert sample.clip_location == video_path
    assert sample.clip_metadata["windows"] == [
        {"openai_caption": "caption a", "start_frame": 0, "end_frame": 80}
    ]
    assert sample.clip_metadata["video_path"] == "frames_81/video/clip_a_0_80.mp4"


def _sample(uuid: str, num_bytes: int) -> ClipSample:
    return ClipSample(
        uuid=uuid,
        width=640,
        height=352,
        num_frames=81,
        framerate=16,
        num_bytes=num_bytes,
        clip_location=Path(f"/tmp/{uuid}.mp4"),
    )


def test_group_samples_by_count_targets_requested_tar_count() -> None:
    samples = [_sample(f"clip-{idx}", 10) for idx in range(4)]

    groups = list(
        _group_samples_by_count(
            samples,
            2,
            drop_small_shards=False,
            min_clips_per_tar=1,
        )
    )

    assert [[sample.uuid for sample in group] for group in groups] == [
        ["clip-0", "clip-1"],
        ["clip-2", "clip-3"],
    ]


def test_group_samples_by_count_does_not_emit_empty_groups() -> None:
    samples = [_sample("clip-0", 10), _sample("clip-1", 10)]

    groups = list(
        _group_samples_by_count(
            samples,
            5,
            drop_small_shards=False,
            min_clips_per_tar=1,
        )
    )

    assert [[sample.uuid for sample in group] for group in groups] == [["clip-0"], ["clip-1"]]
