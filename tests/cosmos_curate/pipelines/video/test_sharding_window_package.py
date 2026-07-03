import json
from pathlib import Path

from cosmos_curate.pipelines.video.sharding_pipeline import extract_window_package_samples


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
