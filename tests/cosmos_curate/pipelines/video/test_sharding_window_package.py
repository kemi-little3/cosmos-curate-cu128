import json
from pathlib import Path

from cosmos_curate.pipelines.video.read_write.download_stages import DownloadPackUpload
from cosmos_curate.pipelines.video.sharding_pipeline import (
    _build_flat_sample_task,
    _group_samples_by_count,
    extract_window_package_samples,
)
from cosmos_curate.pipelines.video.utils.data_model import ClipSample, ShardPipeTask


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


def test_video_manifest_writes_per_file_metadata(tmp_path: Path) -> None:
    sample = ClipSample(
        uuid="clip_a_0_320",
        width=640,
        height=352,
        num_frames=321,
        framerate=16,
        num_bytes=1234,
        clip_location=tmp_path / "clip_a_0_320.mp4",
        clip_metadata={
            "width": 640,
            "height": 352,
            "num_frames": 321,
            "framerate": 16,
            "num_bytes": 1234,
        },
    )
    output_tar = tmp_path / "datasets" / "000000.tar"
    task = ShardPipeTask(
        bin_path=str(tmp_path),
        part_num=0,
        samples=[sample],
        output_tar_video=output_tar,
        output_tar_metas=tmp_path / "metas" / "000000.tar",
        output_tar_t5_xxl=tmp_path / "t5_xxl" / "000000.tar",
        key_count=0,
        write_auxiliary_tars=False,
    )
    stage = DownloadPackUpload(
        input_path=str(tmp_path),
        output_path=str(tmp_path),
        input_s3_profile_name="default",
        output_s3_profile_name="default",
    )

    stage._write_video_manifest(task)

    manifest = json.loads(output_tar.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest == {
        "files": [
            {
                "name": "clip_a_0_320.mp4",
                "meta": {
                    "fps": 16,
                    "width": 640,
                    "height": 352,
                    "num_frames": 321,
                    "num_bytes": 1234,
                    "duration_seconds": 20.0625,
                },
            }
        ]
    }


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


def test_build_flat_sample_task_randomly_samples_twenty_clips(tmp_path: Path) -> None:
    samples = [_sample(f"clip-{idx}", idx + 1) for idx in range(25)]
    for sample in samples:
        sample.clip_metadata["source_video"] = f"video-{sample.uuid}.mp4"
        sample.encoded_data = f"video-bytes-{sample.uuid}".encode()

    task = _build_flat_sample_task(samples, output_path=str(tmp_path), sample_count=20)

    assert task is not None
    assert len(task.samples) == 20
    assert task.output_tar_video == tmp_path / "sample_datasets" / "000000.tar"
    assert task.output_tar_metas == tmp_path / "sample_metas" / "000000.tar"
    assert task.output_tar_t5_xxl == tmp_path / "sample_t5_xxl" / "000000.tar"
    assert task.write_auxiliary_tars is False
    assert {sample.uuid for sample in task.samples}.issubset({sample.uuid for sample in samples})

    selected = task.samples[0]
    original = next(sample for sample in samples if sample.uuid == selected.uuid)
    assert selected is not original
    assert selected.encoded_data is not original.encoded_data
    assert bytes(selected.encoded_data.resolve()) == bytes(original.encoded_data.resolve())
    selected.clip_metadata["mutated"] = True
    assert "mutated" not in original.clip_metadata


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
