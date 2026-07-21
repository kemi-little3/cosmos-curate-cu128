import importlib.util
import json
from pathlib import Path


def _load_package_module():
    data_pipeline_root = Path(__file__).resolve().parents[5]
    package_script = data_pipeline_root / "scripts" / "package.py"
    spec = importlib.util.spec_from_file_location("data_pipeline_package", package_script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sample(run_dir: Path, clip_id: str, *, with_video: bool = True) -> None:
    dataset_dir = run_dir / "cosmos_predict2_video2world_dataset"
    (dataset_dir / "metas").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed_videos").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "metas" / f"{clip_id}.txt").write_text(f"caption {clip_id}", encoding="utf-8")
    if with_video:
        (dataset_dir / "videos" / f"{clip_id}.mp4").write_bytes(b"not-a-real-video")
    (run_dir / "processed_videos" / f"{clip_id}.json").write_text(
        json.dumps(
            {
                "video_uuid": clip_id,
                "video": f"s3://bucket/raw/{clip_id}.mp4",
                "width": 640,
                "height": 360,
                "framerate": 16,
            }
        ),
        encoding="utf-8",
    )


def test_package_skips_failed_sample_and_writes_successful_items(tmp_path, monkeypatch):
    package_module = _load_package_module()
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "package"
    failed_items_jsonl = tmp_path / "failed_items.jsonl"
    monkeypatch.setenv("FAILED_ITEMS_JSONL", str(failed_items_jsonl))

    _write_sample(run_dir, "good_a")
    _write_sample(run_dir, "bad_missing_video", with_video=False)
    _write_sample(run_dir, "good_b")

    package_module.package_runs([run_dir], output_dir)

    package_items = json.loads((output_dir / "package_output.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in package_items] == ["good_a", "good_b"]
    assert {item["source_video"] for item in package_items} == {
        "s3://bucket/raw/good_a.mp4",
        "s3://bucket/raw/good_b.mp4",
    }

    failed_records = [json.loads(line) for line in failed_items_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(failed_records) == 1
    assert failed_records[0]["stage"] == "package"
    assert failed_records[0]["item_id"] == "bad_missing_video"
    assert failed_records[0]["exception_type"] == "FileNotFoundError"
