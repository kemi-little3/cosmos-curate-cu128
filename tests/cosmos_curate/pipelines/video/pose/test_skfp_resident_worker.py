from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cosmos_curate.pipelines.video.pose.skfp_resident_common import (
    SkfpResidentClient,
    attach_skfp_clip_output_slice_to_window,
    summarize_resident_worker_report,
)
from cosmos_curate.pipelines.video.utils.data_model import Window


def _write_fake_skfp(root: Path, *, fail: bool = False) -> None:
    pkg = root / "sparse_keyframe_pose"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    body = """
from pathlib import Path
import json
import numpy as np

class Dense:
    def __init__(self, out):
        self.poses_c2w = np.load(out / 'dense' / 'poses.npy')
        self.relative_poses = np.load(out / 'dense' / 'relative_poses.npy')
        self.intrinsics = np.load(out / 'dense' / 'intrinsics.npy')
        self.pose_meta = json.loads((out / 'dense' / 'pose_meta.json').read_text())

class Result:
    def __init__(self, out):
        self.pose_status = 'ok'
        self.output_dir = out
        self.dense = Dense(out)

def run_sparse_keyframe_pose(**kwargs):
    if __FAIL__:
        raise RuntimeError('fake skfp failure')
    out = Path(kwargs['output_dir'])
    dense = out / 'dense'
    dense.mkdir(parents=True, exist_ok=True)
    n = int(kwargs.get('num_frames') or 3)
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n, dtype=np.float32)
    intrinsics = np.tile(np.array([[1000, 1000, 640, 360]], dtype=np.float32), (n, 1))
    np.save(dense / 'poses.npy', poses)
    np.save(dense / 'relative_poses.npy', poses)
    np.save(dense / 'intrinsics.npy', intrinsics)
    (dense / 'pose_meta.json').write_text(json.dumps({
        'method': 'sparse_keyframe_pose',
        'base_estimator': 'vipe',
        'keyframe_stride': kwargs.get('stride'),
        'num_dense_frames': n,
    }))
    return Result(out)
""".replace("__FAIL__", "True" if fail else "False")
    (pkg / "runner.py").write_text(body)


def _write_clip_input(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "clip.mp4").write_bytes(b"fake")
    np.save(path / "frame_indices.npy", np.arange(3, dtype=np.int32))
    np.save(path / "timestamps.npy", np.arange(3, dtype=np.float64))
    (path / "meta.json").write_text(json.dumps({"num_raw_frames": 3, "sampled_fps": 30.0}))


def test_skfp_resident_client_runs_fake_worker(tmp_path: Path) -> None:
    skfp_root = tmp_path / "skfp"
    _write_fake_skfp(skfp_root)
    clip_dir = tmp_path / "input" / "clip_a"
    _write_clip_input(clip_dir)
    out_root = tmp_path / "output"

    client = SkfpResidentClient(
        skfp_python="python",
        skfp_root=str(skfp_root),
        vipe_python="python",
        vipe_adapter_script="adapter.py",
        vipe_work_root=str(tmp_path / "work"),
        stride=16,
        min_keyframes=3,
        max_keyframes=64,
    )
    try:
        client.start()
        out = client.run_job(clip_dir=clip_dir, out_root=out_root, clip_uuid="clip_a", num_frames=3)
    finally:
        client.stop()

    assert out == out_root / "clip_a" / "skfp"
    assert (out / "dense" / "poses.npy").exists()
    assert (out / "dense" / "intrinsics.npy").exists()
    meta = json.loads((out / "dense" / "pose_meta.json").read_text())
    assert meta["base_estimator"] == "vipe"
    assert meta["keyframe_stride"] == 16


def test_skfp_resident_client_raises_on_worker_error(tmp_path: Path) -> None:
    skfp_root = tmp_path / "skfp"
    _write_fake_skfp(skfp_root, fail=True)
    clip_dir = tmp_path / "input" / "clip_a"
    _write_clip_input(clip_dir)

    client = SkfpResidentClient(
        skfp_python="python",
        skfp_root=str(skfp_root),
        vipe_python="python",
        vipe_adapter_script="adapter.py",
        vipe_work_root=str(tmp_path / "work"),
    )
    try:
        client.start()
        with pytest.raises(RuntimeError, match="resident SKFP job failed"):
            client.run_job(clip_dir=clip_dir, out_root=tmp_path / "output", clip_uuid="clip_a", num_frames=3)
    finally:
        client.stop()


def test_attach_skfp_clip_output_slice_to_window_uses_window_bounds(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    dense.mkdir()
    poses = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    poses[:, 2, 3] = np.arange(6, dtype=np.float32)
    intrinsics = np.tile(np.array([[1000, 1000, 640, 360]], dtype=np.float32), (6, 1))
    np.save(dense / "poses.npy", poses)
    np.save(dense / "intrinsics.npy", intrinsics)
    (dense / "pose_meta.json").write_text(json.dumps({"warnings": [], "status": "ok"}))

    window = Window(start_frame=2, end_frame=4)
    attach_skfp_clip_output_slice_to_window(
        window,
        clip_uuid="clip-a",
        skfp_out=tmp_path,
        quality_extra={"skfp_stride": 4},
    )

    assert window.pose_status == "ok"
    assert window.pose_intrinsics is not None
    assert window.pose_c2w is not None
    assert window.pose_relative is not None
    assert window.pose_intrinsics.shape == (3, 4)
    assert window.pose_c2w.shape == (3, 4, 4)
    np.testing.assert_allclose(window.pose_c2w[:, 2, 3], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(window.pose_relative[:, 1, 3], [0.0, 1.0, 2.0])
    assert window.pose_meta is not None
    assert window.pose_meta["method"] == "skfp_vipe"
    assert window.pose_meta["label_source"] == "skfp_vipe"
    adapter_quality = window.pose_meta["quality"]["adapter_quality"]
    assert adapter_quality["runner"] == "skfp_resident_clip_once_slice"
    assert adapter_quality["slice_source_window"] == {"start_frame": 2, "end_frame": 4}
    assert adapter_quality["skfp_stride"] == 4


def test_summarize_resident_worker_report_keeps_only_production_fields() -> None:
    report = {
        "mode": "internal_stage_worker",
        "report_json": "/tmp/report.json",
        "jobs_jsonl": "/tmp/jobs.jsonl",
        "stage_worker_log": "/tmp/stage_worker.log",
        "gpu_samples_csv": "/tmp/gpu.csv",
        "worker_startup_s": 3.0,
        "worker_total_s": 15.0,
        "jobs_total_s": 12.0,
        "job_count": 1,
        "jobs": [
            {
                "clip_uuid": "clip-a",
                "clip_dir": "/tmp/input/clip-a",
                "output_dir": "/tmp/out/clip-a/vipe",
                "input_mode": "frame_dir",
                "input_path": "/tmp/input/clip-a/images",
                "job_total_s": 10.0,
                "timings": {
                    "compose_config_s": 0.1,
                    "make_pipeline_s": 0.01,
                    "pipeline_run_s": 9.5,
                    "artifact_convert_s": 0.2,
                    "slam_pass1_s": 4.0,
                },
            }
        ],
    }

    summary = summarize_resident_worker_report(report)

    assert summary == {
        "mode": "internal_stage_worker",
        "job_count": 1,
        "worker_startup_s": 3.0,
        "worker_total_s": 15.0,
        "jobs_total_s": 12.0,
        "jobs": [
            {
                "clip_uuid": "clip-a",
                "input_mode": "frame_dir",
                "job_total_s": 10.0,
                "pipeline_run_s": 9.5,
                "artifact_convert_s": 0.2,
                "status": "ok",
            }
        ],
    }
    assert "stage_worker_log" not in summary
    assert "gpu_samples_csv" not in summary
    assert "timings" not in summary["jobs"][0]



def _write_fake_vipe_stage_worker(pose_root: Path) -> Path:
    script = pose_root / "scripts" / "vipe_resident_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--jobs-jsonl', type=Path, required=True)
    parser.add_argument('--report-json', type=Path, required=True)
    parser.add_argument('--gpu-samples-csv', type=Path)
    parser.add_argument('--vipe-repo', type=Path, required=True)
    parser.add_argument('--gpu-index')
    parser.add_argument('--gpu-sample-interval')
    parser.add_argument('--report-level', default='summary')
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.jobs_jsonl.read_text().splitlines() if line.strip()]
    report = {'mode': 'fake_stage_worker', 'jobs': [], 'job_count': len(jobs)}
    for job in jobs:
        clip_dir = Path(job['clip_dir'])
        out = Path(job['out_root']) / job['clip_uuid'] / 'vipe'
        out.mkdir(parents=True, exist_ok=True)
        frame_indices = np.load(clip_dir / 'frame_indices.npy').astype(np.int64)
        n = len(frame_indices)
        poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        poses[:, 2, 3] = frame_indices.astype(np.float32)
        intrinsics = np.tile(np.array([[1000, 1000, 640, 360]], dtype=np.float32), (n, 1))
        np.save(out / 'poses_c2w.npy', poses)
        np.save(out / 'intrinsics.npy', intrinsics)
        np.save(out / 'frame_indices.npy', frame_indices)
        (out / 'quality.json').write_text(json.dumps({'fake': True, 'num_frames': n}))
        report['jobs'].append({'clip_uuid': job['clip_uuid'], 'job_total_s': 0.01, 'timings': {'fake_s': 0.01}})
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
"""
    )
    adapter = pose_root / "adapter" / "scripts" / "run_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# fake adapter\n")
    (pose_root / "repo" / "vipe").mkdir(parents=True)
    return adapter


def test_skfp_resident_client_runs_sparse_stage_jobs_with_vipe_resident_worker(tmp_path: Path) -> None:
    skfp_root = tmp_path / "skfp"
    _write_fake_skfp(skfp_root)
    adapter = _write_fake_vipe_stage_worker(tmp_path / "pose_estimation")
    clip_dir = tmp_path / "input" / "clip_a"
    _write_clip_input(clip_dir)
    out_root = tmp_path / "worker_outputs"

    client = SkfpResidentClient(
        skfp_python="python",
        skfp_root=str(skfp_root),
        vipe_python="python",
        vipe_adapter_script=str(adapter),
        vipe_work_root=str(tmp_path / "work"),
    )
    try:
        client.start()
        report = client.run_stage_jobs(
            jobs=[{"clip_dir": str(clip_dir), "out_root": str(out_root), "clip_uuid": "clip_a"}],
            label="clip_once",
        )
    finally:
        client.stop()

    method_dir = out_root / "clip_a" / "vipe"
    assert report["job_count"] == 1
    assert "stage_worker_log" not in report
    assert "gpu_samples_csv" not in report
    assert report["jobs"] == [{"clip_uuid": "clip_a", "job_total_s": 0.01, "status": "ok"}]
    assert (method_dir / "poses_c2w.npy").exists()
    np.testing.assert_array_equal(np.load(method_dir / "frame_indices.npy"), np.arange(3))
