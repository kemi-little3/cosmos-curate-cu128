import json
import pathlib
import uuid

import numpy as np

from cosmos_curate.pipelines.video.pose.vipe_pose_stage import VipePoseStage
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window


def _make_task() -> SplitPipeTask:
    window = Window(start_frame=0, end_frame=2, mp4_bytes=b"fake-mp4")
    clip = Clip(uuid=uuid.UUID("1b9902f6-9b19-5bf8-adb3-b089912c32f3"), source_video="source.mp4", span=(0.0, 1.0))
    clip.windows = [window]
    video = Video(input_video=pathlib.Path("source.mp4"))
    video.clips = [clip]
    return SplitPipeTask(session_id="source.mp4", videos=[video])


def test_vipe_pose_stage_populates_window(monkeypatch, tmp_path) -> None:
    task = _make_task()
    stage = VipePoseStage(vipe_python="python", adapter_script="adapter.py")
    assert stage.resources.gpus == 1.0

    monkeypatch.setattr(stage, "_probe_video", lambda _path: (1280, 720, 30.0))

    def fake_run_adapter(clip_dir: pathlib.Path, out_root: pathlib.Path, sample_id: str) -> pathlib.Path:
        out_dir = out_root / sample_id / "vipe"
        out_dir.mkdir(parents=True)
        intrinsics = np.array([[1000.0, 1000.0, 640.0, 360.0]] * 3, dtype=np.float32)
        poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
        poses[:, 2, 3] = [0.0, 1.0, 2.0]
        relative = poses.copy()
        relative[:, 2, 3] = [0.0, 0.5, 1.0]
        np.save(out_dir / "intrinsics.npy", intrinsics)
        np.save(out_dir / "poses_c2w.npy", poses)
        np.save(out_dir / "relative_poses.npy", relative)
        (out_dir / "quality.json").write_text(json.dumps({"status": "ok", "warnings": []}))
        return out_dir

    monkeypatch.setattr(stage, "_run_adapter", fake_run_adapter)
    out = stage.process_data([task])
    assert out is not None
    window = out[0].video.clips[0].windows[0]
    assert window.pose_status == "ok"
    assert window.pose_intrinsics is not None
    assert window.pose_c2w is not None
    assert window.pose_relative is not None
    assert window.pose_intrinsics.shape == (3, 4)
    assert window.pose_c2w.shape == (3, 4, 4)
    assert window.pose_relative.shape == (3, 4, 4)
    np.testing.assert_allclose(window.pose_relative[0], np.eye(4), atol=1e-6)
    assert window.pose_meta is not None
    assert window.pose_meta["source_window"]["end_frame"] == 2


def test_vipe_pose_builder_selects_run_modes() -> None:
    from cosmos_curate.pipelines.video.pose.pose_builders import VipePoseConfig, build_vipe_pose_stages
    from cosmos_curate.pipelines.video.pose.vipe_pose_stage import VipePoseStage
    from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_clip import VipeResidentClipStage
    from cosmos_curate.pipelines.video.pose.vipe_pose_stage_resident_window import VipeResidentWindowStage

    modes = {
        "subprocess-window": VipePoseStage,
        "resident-window": VipeResidentWindowStage,
        "resident-clip": VipeResidentClipStage,
    }
    for mode, expected_type in modes.items():
        stages = build_vipe_pose_stages(VipePoseConfig(vipe_python="python", run_mode=mode))
        stage = stages[0].stage if hasattr(stages[0], "stage") else stages[0]
        assert isinstance(stage, expected_type)


def test_vipe_pose_builder_wraps_fixed_worker_count() -> None:
    from cosmos_curate.pipelines.video.pose.pose_builders import VipePoseConfig, build_vipe_pose_stages
    from cosmos_xenna.pipelines.private.specs import StageSpec

    stages = build_vipe_pose_stages(
        VipePoseConfig(
            vipe_python="python",
            run_mode="resident-window",
            num_workers_per_node=2,
        )
    )

    assert len(stages) == 1
    assert isinstance(stages[0], StageSpec)
    assert stages[0].num_workers_per_node == 2
