import uuid

import numpy as np

from cosmos_curate.pipelines.video.pose.pose_utils import (
    VIPE_TO_GT_AXIS_TRANSFORM,
    VIPE_TO_GT_AXIS_TRANSFORM_NAME,
    apply_world_axis_transform,
    build_pose_meta,
    build_relative_pose,
    sample_id_from_window,
)


def test_sample_id_from_window() -> None:
    clip_uuid = uuid.UUID("1b9902f6-9b19-5bf8-adb3-b089912c32f3")
    assert sample_id_from_window(clip_uuid, 0, 255) == "1b9902f6-9b19-5bf8-adb3-b089912c32f3_0_255"


def test_build_relative_pose_normalizes_translation() -> None:
    poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    poses[:, :3, 3] = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0], [10.0, 0.0, 4.0]])
    relative, scale, total_path = build_relative_pose(poses)
    np.testing.assert_allclose(relative[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(relative[:, :3, 3], np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]))
    assert scale == 4.0
    assert total_path == 4.0


def test_apply_world_axis_transform_maps_vipe_to_gt_axes() -> None:
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[1, :3, 3] = np.array([1.0, 2.0, 3.0])
    transformed = apply_world_axis_transform(poses)
    np.testing.assert_allclose(transformed[1, :3, 3], np.array([1.0, 3.0, -2.0]))
    np.testing.assert_allclose(transformed[0], np.eye(4))
    np.testing.assert_allclose(transformed[1, :3, :3], np.eye(3))


def test_build_relative_pose_static_translation_does_not_divide_by_zero() -> None:
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    relative, scale, total_path = build_relative_pose(poses)
    np.testing.assert_allclose(relative, poses)
    assert scale == 1.0
    assert total_path == 0.0


def test_build_pose_meta_has_expected_fields() -> None:
    meta = build_pose_meta(
        clip_uuid="abc",
        start_frame=0,
        end_frame=255,
        num_frames=256,
        status="ok",
        translation_scale_factor=2.0,
        total_path_length_before_norm=3.0,
        quality={"warnings": ["fallback"]},
        num_interpolated_frames=1,
    )
    assert meta["method"] == "vipe"
    assert meta["pose_convention"] == "opencv_c2w"
    assert meta["label_source"] == "vipe"
    assert meta["canonical_coordinate"] == "gt_relative_c2w"
    assert meta["axis_transform"] == VIPE_TO_GT_AXIS_TRANSFORM_NAME
    assert meta["translation_normalization"] == "max_translation_norm"
    assert meta["metric_scale_used"] is False
    assert meta["confidence"] == "pseudo"
    assert meta["relative_pose_rule"]["translation_normalized"] is True
    assert meta["relative_pose_rule"]["axis_transform"] == VIPE_TO_GT_AXIS_TRANSFORM_NAME
    assert meta["quality"]["warnings"] == ["fallback"]
