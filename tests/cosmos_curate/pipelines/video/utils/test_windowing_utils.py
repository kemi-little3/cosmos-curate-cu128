# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for windowing_utils source-timeline mapping helpers."""

import pathlib
from uuid import UUID, uuid4

import pytest

from cosmos_curate.pipelines.video.utils.data_model import Clip, WeightedFrameWindowConfig, Window
from cosmos_curate.pipelines.video.utils.windowing_utils import (
    compute_windows,
    estimate_native_frame_count,
    frame_index_to_source_time_s,
    select_weighted_window_size,
    split_video_into_windows,
    window_source_time_bounds_from_clip,
    window_source_time_bounds_s,
    window_source_time_trace_attributes,
)


def _window_tuples(total_frames: int, window_size: int, *, drop_incomplete_windows: bool) -> list[tuple[int, int]]:
    return [
        (window.start, window.end)
        for window in compute_windows(
            total_frames,
            window_size,
            remainder_threshold=window_size,
            drop_incomplete_windows=drop_incomplete_windows,
        )
    ]


def _make_clip(
    span: tuple[float, float] = (10.0, 20.0),
    windows: list[Window] | None = None,
) -> Clip:
    """Build a minimal ``Clip`` with optional pre-attached windows."""
    clip = Clip(uuid=uuid4(), source_video="s3://bucket/video.mp4", span=span)
    if windows:
        clip.windows.extend(windows)
    return clip


# ---------------------------------------------------------------------------
# compute_windows
# ---------------------------------------------------------------------------


class TestComputeWindows:
    """Window frame ranges for default and strict fixed-size modes."""

    def test_strict_full_windows_only(self) -> None:
        """Strict mode emits only complete fixed-size windows."""
        assert _window_tuples(512, 256, drop_incomplete_windows=True) == [(0, 255), (256, 511)]

    def test_strict_drops_trailing_remainder(self) -> None:
        """Strict mode drops a trailing partial window."""
        assert _window_tuples(511, 256, drop_incomplete_windows=True) == [(0, 255)]

    def test_strict_drops_short_clip(self) -> None:
        """Strict mode emits no windows when the clip is shorter than the target size."""
        assert _window_tuples(255, 256, drop_incomplete_windows=True) == []

    def test_default_can_emit_short_window(self) -> None:
        """Default mode can emit one short window for a short clip."""
        assert _window_tuples(255, 256, drop_incomplete_windows=False) == [(0, 254)]


class TestWeightedFrameWindowSelection:
    """Weighted frame-window selection and emission behavior."""

    def test_excludes_choices_larger_than_total_frames(self) -> None:
        """Selection only considers choices that fit inside the clip."""
        config = WeightedFrameWindowConfig(
            choices=(81, 321, 641, 961),
            weights=(12.0, 3.0, 1.5, 1.0),
            random_seed=1234,
        )
        assert (
            select_weighted_window_size(
                total_frames=320,
                config=config,
                clip_uuid=UUID("00000000-0000-0000-0000-000000000001"),
            )
            == 81
        )

    def test_returns_none_when_no_choice_fits(self) -> None:
        """Selection returns None when the clip is shorter than every configured choice."""
        config = WeightedFrameWindowConfig(
            choices=(81, 321, 641, 961),
            weights=(12.0, 3.0, 1.5, 1.0),
            random_seed=1234,
        )
        assert (
            select_weighted_window_size(
                total_frames=80,
                config=config,
                clip_uuid=UUID("00000000-0000-0000-0000-000000000001"),
            )
            is None
        )

    def test_selection_is_stable_for_same_seed_and_clip_uuid(self) -> None:
        """Selection is deterministic for the same seed and clip UUID."""
        config = WeightedFrameWindowConfig(
            choices=(81, 321, 641, 961),
            weights=(12.0, 3.0, 1.5, 1.0),
            random_seed=1234,
        )
        clip_uuid = UUID("00000000-0000-0000-0000-000000000abc")
        first = select_weighted_window_size(total_frames=2000, config=config, clip_uuid=clip_uuid)
        second = select_weighted_window_size(total_frames=2000, config=config, clip_uuid=clip_uuid)
        assert first == second

    def test_compute_windows_uses_selected_weighted_size_and_drops_tail(self) -> None:
        """Weighted windowing emits complete windows using the selected size."""
        config = WeightedFrameWindowConfig(
            choices=(81,),
            weights=(1.0,),
            random_seed=1234,
        )
        windows = compute_windows(
            total_frames=250,
            window_size=256,
            remainder_threshold=128,
            drop_incomplete_windows=True,
            weighted_frame_window_config=config,
            clip_uuid=UUID("00000000-0000-0000-0000-000000000001"),
        )
        assert [(w.start, w.end) for w in windows] == [(0, 80), (81, 161), (162, 242)]

    def test_compute_windows_returns_empty_when_no_weighted_choice_fits(self) -> None:
        """Weighted windowing emits no windows when no configured size fits."""
        config = WeightedFrameWindowConfig(
            choices=(81, 321),
            weights=(1.0, 1.0),
            random_seed=1234,
        )
        windows = compute_windows(
            total_frames=80,
            window_size=256,
            remainder_threshold=128,
            drop_incomplete_windows=True,
            weighted_frame_window_config=config,
            clip_uuid=UUID("00000000-0000-0000-0000-000000000001"),
        )
        assert windows == []

    def test_fixed_compute_windows_behavior_is_unchanged(self) -> None:
        """Legacy non-weighted windowing behavior is preserved."""
        windows = compute_windows(
            total_frames=511,
            window_size=256,
            remainder_threshold=128,
            drop_incomplete_windows=False,
        )
        assert [(w.start, w.end) for w in windows] == [(0, 255), (256, 510)]

    def test_single_partial_window_is_trimmed_when_returning_mp4_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single window that does not cover the full clip must still be cut with ffmpeg."""
        config = WeightedFrameWindowConfig(choices=(641,), weights=(1.0,), random_seed=1234)
        expected_bytes = b"trimmed-window"
        calls: list[list[str]] = []

        def fake_check_call(command: list[str]) -> None:
            calls.append(command)
            pathlib.Path(command[-1]).write_bytes(expected_bytes)

        monkeypatch.setattr(
            "cosmos_curate.pipelines.video.utils.windowing_utils.get_frame_count",
            lambda _: 1024,
        )
        monkeypatch.setattr(
            "cosmos_curate.pipelines.video.utils.windowing_utils.subprocess.check_call",
            fake_check_call,
        )

        mp4_bytes, frames, windows = split_video_into_windows(
            b"full-clip-bytes",
            return_bytes=True,
            return_video_frames=False,
            weighted_frame_window_config=config,
            clip_uuid=UUID("00000000-0000-0000-0000-000000000001"),
        )

        assert mp4_bytes == [expected_bytes]
        assert frames == [None]
        assert [(window.start, window.end) for window in windows] == [(0, 640)]
        assert calls


# ---------------------------------------------------------------------------
# estimate_native_frame_count
# ---------------------------------------------------------------------------


class TestEstimateNativeFrameCount:
    """``estimate_native_frame_count`` picks the best N from available data."""

    def test_two_partitioned_windows(self) -> None:
        """Contiguous partition 0..9, 10..19 yields N=20."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(windows=[w0, w1])
        assert estimate_native_frame_count(clip) == 20

    def test_single_window(self) -> None:
        """One window 0..127 yields N=128."""
        w = Window(start_frame=0, end_frame=127)
        clip = _make_clip(windows=[w])
        assert estimate_native_frame_count(clip) == 128

    def test_empty_windows_with_fallback(self) -> None:
        """No clip.windows — falls back to fallback_window.end_frame + 1."""
        clip = _make_clip()
        w = Window(start_frame=0, end_frame=4)
        assert estimate_native_frame_count(clip, fallback_window=w) == 5

    def test_empty_windows_no_fallback_returns_one(self) -> None:
        """No clip.windows and no fallback — degenerate case returns 1."""
        clip = _make_clip()
        assert estimate_native_frame_count(clip) == 1

    def test_fallback_ignored_when_clip_has_windows(self) -> None:
        """Fallback window is not used when clip.windows is populated."""
        w0 = Window(start_frame=0, end_frame=9)
        clip = _make_clip(windows=[w0])
        fallback = Window(start_frame=0, end_frame=99)
        assert estimate_native_frame_count(clip, fallback_window=fallback) == 10


# ---------------------------------------------------------------------------
# frame_index_to_source_time_s
# ---------------------------------------------------------------------------


class TestFrameIndexToSourceTimeS:
    """``frame_index_to_source_time_s`` maps one frame to a source second."""

    def test_first_frame(self) -> None:
        """Frame 0 maps to clip start."""
        assert frame_index_to_source_time_s((10.0, 20.0), 0, 20) == 10.0

    def test_last_frame(self) -> None:
        """Frame N-1 maps to clip end."""
        assert frame_index_to_source_time_s((10.0, 20.0), 19, 20) == 20.0

    def test_midpoint(self) -> None:
        """Middle frame of 21 frames maps to exact midpoint of 10s span."""
        result = frame_index_to_source_time_s((0.0, 10.0), 10, 21)
        assert result == pytest.approx(5.0)

    def test_single_frame_denominator(self) -> None:
        """N=1 clamps denominator to 1; result equals t0."""
        assert frame_index_to_source_time_s((5.0, 15.0), 0, 1) == 5.0

    def test_zero_length_span(self) -> None:
        """Degenerate span always returns t0 regardless of index."""
        assert frame_index_to_source_time_s((7.0, 7.0), 5, 10) == 7.0


# ---------------------------------------------------------------------------
# window_source_time_bounds_s
# ---------------------------------------------------------------------------


class TestWindowSourceTimeBoundsS:
    """``window_source_time_bounds_s`` linearly maps start/end frames."""

    def test_first_window_of_twenty_frames(self) -> None:
        """Window 0..9 on span (10, 20) with N=20."""
        t0, t1 = window_source_time_bounds_s((10.0, 20.0), 0, 9, 20)
        assert t0 == 10.0
        assert t1 == pytest.approx(10.0 + (9 / 19) * 10.0)

    def test_second_window_of_twenty_frames(self) -> None:
        """Window 10..19 on span (10, 20) with N=20 ends at clip end."""
        t0, t1 = window_source_time_bounds_s((10.0, 20.0), 10, 19, 20)
        assert t0 == pytest.approx(10.0 + (10 / 19) * 10.0)
        assert t1 == pytest.approx(20.0)

    def test_single_native_frame(self) -> None:
        """N=1 maps both bounds to clip start."""
        t0, t1 = window_source_time_bounds_s((5.0, 15.0), 0, 0, 1)
        assert t0 == 5.0
        assert t1 == 5.0

    def test_zero_length_clip_span(self) -> None:
        """Degenerate span yields constant times."""
        t0, t1 = window_source_time_bounds_s((3.0, 3.0), 0, 5, 10)
        assert t0 == 3.0
        assert t1 == 3.0

    def test_full_range_window_covers_entire_span(self) -> None:
        """A single window covering 0..(N-1) maps exactly to clip start/end."""
        t0, t1 = window_source_time_bounds_s((0.0, 60.0), 0, 99, 100)
        assert t0 == pytest.approx(0.0)
        assert t1 == pytest.approx(60.0)


class TestWindowSourceTimeBoundsFromClip:
    """``window_source_time_bounds_from_clip`` combines N estimation with mapping."""

    def test_end_to_end_with_two_windows(self) -> None:
        """First window of a 10s clip (span 10..20, 20 native frames)."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(span=(10.0, 20.0), windows=[w0, w1])

        t0, t1 = window_source_time_bounds_from_clip(clip, w0)
        assert t0 == 10.0
        assert t1 == pytest.approx(10.0 + (9 / 19) * 10.0)

        t0_b, t1_b = window_source_time_bounds_from_clip(clip, w1)
        assert t1_b == pytest.approx(20.0)
        # Second window starts where first ended (continuous timeline).
        assert t0_b == pytest.approx(10.0 + (10 / 19) * 10.0)

    def test_fallback_when_clip_windows_empty(self) -> None:
        """Uses the passed window for N estimation when clip.windows is empty."""
        w = Window(start_frame=0, end_frame=4)
        clip = _make_clip(span=(0.0, 5.0))

        t0, t1 = window_source_time_bounds_from_clip(clip, w)
        assert t0 == 0.0
        assert t1 == pytest.approx(5.0)


class TestWindowSourceTimeTraceAttributes:
    """``window_source_time_trace_attributes`` returns OTel-safe dict."""

    def test_returns_all_expected_keys(self) -> None:
        """Dict contains source times, clip span, and human-readable bounds."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(span=(10.0, 20.0), windows=[w0, w1])

        attrs = window_source_time_trace_attributes(clip, w0)

        assert set(attrs.keys()) == {
            "window.source_start_s",
            "window.source_end_s",
            "window.clip_span_start_s",
            "window.clip_span_end_s",
            "window.source_bounds",
        }
        assert attrs["window.source_start_s"] == 10.0
        assert attrs["window.source_end_s"] == pytest.approx(10.0 + (9 / 19) * 10.0)
        assert attrs["window.clip_span_start_s"] == 10.0
        assert attrs["window.clip_span_end_s"] == 20.0
        assert isinstance(attrs["window.source_bounds"], str)

    def test_returns_empty_dict_on_bad_clip(self) -> None:
        """Malformed clip data does not raise -- returns empty dict."""
        clip = _make_clip(span=(10.0, 20.0))
        bad_window = Window(start_frame=0, end_frame=-1)
        result = window_source_time_trace_attributes(clip, bad_window)
        assert isinstance(result, dict)
