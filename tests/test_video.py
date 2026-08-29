"""Tests for video probing, frame extraction, and windowing."""

from __future__ import annotations

import math

import pytest

from deepseek_vision.media.video import (
    extract_frames,
    frames_within_duration,
    probe,
    window_frames,
)
from deepseek_vision.models import MediaFrame, VideoOptions


async def test_probe(sample_video: str) -> None:
    info = await probe(sample_video)
    assert info["duration"] == pytest.approx(4.0, abs=0.5)
    assert info["fps"] is not None and info["fps"] > 0
    assert info["width"] == 320
    assert info["height"] == 240


async def test_probe_missing_file() -> None:
    from deepseek_vision.errors import MediaError

    with pytest.raises(MediaError):
        await probe("/nonexistent/missing.mp4")


async def test_uniform_count_order_and_start(sample_video: str) -> None:
    info = await probe(sample_video)
    duration = info["duration"] or 4.0
    options = VideoOptions(sampling="uniform", fps=1.0, max_frames=128)
    frames = await extract_frames(sample_video, options)
    effective_fps = min(1.0, options.max_frames / duration)
    expected = min(options.max_frames, math.ceil(duration * effective_fps))
    assert len(frames) == expected
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=1e-6)
    assert all(f.source == sample_video for f in frames)


async def test_uniform_max_frames_cap(sample_video: str) -> None:
    options = VideoOptions(sampling="uniform", fps=10.0, max_frames=4)
    frames = await extract_frames(sample_video, options)
    assert len(frames) <= 4


async def test_scene_change_finds_cut_and_bookends(sample_video: str) -> None:
    options = VideoOptions(sampling="scene_change", scene_threshold=0.3)
    frames = await extract_frames(sample_video, options)
    timestamps = [f.timestamp for f in frames]
    assert any(abs(t - 2.0) <= 0.3 for t in timestamps)
    assert min(timestamps) == pytest.approx(0.0, abs=0.1)
    assert max(timestamps) == pytest.approx(4.0, abs=0.3)


async def test_keyframes_nonempty_sorted_unique(sample_video: str) -> None:
    options = VideoOptions(sampling="keyframes")
    frames = await extract_frames(sample_video, options)
    assert len(frames) > 0
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


async def test_adaptive_bounded_with_bookends(sample_video: str) -> None:
    options = VideoOptions(sampling="adaptive", max_frames=16)
    frames = await extract_frames(sample_video, options)
    assert len(frames) <= options.max_frames
    timestamps = [f.timestamp for f in frames]
    assert min(timestamps) == pytest.approx(0.0, abs=0.1)
    assert max(timestamps) == pytest.approx(4.0, abs=0.3)
    assert timestamps == sorted(timestamps)


async def test_start_end_clip_respected(sample_video: str) -> None:
    options = VideoOptions(
        sampling="uniform",
        fps=2.0,
        start_seconds=1.0,
        end_seconds=2.0,
        max_frames=64,
    )
    frames = await extract_frames(sample_video, options)
    assert len(frames) > 0
    for frame in frames:
        assert 1.0 - 1e-6 <= frame.timestamp <= 3.0


def test_window_frames_partitions() -> None:
    frames = [
        MediaFrame(image=b"x", timestamp=i * 0.5, source="v.mp4")
        for i in range(8)  # 0.0 .. 3.5
    ]
    windows = window_frames(frames, 1.5)
    assert len(windows) >= 3
    flat = [f for window in windows for f in window]
    assert flat == sorted(flat, key=lambda f: f.timestamp or 0.0)
    bounds = [
        min(f.timestamp or 0.0 for f in window) for window in windows
    ]
    assert bounds == sorted(bounds)


def test_window_frames_empty_input() -> None:
    assert window_frames([], 1.5) == []


def test_frames_within_duration() -> None:
    frames = [
        MediaFrame(image=b"x", timestamp=ts, source="v.mp4")
        for ts in (0.5, 1.0, 2.0, 3.5)
    ]
    within = frames_within_duration(frames, 1.0, 3.0)
    assert [f.timestamp for f in within] == [1.0, 2.0]
