"""Tests for batch sizing and grouping."""

from __future__ import annotations

from deepseek_vision.media.batching import (
    compute_batch_size,
    estimate_per_image_bytes,
    make_batches,
)
from deepseek_vision.models import MediaFrame


def _frames(n: int) -> list[MediaFrame]:
    return [
        MediaFrame(image=b"x" * 100, timestamp=float(i), source="s.mp4")
        for i in range(n)
    ]


def test_tiny_context_yields_one() -> None:
    size = compute_batch_size(
        100,
        context_window=1000,
        image_token_estimate=384,
        reserved_output_tokens=16384,
        prompt_tokens=1000,
        safety_factor=0.5,
        max_images_per_request=16,
    )
    assert size == 1


def test_byte_limit_binds() -> None:
    size = compute_batch_size(
        1000,
        context_window=1_000_000,
        image_token_estimate=384,
        reserved_output_tokens=16384,
        prompt_tokens=1000,
        safety_factor=0.5,
        max_images_per_request=16,
        per_image_bytes=64 * 1024 * 1024,  # one image already hits the cap
    )
    assert size == 1


def test_hard_cap_600() -> None:
    size = compute_batch_size(
        100_000,
        context_window=10_000_000_000,
        image_token_estimate=1,
        reserved_output_tokens=0,
        prompt_tokens=0,
        safety_factor=1.0,
        max_images_per_request=10_000,
        per_image_bytes=1000,
    )
    assert size == 600


def test_max_images_per_request_binds() -> None:
    size = compute_batch_size(
        100,
        context_window=1_000_000_000,
        image_token_estimate=1,
        reserved_output_tokens=0,
        prompt_tokens=0,
        safety_factor=1.0,
        max_images_per_request=4,
    )
    assert size == 4


def test_zero_images_yields_one() -> None:
    size = compute_batch_size(
        0,
        context_window=1_000_000,
        image_token_estimate=384,
        reserved_output_tokens=16384,
        prompt_tokens=1000,
        safety_factor=0.5,
        max_images_per_request=16,
    )
    assert size == 1


def test_make_batches_preserves_order_and_total() -> None:
    frames = _frames(10)
    batches = make_batches(frames, 3)
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    flat = [f for b in batches for f in b]
    assert flat == frames
    assert [f.timestamp for f in flat] == list(range(10))


def test_make_batches_last_smaller() -> None:
    batches = make_batches(_frames(7), 4)
    assert [len(b) for b in batches] == [4, 3]


def test_estimate_per_image_bytes() -> None:
    frames = [
        MediaFrame(image=b"a" * 50, source="x"),
        MediaFrame(image=b"b" * 500, source="y"),
    ]
    assert estimate_per_image_bytes(frames) == 500


def test_estimate_per_image_bytes_empty() -> None:
    assert estimate_per_image_bytes([]) == 1_000_000
