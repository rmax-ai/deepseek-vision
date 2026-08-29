"""Tests for usage accounting and pricing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from deepseek_vision.usage import (
    INPUT_CACHE_HIT_OFFPEAK,
    INPUT_CACHE_MISS_OFFPEAK,
    INPUT_CACHE_MISS_PEAK,
    OUTPUT_OFFPEAK,
    OUTPUT_PEAK,
    UsageTracker,
    is_peak,
)


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_is_peak_weekday_noon_false() -> None:
    assert is_peak(_utc(2026, 8, 24, 12)) is False  # Monday 12:00 UTC


def test_is_peak_monday_0200_true() -> None:
    assert is_peak(_utc(2026, 8, 24, 2)) is True  # Monday 02:00 UTC


def test_is_peak_monday_0500_false() -> None:
    assert is_peak(_utc(2026, 8, 24, 5)) is False  # 05:00 not peak


def test_is_peak_monday_0700_true() -> None:
    assert is_peak(_utc(2026, 8, 24, 7)) is True


def test_is_peak_saturday_0200_false() -> None:
    assert is_peak(_utc(2026, 8, 22, 2)) is False  # Saturday


def test_estimate_cost_offpeak() -> None:
    tracker = UsageTracker()
    tracker.add_request(
        {"prompt_tokens": 1000, "completion_tokens": 100}, latency_s=0.5
    )
    cost = tracker.estimate_cost_usd(now=_utc(2026, 8, 24, 12))
    expected = (
        1000 * INPUT_CACHE_MISS_OFFPEAK + 100 * OUTPUT_OFFPEAK
    ) / 1_000_000
    assert cost == pytest.approx(expected)


def test_estimate_cost_peak() -> None:
    tracker = UsageTracker()
    tracker.add_request(
        {"prompt_tokens": 1000, "completion_tokens": 100}, latency_s=0.5
    )
    cost = tracker.estimate_cost_usd(now=_utc(2026, 8, 24, 2))
    expected = (
        1000 * INPUT_CACHE_MISS_PEAK + 100 * OUTPUT_PEAK
    ) / 1_000_000
    assert cost == pytest.approx(expected)


def test_estimate_cost_cache_hit_mix() -> None:
    tracker = UsageTracker()
    tracker.add_request(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 300,
        },
        latency_s=0.5,
    )
    expected_raw = (
        300 * INPUT_CACHE_HIT_OFFPEAK
        + 700 * INPUT_CACHE_MISS_OFFPEAK
        + 50 * OUTPUT_OFFPEAK
    ) / 1_000_000
    expected = round(expected_raw, 6)  # cost is rounded to 6 decimals
    cost = tracker.estimate_cost_usd(now=_utc(2026, 8, 24, 12))
    assert cost == pytest.approx(expected)


def test_to_summary_sums_everything() -> None:
    tracker = UsageTracker()
    tracker.add_request(
        {"prompt_tokens": 100, "completion_tokens": 20}, latency_s=1.0,
        images=2,
    )
    tracker.add_request(
        {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 30,
        },
        latency_s=2.5,
        retries=1,
        images=1,
    )
    summary = tracker.to_summary()
    assert summary.requests == 2
    assert summary.images_processed == 3
    assert summary.input_tokens == 150
    assert summary.output_tokens == 30
    assert summary.cache_hit_tokens == 30
    assert summary.cache_miss_tokens == 120  # 150 - 30
    assert summary.latency_seconds == pytest.approx(3.5)
    assert summary.retries == 1
    assert summary.from_cache == 0
    assert summary.estimated_cost_usd == pytest.approx(
        tracker.estimate_cost_usd(), abs=1e-9
    )
