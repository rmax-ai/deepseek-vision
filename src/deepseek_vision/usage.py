"""Token, cost, and latency accounting.

Pricing constants are per 1M tokens USD and hard-coded from the verified
provider contract (Aug 2026).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import UsageSummary

# Per-1M-token USD pricing.
INPUT_CACHE_HIT_OFFPEAK = 0.007
INPUT_CACHE_HIT_PEAK = 0.014
INPUT_CACHE_MISS_OFFPEAK = 0.22
INPUT_CACHE_MISS_PEAK = 0.44
OUTPUT_OFFPEAK = 0.66
OUTPUT_PEAK = 1.32

# Peak hours: Mon-Fri 01:00-04:00 and 06:00-10:00 UTC.
PEAK_UTC_HOURS = {1, 2, 3, 6, 7, 8, 9}

_PER_MILLION = 1_000_000


def is_peak(now: datetime) -> bool:
    """True when ``now`` (UTC) falls inside peak pricing hours."""
    return now.weekday() < 5 and now.hour in PEAK_UTC_HOURS


class UsageTracker:
    """Aggregates per-request usage into a :class:`UsageSummary`."""

    def __init__(self) -> None:
        self.requests = 0
        self.images_processed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.latency_seconds = 0.0
        self.retries = 0
        self.from_cache = 0

    def add_request(
        self,
        usage: dict,
        latency_s: float,
        retries: int = 0,
        from_cache: bool = False,
        images: int = 0,
    ) -> None:
        """Record one API request's usage dict."""
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        self.requests += 1
        self.images_processed += images
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        self.latency_seconds += latency_s
        self.retries += retries
        if from_cache:
            self.from_cache += 1

        hit = usage.get("prompt_cache_hit_tokens", 0) or 0
        self.cache_hit_tokens += hit
        if "prompt_cache_hit_tokens" in usage:
            miss = max(prompt_tokens - hit, 0)
        else:
            miss = usage.get("prompt_cache_miss_tokens", prompt_tokens) or 0
        self.cache_miss_tokens += miss

    def estimate_cost_usd(self, now: datetime | None = None) -> float:
        """Estimate cost from accumulated tokens using peak/off-peak prices."""
        now = now or datetime.now(timezone.utc)
        peak = is_peak(now)
        hit_price = INPUT_CACHE_HIT_PEAK if peak else INPUT_CACHE_HIT_OFFPEAK
        miss_price = (
            INPUT_CACHE_MISS_PEAK if peak else INPUT_CACHE_MISS_OFFPEAK
        )
        output_price = OUTPUT_PEAK if peak else OUTPUT_OFFPEAK

        hit_tokens = min(self.cache_hit_tokens, self.input_tokens)
        miss_tokens = self.input_tokens - hit_tokens
        cost = (
            hit_tokens * hit_price
            + miss_tokens * miss_price
            + self.output_tokens * output_price
        ) / _PER_MILLION
        return round(cost, 6)

    def to_summary(self) -> UsageSummary:
        """Return a snapshot :class:`UsageSummary` of all recorded usage."""
        return UsageSummary(
            requests=self.requests,
            images_processed=self.images_processed,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_hit_tokens=self.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens,
            estimated_cost_usd=self.estimate_cost_usd(),
            latency_seconds=self.latency_seconds,
            retries=self.retries,
            from_cache=self.from_cache,
        )
