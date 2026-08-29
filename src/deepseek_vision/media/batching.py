"""Batch sizing and grouping for bounded, context-safe requests."""

from __future__ import annotations

import math

from ..models import MediaFrame

# Hard provider limits.
MAX_IMAGES_PER_REQUEST_LIMIT = 600
MAX_TOTAL_INLINE_IMAGE_BYTES = 64 * 1024 * 1024


def compute_batch_size(
    num_images: int,
    *,
    context_window: int,
    image_token_estimate: int,
    reserved_output_tokens: int,
    prompt_tokens: int,
    safety_factor: float,
    max_images_per_request: int,
    per_image_bytes: int = 1_000_000,
) -> int:
    """Compute the number of images that fit safely in one request.

    The result is bounded by context-token budget, inline-byte budget, the
    provider's 600-image hard cap, and the caller's own per-request cap.
    Never returns less than 1.
    """
    context_budget = context_window * safety_factor - prompt_tokens - reserved_output_tokens
    context_safe = math.floor(context_budget / max(image_token_estimate, 1))
    byte_safe = math.floor(MAX_TOTAL_INLINE_IMAGE_BYTES / max(per_image_bytes, 1))
    return max(
        1,
        min(
            num_images,
            context_safe,
            byte_safe,
            MAX_IMAGES_PER_REQUEST_LIMIT,
            max_images_per_request,
        ),
    )


def make_batches(
    frames: list[MediaFrame], batch_size: int
) -> list[list[MediaFrame]]:
    """Split frames into consecutive batches preserving order."""
    batch_size = max(1, batch_size)
    return [
        frames[i : i + batch_size] for i in range(0, len(frames), batch_size)
    ]


def estimate_per_image_bytes(frames: list[MediaFrame]) -> int:
    """Estimate bytes per image from the largest frame in the set."""
    if not frames:
        return 1_000_000
    return max(len(frame.image) for frame in frames)
