"""Deterministic PDF page rendering via PyMuPDF (lazy-imported)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..errors import MediaError
from ..models import DocumentOptions, MediaFrame


def parse_page_spec(spec: str | None, total: int) -> list[int]:
    """Parse a 1-indexed page spec like ``"1-10,15,20-25"``.

    None means all pages. The result is sorted, deduplicated, and clamped to
    ``[1, total]``. Malformed specs raise :class:`MediaError`.
    """
    if spec is None or spec.strip() == "":
        return list(range(1, total + 1))

    raw_pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise MediaError(f"malformed page spec: {spec!r}")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise MediaError(f"malformed page spec: {spec!r}")
            try:
                lo, hi = int(bounds[0]), int(bounds[1])
            except ValueError as exc:
                raise MediaError(f"malformed page spec: {spec!r}") from exc
            if lo > hi:
                raise MediaError(
                    f"malformed page spec (descending range): {spec!r}"
                )
            raw_pages.extend(range(lo, hi + 1))
        else:
            try:
                raw_pages.append(int(part))
            except ValueError as exc:
                raise MediaError(f"malformed page spec: {spec!r}") from exc

    result: list[int] = []
    for page in raw_pages:
        clamped = max(1, min(page, total))
        if clamped not in result:
            result.append(clamped)
    return sorted(result)


def _render_pages_sync(path: str, options: DocumentOptions) -> list[MediaFrame]:
    """Synchronous PyMuPDF rendering (runs in a worker thread)."""
    import fitz  # lazy import: pymupdf is only needed for documents

    if not Path(path).exists():
        raise MediaError(f"document not found: {path}")
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise MediaError(f"cannot open document {path}: {exc}") from exc
    if doc.needs_pass:
        raise MediaError(f"document is encrypted: {path}")

    total = doc.page_count
    pages = parse_page_spec(options.pages, total)
    frames: list[MediaFrame] = []
    try:
        for page_number in pages:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=options.dpi, colorspace=fitz.csRGB)
            data = pix.tobytes("jpeg", jpg_quality=85)
            frames.append(
                MediaFrame(
                    image=data,
                    timestamp=None,
                    source=path,
                    metadata={
                        "page": page_number,
                        "total_pages": total,
                        "format": "jpeg",
                    },
                )
            )
    finally:
        doc.close()
    return frames


async def render_pages(
    path: str, options: DocumentOptions
) -> list[MediaFrame]:
    """Render selected PDF pages to JPEG :class:`MediaFrame` objects."""
    return await asyncio.to_thread(_render_pages_sync, path, options)
