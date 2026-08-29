"""Live API integration tests.

Skipped unless ``DEEPSEEK_API_KEY`` is set. These tests make real calls to
the DeepSeek API and assert STRUCTURAL properties only (request counts,
frame counts, token accounting, schema types). LLM outputs are stochastic,
so semantic content is never asserted; suspicious values are surfaced via
``print`` warnings instead.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from PIL import Image, ImageDraw
from pydantic import BaseModel

from deepseek_vision import analyze_media
from deepseek_vision.models import CompareResult
from deepseek_vision.presets.documents import DocumentAnalysis

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set; live integration tests skipped",
)

# Shared session-scoped options and client settings for all live tests.
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_OPTIONS: dict = {"use_cache": False}


def _opts(**extra: object) -> dict:
    opts = dict(BASE_OPTIONS)
    opts.update(extra)
    return opts


def _invoice_image(path) -> None:
    """Invoice-like fixture generated inline (no committed fixtures)."""
    img = Image.new("RGB", (500, 400), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "INVOICE",
        "Invoice No: INV-2026-0042",
        "Date: 2026-08-20",
        "Vendor: Acme Widgets Ltd",
        "Total: $1,234.50",
        "Currency: USD",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 45
    img.save(path)


class Invoice(BaseModel):
    vendor: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    total: float | None = None
    currency: str | None = None


@pytest.fixture(scope="session")
def live_video(tmp_path_factory) -> str:
    """4s 320x240 10fps two-color video (black then red)."""
    out = tmp_path_factory.mktemp("live") / "two_color.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "10",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        fallback = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
            "-c:v", "mpeg4", "-q:v", "3", "-r", "10",
            str(out),
        ]
        subprocess.run(fallback, capture_output=True, check=True)
    return str(out)


async def test_live_single_image(sample_png_bytes, tmp_path) -> None:
    """One image, describe task: 1 request, frames == 1."""
    path = tmp_path / "sample.png"
    path.write_bytes(sample_png_bytes)
    result = await analyze_media(
        str(path), task="describe", options=_opts()
    )
    assert result.usage.requests >= 1
    assert result.usage.input_tokens > 0
    assert result.media.frames == 1


async def test_live_multi_image(sample_png_bytes, tmp_path) -> None:
    """Two images, compare task: one request, two images processed."""
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(sample_png_bytes)
    p2.write_bytes(sample_png_bytes)
    result = await analyze_media(
        [str(p1), str(p2)], task="compare", options=_opts()
    )
    assert isinstance(result.data, CompareResult)
    assert result.usage.requests == 1
    assert result.usage.images_processed == 2


async def test_live_structured_extraction(tmp_path) -> None:
    """Structured extraction with a caller-supplied schema."""
    path = tmp_path / "invoice.png"
    _invoice_image(path)
    result = await analyze_media(
        str(path),
        task="extract",
        output_schema=Invoice,
        options=_opts(),
    )
    assert isinstance(result.data, Invoice)
    assert result.usage.requests >= 1
    print(f"extracted invoice: {result.data.model_dump()}")
    if result.data.invoice_number is None:
        print("warning: invoice_number is None (LLM output is stochastic)")


async def test_live_short_video(live_video) -> None:
    """4s video, hierarchical video_summary: timestamped evidence."""
    result = await analyze_media(
        live_video,
        task="video_summary",
        options=_opts(
            video={"sampling": "uniform", "fps": 1.0, "max_frames": 6},
            max_images_per_request=3,
        ),
    )
    assert result.evidence, "expected evidence entries for the video"
    for entry in result.evidence:
        assert entry.source
        assert entry.timestamp_seconds is not None
    assert result.synthesis or result.data is not None


async def test_live_multibatch_document(tmp_path) -> None:
    """3-page PDF, 2 pages per batch: 3 requests, page-cited evidence."""
    import pymupdf

    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
        page.insert_text((72, 120), f"Fact {i + 1} on page {i + 1}.")
    doc.save(str(path))
    doc.close()

    result = await analyze_media(
        str(path),
        task="document_analysis",
        options=_opts(document={"max_pages_per_batch": 2}),
    )
    assert result.usage.requests >= 3  # 2 batches + 1 global synthesis
    assert result.media.frames == 3
    for entry in result.evidence:
        assert entry.source
        assert entry.page in {1, 2, 3}
    assert isinstance(result.data, DocumentAnalysis)
