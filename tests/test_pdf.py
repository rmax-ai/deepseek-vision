"""Tests for PDF page parsing and rendering."""

from __future__ import annotations

import pytest

from deepseek_vision.errors import MediaError
from deepseek_vision.media.pdf import parse_page_spec, render_pages
from deepseek_vision.models import DocumentOptions


def test_parse_page_spec_none_means_all() -> None:
    assert parse_page_spec(None, 5) == [1, 2, 3, 4, 5]


def test_parse_page_spec_ranges() -> None:
    assert parse_page_spec("1-10,15,20-25", 30) == (
        list(range(1, 11)) + [15] + list(range(20, 26))
    )


def test_parse_page_spec_descending_raises() -> None:
    with pytest.raises(MediaError):
        parse_page_spec("3-1", 30)


def test_parse_page_spec_clamps() -> None:
    assert parse_page_spec("99", 3) == [3]
    assert parse_page_spec("0", 3) == [1]


def test_parse_page_spec_malformed() -> None:
    with pytest.raises(MediaError):
        parse_page_spec("abc", 30)
    with pytest.raises(MediaError):
        parse_page_spec("1,2,,3", 30)
    with pytest.raises(MediaError):
        parse_page_spec("1-2-3", 30)


def test_parse_page_spec_dedupes_and_sorts() -> None:
    assert parse_page_spec("5,1,5,3", 10) == [1, 3, 5]


async def test_render_pages(sample_pdf: str) -> None:
    options = DocumentOptions(dpi=100)
    frames = await render_pages(sample_pdf, options)
    assert len(frames) == 3
    assert [f.metadata["page"] for f in frames] == [1, 2, 3]
    assert all(f.metadata["format"] == "jpeg" for f in frames)
    assert all(f.metadata["total_pages"] == 3 for f in frames)
    assert all(f.image.startswith(b"\xff\xd8") for f in frames)


async def test_render_pages_subset(sample_pdf: str) -> None:
    options = DocumentOptions(pages="2-3", dpi=100)
    frames = await render_pages(sample_pdf, options)
    assert [f.metadata["page"] for f in frames] == [2, 3]


async def test_render_pages_missing_file() -> None:
    with pytest.raises(MediaError):
        await render_pages("/nonexistent/does-not-exist.pdf", DocumentOptions())
