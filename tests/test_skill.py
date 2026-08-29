"""End-to-end tests for the analyze_media skill (respx-mocked)."""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from deepseek_vision.errors import ConfigurationError
from deepseek_vision.models import AnalysisResult, CompareResult
from deepseek_vision.presets.documents import DocumentAnalysis
from deepseek_vision.skill import analyze_media, analyze_media_sync


class InvoiceTest(BaseModel):
    vendor: str
    invoice_number: str
    total: float


def _opts(**extra) -> dict:
    opts = {"use_cache": False, "api_key": "sk-test-placeholder"}
    opts.update(extra)
    return opts


async def test_describe_image(mock_api, tmp_path, sample_png_bytes) -> None:
    path = tmp_path / "img.png"
    path.write_bytes(sample_png_bytes)
    result = await analyze_media(str(path), task="describe", options=_opts())
    assert isinstance(result, AnalysisResult)
    assert result.task == "describe"
    assert result.usage.requests == 1
    assert result.media.frames == 1
    assert result.media.batches == 1
    assert result.evidence[0].source == str(path)


async def test_compare_collection(mock_api, tmp_path, sample_png_bytes) -> None:
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
    assert result.media.frames == 2


async def test_document_analysis_hierarchical(
    mock_api_json_sequence, sample_pdf
) -> None:
    responses = [
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"observations": [{"page": 1, "text": "o1", '
                                '"confidence": 0.9}, {"page": 2, "text": "o2", '
                                '"confidence": 0.9}]}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"observations": [{"page": 3, "text": "o3", '
                                '"confidence": 0.8}]}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary": "doc", "sections": [], '
                                '"observations": [], "answer": null}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
    ]
    mock_api_json_sequence(responses)
    result = await analyze_media(
        sample_pdf,
        task="document_analysis",
        options=_opts(document={"max_pages_per_batch": 2}),
    )
    assert result.usage.requests == 3  # 2 batches + 1 global
    assert result.media.frames == 3
    pages = {ev.page for ev in result.evidence}
    assert pages == {1, 2, 3}
    assert isinstance(result.data, DocumentAnalysis)


async def test_video_summary_hierarchical(
    mock_api_json_sequence, sample_video
) -> None:
    responses = [
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"observations": [{"timestamp_seconds": 0.5, '
                                '"text": "w0", "confidence": 0.9}]}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"observations": [{"timestamp_seconds": 3.0, '
                                '"text": "w1", "confidence": 0.9}]}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary": "video summary", "key_moments": '
                                '[], "segments": []}'
                            )
                        }
                    }
                ],
                "usage": {},
            },
        ),
    ]
    mock_api_json_sequence(responses)
    result = await analyze_media(
        sample_video,
        task="video_summary",
        options=_opts(
            video={"sampling": "uniform", "fps": 1.0, "max_frames": 4},
            max_images_per_request=2,
        ),
    )
    assert result.media.frames <= 4
    assert result.media.batches >= 2
    assert result.synthesis
    for ev in result.evidence:
        assert ev.timestamp_seconds is not None
        assert 0.0 <= ev.timestamp_seconds <= 4.0


async def test_extract_with_caller_schema(
    mock_api_json_sequence, tmp_path, sample_jpeg_bytes
) -> None:
    path = tmp_path / "invoice.jpg"
    path.write_bytes(sample_jpeg_bytes)
    content = (
        '{"vendor": "Acme", "invoice_number": "INV-42", "total": 123.45}'
    )
    mock_api_json_sequence(
        [
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": content}}],
                    "usage": {},
                },
            )
        ]
    )
    result = await analyze_media(
        str(path),
        task="extract",
        output_schema=InvoiceTest,
        options=_opts(),
    )
    assert isinstance(result.data, InvoiceTest)
    assert result.data.vendor == "Acme"
    assert result.data.total == 123.45


async def test_unknown_task_raises(tmp_path, sample_png_bytes) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(sample_png_bytes)
    with pytest.raises(ConfigurationError):
        await analyze_media(
            str(path), task="nonexistent_task", options=_opts()
        )


async def test_missing_api_key_raises(
    monkeypatch, tmp_path, sample_png_bytes
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = tmp_path / "x.png"
    path.write_bytes(sample_png_bytes)
    with pytest.raises(ConfigurationError) as exc_info:
        await analyze_media(str(path), task="describe", options={"use_cache": False})
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)


def test_analyze_media_sync_wrapper(
    mock_api, tmp_path, sample_png_bytes
) -> None:
    path = tmp_path / "img.png"
    path.write_bytes(sample_png_bytes)
    result = analyze_media_sync(
        str(path), task="describe", options=_opts()
    )
    assert isinstance(result, AnalysisResult)
    assert result.usage.requests == 1


async def test_cache_enabled_single_batch(mock_api, tmp_path, sample_png_bytes) -> None:
    """Cached runs must not raise (regression: cache.key was not a method)."""
    path = tmp_path / "img.png"
    path.write_bytes(sample_png_bytes)
    cache_dir = tmp_path / "cache"
    opts = {
        "use_cache": True,
        "cache_dir": str(cache_dir),
        "api_key": "sk-test-placeholder",
    }
    first = await analyze_media(str(path), task="describe", options=opts)
    assert first.usage.requests == 1
    second = await analyze_media(str(path), task="describe", options=opts)
    assert second.meta.get("from_cache") is True
