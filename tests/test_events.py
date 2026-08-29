"""Tests for pipeline event emission."""

from __future__ import annotations

import httpx

from deepseek_vision.skill import analyze_media


def _options(collect_events) -> dict:
    return {
        "use_cache": False,
        "api_key": "sk-test-placeholder",
        "event_handler": collect_events,
    }


async def test_single_image_event_order(
    mock_api, tmp_path, sample_png_bytes, collect_events
) -> None:
    path = tmp_path / "img.png"
    path.write_bytes(sample_png_bytes)
    result = await analyze_media(
        str(path), task="describe", options=_options(collect_events)
    )
    assert result.task == "describe"
    names = [name for name, _ in collect_events.recorded]
    assert names[0] == "media_loaded"
    assert names[1] == "media_preprocessed"
    assert names[2] == "batch_created"
    assert names.index("request_started") > names.index("batch_created")
    assert names.index("request_completed") > names.index("request_started")


async def test_document_hierarchical_events(
    mock_api_json_sequence, tmp_path, sample_pdf, collect_events
) -> None:
    responses = []
    for i in range(3):
        content = (
            f'{{"observations": [{{"page": {i + 1}, "text": "obs {i + 1}", '
            f'"confidence": 0.9}}]}}'
        )
        responses.append(
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": content}}],
                    "usage": {},
                },
            )
        )
    responses.append(
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
        )
    )
    mock_api_json_sequence(responses)
    options = _options(collect_events)
    options["document"] = {"max_pages_per_batch": 2}
    result = await analyze_media(
        sample_pdf, task="document_analysis", options=options
    )
    assert result.usage.requests == 3  # 2 batches + 1 global
    names = [name for name, _ in collect_events.recorded]
    assert "synthesis_started" in names
    assert "synthesis_completed" in names
    last_batch = max(i for i, n in enumerate(names) if n == "batch_created")
    assert names.index("synthesis_started") > last_batch
    assert names.index("synthesis_completed") > names.index("synthesis_started")
