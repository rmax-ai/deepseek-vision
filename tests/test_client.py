"""Tests for the DeepSeek API transport client (respx-mocked)."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from pydantic import BaseModel

from deepseek_vision.client import DeepSeekMultimodalClient
from deepseek_vision.errors import (
    ProviderAPIError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseValidationError,
)
from deepseek_vision.models import MediaFrame


class SummarySchema(BaseModel):
    summary: str


def _frames(n: int = 1) -> list[MediaFrame]:
    return [
        MediaFrame(
            image=b"\xff\xd8\xff" + b"0" * 64,
            source=f"img{i}.jpg",
            metadata={"format": "jpeg"},
        )
        for i in range(n)
    ]


async def test_request_shape(mock_api, client) -> None:
    await client.analyze(
        _frames(2),
        prompt="Describe",
        output_schema=SummarySchema,
        user_id="user-42",
        temperature=0.5,
        max_output_tokens=2048,
    )
    request = mock_api.calls[-1].request
    assert str(request.url).endswith("/chat/completions")
    body = json.loads(request.content)
    assert body["model"] == "deepseek-v4-flash-vision-exp"
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.5
    assert body["user_id"] == "user-42"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    messages = body["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Describe"}
    for part in content[1:]:
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert part["image_url"]["detail"] == "original"


async def test_temperature_omitted_when_none(mock_api, client) -> None:
    await client.analyze(_frames(), prompt="P")
    body = json.loads(mock_api.calls[-1].request.content)
    assert "temperature" not in body


async def test_response_format_absent_without_schema(mock_api, client) -> None:
    await client.analyze(_frames(), prompt="P")
    body = json.loads(mock_api.calls[-1].request.content)
    assert "response_format" not in body


async def test_system_message_first(mock_api, client) -> None:
    await client.analyze(_frames(), prompt="P", system="You are strict.")
    body = json.loads(mock_api.calls[-1].request.content)
    assert body["messages"][0] == {"role": "system", "content": "You are strict."}


async def test_usage_recorded_in_tracker(mock_api, client) -> None:
    await client.analyze(_frames(3), prompt="P")
    summary = client.usage_tracker.to_summary()
    assert summary.requests == 1
    assert summary.images_processed == 3
    assert summary.input_tokens == 100
    assert summary.output_tokens == 50


async def test_429_then_success(mock_api_json_sequence) -> None:
    responses = [
        httpx.Response(
            429,
            headers={"Retry-After": "0.1"},
            json={"error": "rate limited"},
        ),
        httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"summary": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ),
    ]
    router = mock_api_json_sequence(responses)
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=2,
        retry_base_delay=0.01,
    )
    start = time.monotonic()
    result = await client.analyze(_frames(), prompt="P")
    elapsed = time.monotonic() - start
    assert len(router.calls) == 2
    assert client.usage_tracker.to_summary().retries == 1
    assert result["data"]["summary"] == "ok"
    assert elapsed >= 0.1  # Retry-After honored


async def test_500_then_success(mock_api_json_sequence) -> None:
    responses = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"summary": "ok"}'}}],
                "usage": {},
            },
        ),
    ]
    router = mock_api_json_sequence(responses)
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=2,
        retry_base_delay=0.001,
    )
    result = await client.analyze(_frames(), prompt="P")
    assert len(router.calls) == 2
    assert result["data"]["summary"] == "ok"


async def test_429_exhausted_raises_rate_limit(mock_api_json_sequence) -> None:
    resp = httpx.Response(
        429, headers={"Retry-After": "0.01"}, json={"error": "slow down"}
    )
    router = mock_api_json_sequence([resp])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=1,
        retry_base_delay=0.001,
    )
    with pytest.raises(RateLimitError) as exc_info:
        await client.analyze(_frames(), prompt="P")
    assert exc_info.value.retry_after == 0.01
    assert len(router.calls) == 2


async def test_timeout_then_success(mock_api_json_sequence) -> None:
    mock_api_json_sequence(
        [
            httpx.ConnectTimeout("connection timed out"),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"summary": "ok"}'}}
                    ],
                    "usage": {},
                },
            ),
        ]
    )
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=2,
        retry_base_delay=0.001,
    )
    result = await client.analyze(_frames(), prompt="P")
    assert result["data"]["summary"] == "ok"


async def test_timeout_exhausted_raises(mock_api_json_sequence) -> None:
    mock_api_json_sequence([httpx.ConnectTimeout("connection timed out")])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=1,
        retry_base_delay=0.001,
    )
    with pytest.raises(ProviderTimeoutError):
        await client.analyze(_frames(), prompt="P")


async def test_400_raises_provider_error(mock_api_json_sequence) -> None:
    resp = httpx.Response(
        400, json={"error": "This model does not support image"}
    )
    router = mock_api_json_sequence([resp])
    client = DeepSeekMultimodalClient(api_key="sk-test-placeholder")
    with pytest.raises(ProviderAPIError) as exc_info:
        await client.analyze(_frames(), prompt="P")
    assert exc_info.value.status_code == 400
    assert len(router.calls) == 1


async def test_empty_content_then_success(mock_api_json_sequence) -> None:
    empty = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}}], "usage": {}},
    )
    ok = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"summary": "ok"}'}}],
            "usage": {},
        },
    )
    router = mock_api_json_sequence([empty, ok])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=2,
        retry_base_delay=0.001,
    )
    result = await client.analyze(_frames(), prompt="P", output_schema=SummarySchema)
    assert len(router.calls) == 2
    assert isinstance(result["data"], SummarySchema)


async def test_invalid_json_with_schema_raises(mock_api_json_sequence) -> None:
    bad = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "not json"}}], "usage": {}},
    )
    mock_api_json_sequence([bad])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=2,
        retry_base_delay=0.001,
    )
    with pytest.raises(ResponseValidationError):
        await client.analyze(_frames(), prompt="P", output_schema=SummarySchema)


async def test_valid_dict_json_without_schema(mock_api, client) -> None:
    result = await client.analyze(_frames(), prompt="P")
    assert result["data"] == {"summary": "ok"}
    assert result["text"] == '{"summary": "ok"}'


async def test_non_json_text_without_schema(mock_api_json_sequence) -> None:
    resp = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "plain text answer"}}],
            "usage": {},
        },
    )
    mock_api_json_sequence([resp])
    client = DeepSeekMultimodalClient(api_key="sk-test-placeholder")
    result = await client.analyze(_frames(), prompt="P")
    assert result["data"] == "plain text answer"


async def test_markdown_fenced_json_stripped(mock_api_json_sequence) -> None:
    content = '```json\n{"summary": "fenced"}\n```'
    resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}], "usage": {}},
    )
    mock_api_json_sequence([resp])
    client = DeepSeekMultimodalClient(api_key="sk-test-placeholder")
    result = await client.analyze(_frames(), prompt="P", output_schema=SummarySchema)
    assert result["data"].summary == "fenced"


async def test_aenter_aexit(mock_api) -> None:
    async with DeepSeekMultimodalClient(api_key="sk-test-placeholder") as client:
        result = await client.analyze(_frames(), prompt="P")
        assert result["data"]["summary"] == "ok"
