"""Tests for schema-driven prompting and structured extraction."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

from deepseek_vision.client import DeepSeekMultimodalClient
from deepseek_vision.errors import ResponseValidationError
from deepseek_vision.models import MediaFrame
from deepseek_vision.prompts import (
    build_extraction_prompt,
    schema_to_prompt,
)


class InvoiceExtraction(BaseModel):
    vendor: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    total: float | None = None
    currency: str | None = None


class RequiredInvoice(BaseModel):
    vendor: str
    invoice_number: str
    total: float


class Child(BaseModel):
    label: str


class Nested(BaseModel):
    name: str
    count: int
    ratio: float
    active: bool
    tags: list[str]
    mode: Literal["a", "b"]
    child: Child
    note: str | None = None


def _frames(n: int = 1) -> list[MediaFrame]:
    return [
        MediaFrame(
            image=b"\xff\xd8\xff" + b"0" * 64,
            source=f"f{i}.jpg",
            metadata={"format": "jpeg"},
        )
        for i in range(n)
    ]


def test_schema_to_prompt_includes_fields_and_json_word() -> None:
    json_schema, _ = schema_to_prompt(InvoiceExtraction)
    assert "vendor" in json_schema
    assert "invoice_number" in json_schema
    prompt = build_extraction_prompt("Extract invoice data.", InvoiceExtraction)
    assert "json" in prompt.lower()
    assert "vendor" in prompt
    assert "example" in prompt.lower()


def test_example_generator_valid_json() -> None:
    _, example = schema_to_prompt(Nested)
    parsed = json.loads(example)
    assert parsed["name"] == "value"
    assert parsed["count"] == 42
    assert parsed["ratio"] == 0.5
    assert parsed["active"] is True
    assert parsed["tags"] == ["value"]
    assert parsed["mode"] == "a"
    assert parsed["child"] == {"label": "value"}
    assert "note" not in parsed  # optional omitted


async def test_extraction_success(mock_api_json_sequence) -> None:
    content = (
        '{"vendor": "Acme", "invoice_number": "INV-1", "date": "2026-08-01", '
        '"total": 123.45, "currency": "USD"}'
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
    client = DeepSeekMultimodalClient(api_key="sk-test-placeholder")
    result = await client.analyze(
        _frames(),
        prompt=build_extraction_prompt(
            "Extract invoice fields.", InvoiceExtraction
        ),
        output_schema=InvoiceExtraction,
    )
    assert isinstance(result["data"], InvoiceExtraction)
    assert result["data"].vendor == "Acme"
    assert result["data"].total == 123.45
    assert result["data"].currency == "USD"


async def test_invalid_first_response_retries_validation(
    mock_api_json_sequence,
) -> None:
    bad = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"invoice_number": "INV-2"}'}}],
            "usage": {},
        },
    )
    good = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"vendor": "Acme", "invoice_number": "INV-2", '
                            '"total": 9.99}'
                        )
                    }
                }
            ],
            "usage": {},
        },
    )
    router = mock_api_json_sequence([bad, good])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=3,
        retry_base_delay=0.001,
    )
    result = await client.analyze(
        _frames(), prompt="Extract invoice data.", output_schema=RequiredInvoice
    )
    assert len(router.calls) == 2
    assert isinstance(result["data"], RequiredInvoice)
    assert result["data"].vendor == "Acme"
    body = json.loads(router.calls[1].request.content)
    assert "PREVIOUS RESPONSE FAILED VALIDATION" in body["messages"][-1]["content"][0]["text"]


async def test_twice_invalid_raises(mock_api_json_sequence) -> None:
    bad = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"invoice_number": "INV-2"}'}}],
            "usage": {},
        },
    )
    router = mock_api_json_sequence([bad])
    client = DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        max_retries=3,
        retry_base_delay=0.001,
    )
    with pytest.raises(ResponseValidationError):
        await client.analyze(
            _frames(), prompt="Extract invoice data.", output_schema=RequiredInvoice
        )
    assert len(router.calls) == 2


async def test_extraction_markdown_fence_parsed(mock_api_json_sequence) -> None:
    content = (
        '```json\n{"vendor": "Globex", "invoice_number": "INV-9", '
        '"total": 42.0}\n```'
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
    client = DeepSeekMultimodalClient(api_key="sk-test-placeholder")
    result = await client.analyze(
        _frames(),
        prompt=build_extraction_prompt(
            "Extract invoice fields.", InvoiceExtraction
        ),
        output_schema=InvoiceExtraction,
    )
    assert result["data"].vendor == "Globex"
    assert result["data"].total == 42.0
