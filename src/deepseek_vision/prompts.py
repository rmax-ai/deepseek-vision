"""Prompt construction helpers: frame labels, token estimates, schemas."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .models import MediaFrame

_JSON_DIRECTIVE = (
    "Respond with a single JSON object that matches the schema exactly. "
    "No markdown fences, no commentary."
)


def format_frames(frames: list[MediaFrame], label: str = "Frame") -> str:
    """Numbered frame labels with provenance.

    Example::

        Frame 1 [source=session.mp4, t=12.30s]
        Frame 2 [source=report.pdf, page=3]
    """
    lines: list[str] = []
    for i, frame in enumerate(frames, start=1):
        provenance: list[str] = [f"source={frame.source}"]
        if frame.timestamp is not None:
            provenance.append(f"t={frame.timestamp:.2f}s")
        if "page" in frame.metadata:
            provenance.append(f"page={frame.metadata['page']}")
        lines.append(f"{label} {i} [{', '.join(provenance)}]")
    return "\n".join(lines)


def estimate_prompt_tokens(text: str) -> int:
    """Conservative prompt-token estimate (~3 chars per token)."""
    return len(text) // 3


def _example_for_schema(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Generate a representative example value for a JSON-schema fragment."""
    ref = schema.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        return _example_for_schema(defs.get(name, {}), defs)

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if variants:
            for variant in variants:
                if variant.get("type") == "null":
                    continue
                return _example_for_schema(variant, defs)
            return None

    all_of = schema.get("allOf")
    if all_of:
        return _example_for_schema(all_of[0], defs)

    schema_type = schema.get("type")
    if schema_type == "object":
        example: dict[str, Any] = {}
        properties = schema.get("properties", {})
        for prop_name in schema.get("required", []):
            example[prop_name] = _example_for_schema(
                properties.get(prop_name, {}), defs
            )
        return example
    if schema_type == "array":
        items = schema.get("items", {})
        return [_example_for_schema(items, defs)]
    if schema_type in ("string", "integer", "number", "boolean"):
        enum = schema.get("enum") or schema.get("const")
        if isinstance(enum, list) and enum:
            return enum[0]
        if enum is not None:
            return enum
        if schema_type == "string":
            return "value"
        if schema_type == "integer":
            return 42
        if schema_type == "number":
            return 0.5
        return True
    return "value"


def schema_to_prompt(schema: type[BaseModel]) -> tuple[str, str]:
    """Return (json_schema_str, example_str) for a Pydantic model."""
    json_schema = schema.model_json_schema(mode="validation")
    defs = json_schema.get("$defs", {})
    json_schema_str = json.dumps(json_schema, indent=2)
    example = _example_for_schema(json_schema, defs)
    example_str = json.dumps(example)
    return json_schema_str, example_str


def build_extraction_prompt(
    instructions: str, schema: type[BaseModel]
) -> str:
    """Prompt for structured extraction against a caller-supplied schema."""
    json_schema_str, example_str = schema_to_prompt(schema)
    return (
        "Extract structured information as json.\n\n"
        f"Instructions:\n{instructions}\n\n"
        f"JSON Schema:\n{json_schema_str}\n\n"
        f"Example JSON:\n{example_str}\n\n"
        + _JSON_DIRECTIVE
    )


def build_analysis_prompt(
    task_instructions: str,
    user_instructions: str | None,
    frame_labels: str,
) -> str:
    """Compose a plain analysis prompt from task/user instructions."""
    parts: list[str] = [task_instructions]
    if user_instructions:
        parts.append(user_instructions)
    if frame_labels:
        parts.append(f"Media:\n{frame_labels}")
    parts.append("Respond in valid json.")
    return "\n\n".join(parts)
