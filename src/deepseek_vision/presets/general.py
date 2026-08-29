"""General-purpose presets: describe, Q&A, extract, compare, diagrams, UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..models import CompareResult
from ._base import Preset


class DiagramAnalysis(BaseModel):
    """Structured output for architecture/technical diagram analysis."""

    components: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    data_flow: str
    potential_risks: list[str]
    confidence: float


class UiAnalysis(BaseModel):
    """Structured output for UI state inspection."""

    summary: str
    elements: list[dict[str, Any]]
    anomalies: list[str]
    issues: list[str]
    recommendations: list[str]


describe = Preset(
    name="describe",
    description="Precise, concrete description of what is visible.",
    system_prompt=(
        "You are a precise visual analyst. Describe what is concretely "
        "visible; do not speculate about intent."
    ),
    task_instructions=(
        "Describe the image(s) precisely and concretely: what is shown, key "
        "objects, layout, notable details."
    ),
    temperature=0.3,
)

question_answer = Preset(
    name="question_answer",
    description="Answer user questions grounded in visual evidence.",
    system_prompt=(
        "You answer questions using visual evidence from the provided images."
    ),
    task_instructions=(
        "Answer the user's question about the provided media, grounding every "
        "claim in what is visible. Cite image numbers."
    ),
    temperature=0.3,
)

extract = Preset(
    name="extract",
    description="Extract structured information from visual media.",
    system_prompt="You extract structured information from visual media. Return only valid JSON.",
    task_instructions=(
        "Extract the requested structured information from the media. Every "
        "field must come from evidence visible in the media; use null when a "
        "field is not determinable."
    ),
    temperature=0.0,
    max_output_tokens=16384,
    output_schema=None,
)

compare = Preset(
    name="compare",
    description="Compare images and produce structured differences.",
    system_prompt="You compare images and produce structured differences.",
    task_instructions=(
        "Compare the provided images. Report common elements, precise "
        "differences (with image references), net changes between images when "
        "order implies before/after, and confidence per finding. Respond in "
        "valid json."
    ),
    output_schema=CompareResult,
    temperature=0.0,
)

diagram_analysis = Preset(
    name="diagram_analysis",
    description="Analyze architecture and technical diagrams.",
    system_prompt="You analyze architecture and technical diagrams.",
    task_instructions=(
        "Analyze the diagram: identify components (name, role, key "
        "inputs/outputs), relationships/dependencies between them, the likely "
        "data flow, and potential risks or bottlenecks. Ground every claim in "
        "what is visible."
    ),
    output_schema=DiagramAnalysis,
    temperature=0.2,
)

ui_analysis = Preset(
    name="ui_analysis",
    description="Inspect UI states and screenshots like a QA engineer.",
    system_prompt="You inspect UI states and screenshots like a QA engineer.",
    task_instructions=(
        "Inspect the UI state: enumerate visible elements, note visual "
        "anomalies, usability or accessibility problems, and concrete "
        "recommendations."
    ),
    output_schema=UiAnalysis,
    temperature=0.2,
)
