"""Document-analysis presets: hierarchical page analysis and extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ._base import Preset


class DocumentObservation(BaseModel):
    """A single observation anchored to a document page."""

    page: int
    text: str
    confidence: float | None = None
    kind: str | None = None


class DocumentObservations(BaseModel):
    """Batch-stage output for document analysis."""

    observations: list[DocumentObservation]


class DocumentAnalysis(BaseModel):
    """Global-stage output for document analysis."""

    summary: str
    sections: list[dict[str, Any]]
    observations: list[DocumentObservation]
    answer: str | None = None


document_analysis = Preset(
    name="document_analysis",
    description="Hierarchical multi-page document analysis with page citations.",
    system_prompt=(
        "You analyze document pages. Always cite the page number for every "
        "observation."
    ),
    task_instructions=(
        "Analyze this document: summarize its purpose and structure, extract "
        "key claims, tables, and figures with page citations."
    ),
    batch_instructions=(
        "These are pages from one document. Extract every notable observation "
        "(claims, numbers, tables, diagrams, structure) with the page number "
        "of each observation. Respond in valid json with "
        '{"observations": [{"page": int, "text": str, "confidence": '
        "float|null, \"kind\": str|null}]}."
    ),
    output_schema=DocumentObservations,
    final_output_schema=DocumentAnalysis,
    hierarchical=True,
    temperature=0.0,
)

page_extraction = Preset(
    name="page_extraction",
    description="Extract caller-specified structured data from document pages.",
    system_prompt=(
        "You extract structured information from rendered document pages. "
        "Return only valid JSON."
    ),
    task_instructions=(
        "Extract the requested structured information from these rendered "
        "document pages. Cite page numbers for each extracted item when "
        "possible."
    ),
    output_schema=None,
    temperature=0.0,
)
