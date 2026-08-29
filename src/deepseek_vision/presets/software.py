"""Software-engineering presets: regression, bug triage, verification, UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..models import VerificationResult
from ._base import Preset


class VisualRegressionResult(BaseModel):
    """Structured output for expected-vs-implementation UI comparison."""

    common_elements: list[str]
    differences: list[dict[str, Any]]
    changes: list[str]
    implementation_notes: list[str]
    confidence: dict[str, float]


class BugReport(BaseModel):
    """Structured UI bug report from a screenshot."""

    severity: str
    symptoms: list[str]
    likely_cause: str
    affected_component: str
    suggested_fix: str
    confidence: float


class ScreenAnalysis(BaseModel):
    """Structured analysis of a terminal/error screenshot."""

    summary: str
    error_message: str | None = None
    elements: list[dict[str, Any]]
    next_actions: list[str]


visual_regression = Preset(
    name="visual_regression",
    description="Compare expected vs actual UI as a frontend QA analyst.",
    system_prompt=(
        "You are a frontend QA analyst comparing expected vs actual UI."
    ),
    task_instructions=(
        "Compare the images as reference vs implementation (first image is "
        "reference). Identify implementation differences a frontend coding "
        "agent must fix: layout, spacing, color, typography, missing or extra "
        "elements. Be concrete enough to act on. Respond in valid json."
    ),
    output_schema=VisualRegressionResult,
    temperature=0.0,
)

bug_report = Preset(
    name="bug_report",
    description="Triage UI bugs from screenshots.",
    system_prompt="You triage UI bugs from screenshots.",
    task_instructions=(
        "Analyze the screenshot as a bug report: severity "
        "(critical/major/minor/cosmetic), visible symptoms, likely root cause, "
        "affected component, suggested fix, confidence."
    ),
    output_schema=BugReport,
    temperature=0.2,
)

verification = Preset(
    name="verification",
    description="Verify visual claims against evidence.",
    system_prompt=(
        "You verify visual claims against evidence. Be strict: only pass when "
        "clearly visible."
    ),
    task_instructions=(
        "The user provides claims to verify. For each claim, determine "
        "status: pass (clearly visible), fail (contradicted or clearly "
        "absent), inconclusive (cannot determine). Provide evidence text and "
        "confidence per check. Respond in valid json with "
        '{"checks": [{"claim": str, "status": "pass|fail|inconclusive", '
        '"evidence": str, "confidence": float}]}.'
    ),
    output_schema=VerificationResult,
    temperature=0.0,
)

screenshot_analysis = Preset(
    name="screenshot_analysis",
    description="Inspect terminal output, error dialogs, and screenshots.",
    system_prompt=(
        "You inspect terminal output, error dialogs, and screenshots for "
        "software issues."
    ),
    task_instructions=(
        "Analyze the screenshot: summarize what is shown, extract any error "
        "message verbatim if present, enumerate visible elements, and propose "
        "next actions for a debugging agent."
    ),
    output_schema=ScreenAnalysis,
    temperature=0.2,
)
