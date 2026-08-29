"""Tests for task presets."""

from __future__ import annotations

import pytest

from deepseek_vision.errors import ConfigurationError
from deepseek_vision.models import CompareResult, VerificationResult
from deepseek_vision.presets import (
    PRESETS,
    available_presets,
    get_preset,
)

EXPECTED_NAMES = {
    "describe",
    "question_answer",
    "extract",
    "compare",
    "diagram_analysis",
    "ui_analysis",
    "document_analysis",
    "page_extraction",
    "visual_regression",
    "bug_report",
    "verification",
    "screenshot_analysis",
    "video_summary",
    "temporal_analysis",
    "movement_analysis",
}

HIERARCHICAL = {
    "document_analysis",
    "video_summary",
    "temporal_analysis",
    "movement_analysis",
}


def test_preset_names_exact_set() -> None:
    assert set(PRESETS) == EXPECTED_NAMES
    assert len(PRESETS) == len(EXPECTED_NAMES)


def test_all_presets_have_prompts() -> None:
    for name, preset in PRESETS.items():
        assert preset.system_prompt.strip(), name
        assert preset.task_instructions.strip(), name
        assert preset.description.strip(), name


def test_hierarchical_presets_complete() -> None:
    for name in HIERARCHICAL:
        preset = PRESETS[name]
        assert preset.hierarchical, name
        assert preset.batch_instructions, name
        assert preset.output_schema is not None, name
        assert preset.final_output_schema is not None, name


def test_non_hierarchical_presets_not_hierarchical() -> None:
    for name in EXPECTED_NAMES - HIERARCHICAL:
        assert PRESETS[name].hierarchical is False, name


def test_compare_schema_imports_from_models() -> None:
    assert PRESETS["compare"].output_schema is CompareResult


def test_verification_schema_imports_from_models() -> None:
    assert PRESETS["verification"].output_schema is VerificationResult


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        get_preset("does_not_exist")
    assert "does_not_exist" in str(exc_info.value)


def test_available_presets() -> None:
    assert set(available_presets()) == EXPECTED_NAMES


def test_extract_and_page_extraction_require_caller_schema() -> None:
    assert PRESETS["extract"].output_schema is None
    assert PRESETS["page_extraction"].output_schema is None
