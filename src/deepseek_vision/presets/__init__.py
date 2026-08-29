"""Task presets: prompt/config data with no business logic."""

from __future__ import annotations

from ..errors import ConfigurationError
from ._base import Preset
from .documents import document_analysis, page_extraction
from .general import (
    compare,
    describe,
    diagram_analysis,
    extract,
    question_answer,
    ui_analysis,
)
from .software import (
    bug_report,
    screenshot_analysis,
    verification,
    visual_regression,
)
from .video import movement_analysis, temporal_analysis, video_summary

__all__ = [
    "PRESETS",
    "Preset",
    "available_presets",
    "get_preset",
]


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        describe,
        question_answer,
        extract,
        compare,
        diagram_analysis,
        ui_analysis,
        document_analysis,
        page_extraction,
        visual_regression,
        bug_report,
        verification,
        screenshot_analysis,
        video_summary,
        temporal_analysis,
        movement_analysis,
    )
}


def get_preset(name: str) -> Preset:
    """Return a preset by name; raise :class:`ConfigurationError` on miss."""
    preset = PRESETS.get(name)
    if preset is None:
        available = ", ".join(sorted(PRESETS))
        raise ConfigurationError(
            f"unknown task preset: {name!r}. Available presets: {available}"
        )
    return preset


def available_presets() -> list[str]:
    """List all registered preset names."""
    return sorted(PRESETS)
