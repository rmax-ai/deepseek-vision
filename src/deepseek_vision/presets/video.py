"""Video-analysis presets: hierarchical, timestamp-cited analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ._base import Preset

_BATCH_INSTRUCTIONS = (
    "These frames are from one temporal window of a video. Describe what "
    "happens in this window, citing timestamps (format t=SS.s). Respond in "
    'valid json with {"observations": [{"timestamp_seconds": float, "text": '
    'str, "confidence": float|null}]}.'
)


class TimedObservation(BaseModel):
    """An observation anchored to a video timestamp."""

    timestamp_seconds: float
    text: str
    confidence: float | None = None


class WindowObservations(BaseModel):
    """Batch-stage output for video window analysis."""

    observations: list[TimedObservation]


class VideoSummary(BaseModel):
    """Global-stage output summarizing a whole video."""

    summary: str
    key_moments: list[TimedObservation]
    segments: list[dict[str, Any]]


class TimelineAnalysis(BaseModel):
    """Global-stage output for temporal-structure extraction."""

    phases: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    summary: str


class MovementAnalysis(BaseModel):
    """Global-stage output for biomechanical movement analysis."""

    phases: list[dict[str, Any]]
    patterns: list[str]
    issues: list[str]
    coaching_feedback: str
    cited_timestamps: list[float]


video_summary = Preset(
    name="video_summary",
    description="Hierarchical video summarization with timestamps.",
    system_prompt=(
        "You analyze video frames with timestamps. Cite t=SS.s for every "
        "observation."
    ),
    task_instructions=(
        "Summarize this video: what happens, in what order, with timestamps "
        "for key moments."
    ),
    batch_instructions=_BATCH_INSTRUCTIONS,
    output_schema=WindowObservations,
    final_output_schema=VideoSummary,
    hierarchical=True,
    temperature=0.2,
)

temporal_analysis = Preset(
    name="temporal_analysis",
    description="Extract temporal structure (phases, sequence, transitions).",
    system_prompt=(
        "You analyze video frames with timestamps. Cite t=SS.s for every "
        "observation."
    ),
    task_instructions=(
        "Extract the temporal structure: phases, sequence of events, "
        "transitions, timeline."
    ),
    batch_instructions=_BATCH_INSTRUCTIONS,
    output_schema=WindowObservations,
    final_output_schema=TimelineAnalysis,
    hierarchical=True,
    temperature=0.2,
)

movement_analysis = Preset(
    name="movement_analysis",
    description="Biomechanical movement analysis from timestamped frames.",
    system_prompt=(
        "You are a movement analyst. Analyze biomechanics from timestamped "
        "frames; cite t=SS.s for every observation."
    ),
    task_instructions=(
        "Analyze movement: weight transfers, momentum continuity, balance and "
        "off-axis movement, movement quality, repeated movement patterns. "
        "Cite timestamps for every important observation."
    ),
    batch_instructions=_BATCH_INSTRUCTIONS,
    output_schema=WindowObservations,
    final_output_schema=MovementAnalysis,
    hierarchical=True,
    temperature=0.2,
)
