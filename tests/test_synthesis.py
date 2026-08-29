"""Tests for hierarchical synthesis orchestration."""

from __future__ import annotations

import asyncio
import json
import random

from pydantic import BaseModel

from deepseek_vision.events import EventEmitter
from deepseek_vision.models import AnalyzeOptions, MediaFrame
from deepseek_vision.synthesis import HierarchicalSynthesizer
from deepseek_vision.usage import UsageTracker


class VideoSummaryTest(BaseModel):
    summary: str


class MockClient:
    """Canned responses keyed by prompt markers."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def analyze(self, frames, *, prompt, system=None, output_schema=None,
                      temperature=None, max_output_tokens=8192, detail="original",
                      user_id=None, thinking=None):
        self.calls.append(prompt)
        await asyncio.sleep(random.uniform(0.001, 0.02))
        if "Synthesize a global analysis" in prompt:
            data = {"summary": "whole video"}
        elif "batch 0 of 3" in prompt:
            data = {
                "observations": [
                    {"timestamp_seconds": 0.5, "text": "window zero", "confidence": 0.9}
                ]
            }
        elif "batch 1 of 3" in prompt:
            data = {
                "observations": [
                    {"timestamp_seconds": 1.5, "text": "window one", "confidence": 0.8}
                ]
            }
        else:
            data = {
                "observations": [
                    {"timestamp_seconds": 2.5, "text": "window two", "confidence": 0.7}
                ]
            }
        return {
            "data": data,
            "text": json.dumps(data),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _synth(client) -> HierarchicalSynthesizer:
    return HierarchicalSynthesizer(
        client=client,
        event_emitter=EventEmitter(),
        options=AnalyzeOptions(model="test-model"),
        usage_tracker=UsageTracker(),
    )


def _frame(ts: float) -> MediaFrame:
    return MediaFrame(
        image=b"\xff\xd8\xff" + b"0" * 32,
        timestamp=ts,
        source="v.mp4",
        metadata={"format": "jpeg"},
    )


async def test_analyze_batches_sorted_by_index() -> None:
    client = MockClient()
    synth = _synth(client)
    batches = [[_frame(float(i))] for i in range(3)]

    def builder(index: int, frames):
        return f"Analyze window {index}: This is batch {index} of 3."

    results = await synth.analyze_batches(
        batches,
        task="video_summary",
        instructions=None,
        system=None,
        output_schema=None,
        temperature=0.2,
        max_output_tokens=1024,
        detail="original",
        batch_prompt_builder=builder,
    )
    assert [r["batch_index"] for r in results] == [0, 1, 2]
    assert results[0]["frames"][0]["source"] == "v.mp4"
    assert results[0]["frames"][0]["timestamp"] == 0.0
    assert results[0]["from_cache"] is False
    assert results[2]["result"]["data"]["observations"][0]["text"] == "window two"


def test_extract_evidence_with_page() -> None:
    synth = _synth(MockClient())
    results = [
        {
            "batch_index": 0,
            "frames": [{"source": "doc.pdf", "timestamp": None, "page": 1}],
            "result": {
                "data": {
                    "observations": [
                        {"page": 1, "text": "claim one", "confidence": 0.8}
                    ]
                }
            },
            "from_cache": False,
        }
    ]
    evidence = synth.extract_evidence(results, None)
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source == "doc.pdf"
    assert ev.page == 1
    assert ev.batch == 0
    assert ev.observation == "claim one"
    assert ev.confidence == 0.8


def test_extract_evidence_with_timestamp() -> None:
    synth = _synth(MockClient())
    results = [
        {
            "batch_index": 2,
            "frames": [{"source": "v.mp4", "timestamp": 0.0, "page": None}],
            "result": {
                "data": {
                    "observations": [
                        {"timestamp_seconds": 1.5, "text": "event", "confidence": None}
                    ]
                }
            },
            "from_cache": False,
        }
    ]
    evidence = synth.extract_evidence(results, None)
    ev = evidence[0]
    assert ev.source == "v.mp4"
    assert ev.timestamp_seconds == 1.5
    assert ev.batch == 2
    assert ev.observation == "event"


def test_extract_evidence_fallback_wraps_segment() -> None:
    synth = _synth(MockClient())
    results = [
        {
            "batch_index": 1,
            "frames": [{"source": "a.png", "timestamp": None, "page": None}],
            "result": {"data": {"summary": "just a summary"}},
            "from_cache": False,
        }
    ]
    evidence = synth.extract_evidence(results, None)
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source == "a.png"
    assert ev.batch == 1
    assert "just a summary" in ev.observation


async def test_synthesize_global_prompt_and_data() -> None:
    client = MockClient()
    synth = _synth(client)
    segments = [
        {
            "batch_index": 0,
            "frames": [{"source": "v.mp4", "timestamp": 0.5, "page": None}],
            "result": {
                "data": {"observations": [{"text": "segment zero content"}]}
            },
            "from_cache": False,
        },
        {
            "batch_index": 1,
            "frames": [{"source": "v.mp4", "timestamp": 2.0, "page": None}],
            "result": {
                "data": {"observations": [{"text": "segment one content"}]}
            },
            "from_cache": False,
        },
    ]
    repr_frames = [_frame(0.0), _frame(2.0)]
    result = await synth.synthesize_global(
        segments,
        repr_frames,
        task="video_summary",
        instructions=None,
        system=None,
        output_schema=VideoSummaryTest,
        temperature=0.2,
        max_output_tokens=1024,
        detail="original",
    )
    prompt = client.calls[-1]
    assert "provenance" in prompt
    assert "cite" in prompt
    assert "segment zero content" in prompt
    assert "segment one content" in prompt
    parsed = VideoSummaryTest.model_validate(result["data"])
    assert parsed.summary == "whole video"


async def test_synthesize_global_representative_cap() -> None:
    client = MockClient()
    synth = _synth(client)
    segments = [
        {
            "batch_index": i,
            "frames": [{"source": "v.mp4", "timestamp": float(i), "page": None}],
            "result": {"data": {"summary": f"s{i}"}},
            "from_cache": False,
        }
        for i in range(10)
    ]
    repr_frames = [_frame(float(i)) for i in range(10)]
    result = await synth.synthesize_global(
        segments,
        repr_frames,
        task="video_summary",
        instructions=None,
        system=None,
        output_schema=None,
        temperature=0.2,
        max_output_tokens=1024,
        detail="original",
        max_repr_frames=3,
    )
    assert result["data"]["summary"] == "whole video"
