"""Hierarchical (multi-batch + global synthesis) orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .cache import Cache, key, schema_digest
from .events import (
    BATCH_CREATED,
    REQUEST_COMPLETED,
    EventEmitter,
)
from .models import AnalyzeOptions, Evidence, MediaFrame
from .prompts import schema_to_prompt
from .usage import UsageTracker


def _jsonable(value: Any) -> Any:
    """Convert a value into a JSON-serializable form."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _frame_provenance(frames: list[MediaFrame]) -> list[dict]:
    return [
        {
            "source": frame.source,
            "timestamp": frame.timestamp,
            "page": frame.metadata.get("page"),
        }
        for frame in frames
    ]


class HierarchicalSynthesizer:
    """Runs per-batch analysis concurrently, then a global synthesis pass."""

    def __init__(
        self,
        client: Any,
        event_emitter: EventEmitter,
        options: AnalyzeOptions,
        usage_tracker: UsageTracker,
    ) -> None:
        self.client = client
        self.event_emitter = event_emitter
        self.options = options
        self.usage_tracker = usage_tracker

    async def analyze_batches(
        self,
        batches: list[list[MediaFrame]],
        *,
        task: str,
        instructions: str | None,
        system: str | None,
        output_schema: type | None,
        temperature: float | None,
        max_output_tokens: int,
        detail: str,
        thinking: str | None = None,
        cache: Cache | None = None,
        batch_prompt_builder: Callable[[int, list[MediaFrame]], str],
    ) -> list[dict]:
        """Analyze every batch, concurrently but bounded by the client."""

        async def run_batch(
            index: int, frames: list[MediaFrame]
        ) -> dict:
            prompt = batch_prompt_builder(index, frames)
            cache_key: str | None = None
            if cache is not None:
                cache_key = key(
                    {
                        "media_hashes": cache.match_media(frames),
                        "model": self.options.model,
                        "task": task,
                        "instructions": prompt,
                        "schema_digest": schema_digest(output_schema),
                        "detail": detail,
                        "temperature": temperature,
                        "thinking": thinking,
                        "max_output_tokens": max_output_tokens,
                    }
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    await self.event_emitter.emit(
                        BATCH_CREATED,
                        index=index,
                        size=len(frames),
                        from_cache=True,
                    )
                    await self.event_emitter.emit(
                        REQUEST_COMPLETED,
                        latency_ms=0,
                        tokens=0,
                        attempt=0,
                        from_cache=True,
                    )
                    return {
                        "batch_index": index,
                        "frames": _frame_provenance(frames),
                        "result": {
                            "data": cached.get("data"),
                            "text": cached.get("text", ""),
                            "usage": cached.get("usage", {}),
                        },
                        "from_cache": True,
                    }

            result = await self.client.analyze(
                frames,
                prompt=prompt,
                system=system,
                output_schema=output_schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                detail=detail,
                thinking=thinking,
            )
            if cache is not None and cache_key is not None:
                cache.put(
                    cache_key,
                    {
                        "data": _jsonable(result.get("data")),
                        "text": result.get("text", ""),
                        "usage": result.get("usage", {}),
                    },
                )
            return {
                "batch_index": index,
                "frames": _frame_provenance(frames),
                "result": result,
                "from_cache": False,
            }

        tasks = [
            asyncio.create_task(run_batch(index, batch))
            for index, batch in enumerate(batches)
        ]
        results: list[dict] = []
        for done in asyncio.as_completed(tasks):
            results.append(await done)
        results.sort(key=lambda item: item["batch_index"])
        return results

    async def synthesize_global(
        self,
        segment_results: list[dict],
        representative_frames: list[MediaFrame],
        *,
        task: str,
        instructions: str | None,
        system: str | None,
        output_schema: type | None,
        temperature: float | None,
        max_output_tokens: int,
        detail: str,
        thinking: str | None = None,
        max_repr_frames: int = 8,
        cache: Cache | None = None,
    ) -> dict:
        """One global analysis over all segment results."""
        segments = sorted(
            segment_results, key=lambda item: item["batch_index"]
        )
        repr_frames = _select_representative(
            representative_frames, max_repr_frames
        )

        prompt = (
            f"You analyzed {len(segments)} segments of media. Segment results "
            "follow. Synthesize a global analysis. Preserve provenance: every "
            "claim must cite its source and timestamp/page. Respond in valid "
            "json.\n\n"
        )
        parts: list[str] = []
        for segment in segments:
            payload = {
                "batch_index": segment["batch_index"],
                "frames": segment["frames"],
                "result": _jsonable(segment.get("result", {})),
            }
            parts.append(
                f"Segment {segment['batch_index']}:\n"
                + json.dumps(payload, default=str)[:2000]
            )
        prompt += "\n\n".join(parts)

        if output_schema is not None:
            schema_str, example_str = schema_to_prompt(output_schema)
            prompt += (
                "\n\nRespond with a single JSON object that matches the "
                "following schema exactly. No markdown fences, no commentary.\n"
                f"JSON Schema:\n{schema_str}\n\nExample JSON:\n{example_str}"
            )

        cache_key: str | None = None
        if cache is not None:
            cache_key = key(
                {
                    "media_hashes": cache.match_media(repr_frames),
                    "model": self.options.model,
                    "task": f"{task}:synthesize_global",
                    "instructions": prompt,
                    "schema_digest": schema_digest(output_schema),
                    "detail": detail,
                    "temperature": temperature,
                    "thinking": thinking,
                    "max_output_tokens": max_output_tokens,
                }
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return {
                    "data": cached.get("data"),
                    "text": cached.get("text", ""),
                    "usage": cached.get("usage", {}),
                }

        result = await self.client.analyze(
            repr_frames,
            prompt=prompt,
            system=system,
            output_schema=output_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            detail=detail,
            thinking=thinking,
        )
        if cache is not None and cache_key is not None:
            cache.put(
                cache_key,
                {
                    "data": _jsonable(result.get("data")),
                    "text": result.get("text", ""),
                    "usage": result.get("usage", {}),
                },
            )
        return result

    def extract_evidence(
        self, results: list[dict], output_schema: type | None
    ) -> list[Evidence]:
        """Extract provenance-carrying Evidence from segment results."""
        evidence: list[Evidence] = []
        for segment in results:
            frames_meta = segment.get("frames", [])
            first = frames_meta[0] if frames_meta else {}
            result = segment.get("result", {})
            data = result.get("data") if isinstance(result, dict) else None
            if isinstance(data, BaseModel):
                data_dict = data.model_dump()
            else:
                data_dict = data

            observations: list[dict] = []
            if isinstance(data_dict, dict):
                raw = data_dict.get("observations")
                if isinstance(raw, list):
                    observations = [
                        item for item in raw if isinstance(item, dict)
                    ]

            single_page = (
                first.get("page") is not None
                and all(f.get("page") == first.get("page") for f in frames_meta)
            )

            if observations:
                for item in observations:
                    text = item.get("text") or item.get("observation")
                    if not text:
                        continue
                    evidence.append(
                        Evidence(
                            source=first.get("source", ""),
                            page=(
                                item.get("page")
                                if item.get("page") is not None
                                else (first.get("page") if single_page else None)
                            ),
                            timestamp_seconds=(
                                item.get("timestamp_seconds")
                                if item.get("timestamp_seconds") is not None
                                else (
                                    item.get("timestamp")
                                    if item.get("timestamp") is not None
                                    else first.get("timestamp")
                                )
                            ),
                            batch=segment.get("batch_index"),
                            observation=str(text),
                            confidence=item.get("confidence"),
                        )
                    )
            else:
                wrapped = (
                    json.dumps(data_dict, default=str)[:2000]
                    if data_dict is not None
                    else ""
                )
                evidence.append(
                    Evidence(
                        source=first.get("source", ""),
                        page=first.get("page") if single_page else None,
                        timestamp_seconds=first.get("timestamp"),
                        batch=segment.get("batch_index"),
                        observation=wrapped,
                        confidence=None,
                    )
                )
        return evidence


def _select_representative(
    frames: list[MediaFrame], max_frames: int
) -> list[MediaFrame]:
    """Evenly subsample frames, always keeping both ends."""
    if not frames or max_frames <= 0:
        return []
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return frames[:1]
    indices = {
        round(i * (len(frames) - 1) / (max_frames - 1))
        for i in range(max_frames)
    }
    return [frames[i] for i in sorted(indices)]
