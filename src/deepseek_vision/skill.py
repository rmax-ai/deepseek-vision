"""The single public multimodal analysis primitive."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .cache import Cache, schema_digest
from .client import DeepSeekMultimodalClient
from .errors import ConfigurationError
from .events import (
    BATCH_CREATED,
    MEDIA_LOADED,
    MEDIA_PREPROCESSED,
    SYNTHESIS_COMPLETED,
    SYNTHESIS_STARTED,
    EventEmitter,
)
from .media.batching import (
    compute_batch_size,
    estimate_per_image_bytes,
    make_batches,
)
from .media.images import load_collection, load_image
from .media.pdf import render_pages
from .media.video import extract_frames, probe, window_frames
from .models import (
    AnalysisResult,
    AnalyzeOptions,
    DocumentInput,
    Evidence,
    ImageCollectionInput,
    ImageInput,
    MediaFrame,
    MediaInput,
    MediaStats,
    Observation,
    UsageSummary,
    VideoInput,
)
from .presets import get_preset
from .prompts import (
    build_analysis_prompt,
    estimate_prompt_tokens,
    format_frames,
)
from .synthesis import HierarchicalSynthesizer, _jsonable
from .usage import UsageTracker

__version__ = "0.1.0"

_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
_IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"

MediaLike = str | Path | MediaInput | list[str | Path | MediaInput]


def _make_emitter(handler: object | None) -> EventEmitter:
    """Build an EventEmitter from an options event_handler."""
    if isinstance(handler, EventEmitter):
        return handler
    if handler is not None and callable(handler):
        return EventEmitter(handler=handler)  # type: ignore[arg-type]
    return EventEmitter()


def _normalize_media(item: str | Path | MediaInput) -> MediaInput:
    """Resolve a str/Path/media object into a validated MediaInput."""
    if isinstance(item, BaseModel):
        return item
    path = str(item)
    lowered = path.lower()
    if lowered.endswith(".pdf"):
        return DocumentInput(path=path)
    if lowered.endswith(_VIDEO_EXTENSIONS):
        return VideoInput(path=path)
    if lowered.startswith(("http://", "https://")):
        return ImageInput(url=path)
    return ImageInput(path=path)


def _serialize(data: Any) -> str:
    """Serialize parsed data into a stable string for observations."""
    if isinstance(data, BaseModel):
        return json.dumps(data.model_dump(mode="json"), default=str)
    if isinstance(data, (dict, list)):
        return json.dumps(data, default=str)
    if data is None:
        return ""
    return str(data)


def _source_of(input_model: MediaInput) -> str:
    if isinstance(input_model, ImageInput):
        return input_model.path or input_model.url or input_model.name or ""
    if isinstance(input_model, ImageCollectionInput):
        return input_model.name or "collection"
    if isinstance(input_model, (VideoInput, DocumentInput)):
        return input_model.path
    return ""


def _media_kinds_and_sources(
    inputs: list[MediaInput],
) -> tuple[list[str], list[str]]:
    kinds: list[str] = []
    sources: list[str] = []
    for item in inputs:
        if isinstance(item, ImageInput):
            kinds.append("image")
            sources.append(_source_of(item))
        elif isinstance(item, ImageCollectionInput):
            kinds.append("collection")
            sources.extend(
                image.path or image.url or image.name or ""
                for image in item.images
            )
        elif isinstance(item, VideoInput):
            kinds.append("video")
            sources.append(item.path)
        elif isinstance(item, DocumentInput):
            kinds.append("document")
            sources.append(item.path)
    return kinds, sources


def _extract_evidence_from_frames(
    frames: list[MediaFrame], observation: str
) -> list[Evidence]:
    return [
        Evidence(
            source=frame.source,
            page=frame.metadata.get("page"),
            timestamp_seconds=frame.timestamp,
            observation=observation,
        )
        for frame in frames
    ]


async def analyze_media(
    media: MediaLike,
    *,
    task: str = "describe",
    instructions: str | None = None,
    output_schema: type[BaseModel] | None = None,
    options: AnalyzeOptions | dict | None = None,
) -> AnalysisResult:
    """Analyze images, collections, videos, or documents in one call."""
    # 1. Resolve options and credentials.
    if options is None:
        opts = AnalyzeOptions()
    elif isinstance(options, AnalyzeOptions):
        opts = options
    else:
        opts = AnalyzeOptions.model_validate(options)

    api_key = opts.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigurationError("DEEPSEEK_API_KEY not set")
    base_url = (
        opts.base_url
        or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    )
    model = (
        opts.model
        or os.environ.get("DEEPSEEK_MULTIMODAL_MODEL", DEFAULT_MODEL)
    )
    opts = opts.model_copy(update={"model": model, "base_url": base_url})

    emitter = _make_emitter(opts.event_handler)
    tracker = UsageTracker()
    cache = Cache(opts.cache_dir) if opts.use_cache else None

    # 2. Normalize media inputs.
    if isinstance(media, (str, Path)):
        normalized: list[MediaInput] = [_normalize_media(media)]
    elif isinstance(media, BaseModel):
        normalized = [media]
    elif isinstance(media, list):
        resolved = [_normalize_media(item) for item in media]
        if all(isinstance(item, ImageInput) for item in resolved):
            normalized = [
                ImageCollectionInput(
                    images=resolved  # type: ignore[arg-type]
                )
            ]
        else:
            raise ConfigurationError(
                "list media must contain only images; pass explicit "
                "VideoInput/DocumentInput objects for mixed media"
            )
    else:
        raise ConfigurationError(f"unsupported media input: {type(media)}")

    kinds, sources = _media_kinds_and_sources(normalized)
    await emitter.emit(MEDIA_LOADED, kinds=kinds, sources=sources)

    # 3. Resolve the preset and schemas.
    preset = get_preset(task)
    schema = output_schema or preset.output_schema
    if preset.name in ("extract", "page_extraction") and schema is None:
        raise ConfigurationError(
            f"preset {task!r} requires an output_schema"
        )
    hierarchical = preset.hierarchical or False
    batch_schema = preset.output_schema if hierarchical else schema
    final_schema = (
        preset.final_output_schema
        or (output_schema or preset.output_schema)
    ) if hierarchical else None

    # 4. Preprocess per kind.
    frames: list[MediaFrame] = []
    duration_seconds: float | None = None
    pages = 0
    for item in normalized:
        if isinstance(item, ImageInput):
            frames.append(await load_image(item))
        elif isinstance(item, ImageCollectionInput):
            frames.extend(await load_collection(item))
        elif isinstance(item, VideoInput):
            info = await probe(item.path)
            duration_seconds = info.get("duration")
            frames.extend(
                await extract_frames(item.path, opts.video)
            )
        elif isinstance(item, DocumentInput):
            rendered = await render_pages(item.path, opts.document)
            pages = len(rendered)
            frames.extend(rendered)

    frames.sort(
        key=lambda f: (
            f.timestamp if f.timestamp is not None else float("inf")
        )
    )
    all_sources = sorted({frame.source for frame in frames})
    await emitter.emit(
        MEDIA_PREPROCESSED,
        frames=len(frames),
        pages=pages,
        sources=all_sources,
    )

    if not frames:
        raise ConfigurationError("no frames could be extracted from media")

    # 5. Batching.
    prompt_tokens = (
        estimate_prompt_tokens(instructions or "")
        + estimate_prompt_tokens(preset.task_instructions)
        + 1024
    )
    batch_size = compute_batch_size(
        len(frames),
        context_window=opts.context_window,
        image_token_estimate=opts.image_token_estimate,
        reserved_output_tokens=opts.reserved_output_tokens,
        prompt_tokens=prompt_tokens,
        safety_factor=opts.context_safety_factor,
        max_images_per_request=opts.max_images_per_request,
        per_image_bytes=estimate_per_image_bytes(frames),
    )
    if isinstance(normalized[0], DocumentInput) and opts.document.max_pages_per_batch:
        batch_size = min(batch_size, opts.document.max_pages_per_batch)

    batches: list[list[MediaFrame]] = []
    if isinstance(normalized[0], VideoInput) and opts.video.window_seconds:
        windows = window_frames(frames, opts.video.window_seconds)
        for window in windows:
            batches.extend(make_batches(window, batch_size))
    else:
        batches.extend(make_batches(frames, batch_size))

    for index, batch in enumerate(batches):
        await emitter.emit(
            BATCH_CREATED, index=index, size=len(batch)
        )

    stats = MediaStats(
        sources=all_sources,
        frames=len(frames),
        pages=pages,
        batches=len(batches),
        batch_sizes=[len(batch) for batch in batches],
        duration_seconds=duration_seconds,
    )

    temperature = (
        opts.temperature if opts.temperature is not None else preset.temperature
    )
    detail = opts.detail
    max_output_tokens = _effective_max_output_tokens(options, opts, preset)

    # 6. Non-hierarchical single-batch path.
    if not hierarchical and len(batches) == 1:
        return await _run_single_batch(
            frames=frames,
            preset_task_instructions=preset.task_instructions,
            preset_system_prompt=preset.system_prompt,
            instructions=instructions,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            detail=detail,
            user_id=opts.user_id,
            task=task,
            stats=stats,
            tracker=tracker,
            emitter=emitter,
            cache=cache,
            api_key=api_key,
            base_url=base_url,
            model=model,
            opts=opts,
        )

    # 7. Hierarchical (or multi-batch) path.
    full_pipeline_key: str | None = None
    if cache is not None:
        media_options_digest = json.dumps(
            {
                "video": opts.video.model_dump(),
                "document": opts.document.model_dump(),
                "max_images_per_request": opts.max_images_per_request,
            },
            sort_keys=True,
            default=str,
        )
        full_pipeline_key = cache.key(
            {
                "media_hashes": cache.match_media(frames),
                "model": model,
                "task": task,
                "instructions": f"{instructions or ''}|{media_options_digest}",
                "schema_digest": f"{schema_digest(batch_schema)}|{schema_digest(final_schema)}",
                "detail": detail,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        cached = cache.get(full_pipeline_key)
        if cached is not None:
            return _result_from_cache(cached, task=task, stats=stats)

    await emitter.emit(SYNTHESIS_STARTED, batches=len(batches))
    errors: list[str] = []
    client = DeepSeekMultimodalClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries=opts.max_retries,
        timeout_seconds=opts.timeout_seconds,
        retry_base_delay=opts.retry_base_delay,
        max_concurrency=opts.max_concurrency,
        event_emitter=emitter,
        usage_tracker=tracker,
    )
    synthesizer = HierarchicalSynthesizer(
        client=client,
        event_emitter=emitter,
        options=opts,
        usage_tracker=tracker,
    )
    total_batches = len(batches)

    def batch_prompt_builder(
        index: int, batch_frames: list[MediaFrame]
    ) -> str:
        template = preset.batch_instructions or preset.task_instructions
        labels = format_frames(batch_frames)
        return (
            f"{template}\n\nMedia:\n{labels}\n\n"
            f"This is batch {index} of {total_batches}."
        )

    try:
        segment_results = await synthesizer.analyze_batches(
            batches,
            task=task,
            instructions=instructions,
            system=preset.system_prompt,
            output_schema=batch_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            detail=detail,
            cache=cache,
            batch_prompt_builder=batch_prompt_builder,
        )

        errors = [
            segment["error"]
            for segment in segment_results
            if segment.get("error")
        ]
        successful = [
            segment
            for segment in segment_results
            if not segment.get("error")
        ]
        if not successful:
            raise ConfigurationError(
                f"all {len(segment_results)} batch(es) failed: {errors}"
            )

        representative_frames = [batch[0] for batch in batches]
        global_result = await synthesizer.synthesize_global(
            successful,
            representative_frames,
            task=task,
            instructions=instructions,
            system=preset.system_prompt,
            output_schema=final_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            detail=detail,
            cache=cache,
        )
    finally:
        await client.aclose()

    evidence = synthesizer.extract_evidence(successful, batch_schema)
    observations = [
        Observation(
            text=_serialize(segment.get("result", {}).get("data")),
            source_refs=sorted(
                {frame.get("source", "") for frame in segment.get("frames", [])}
            ),
            confidence=None,
            evidence=[
                ev
                for ev in evidence
                if ev.batch == segment.get("batch_index")
            ],
        )
        for segment in successful
    ]

    if final_schema is not None:
        data = global_result.get("data")
        synthesis = _serialize(data)
    else:
        data = global_result.get("data")
        synthesis = global_result.get("text")

    await emitter.emit(SYNTHESIS_COMPLETED)
    result = AnalysisResult(
        task=task,
        data=data,
        observations=observations,
        evidence=evidence,
        synthesis=synthesis,
        usage=tracker.to_summary(),
        media=stats,
        errors=errors,
    )
    if cache is not None and full_pipeline_key is not None:
        cache.put(
            full_pipeline_key,
            _result_to_cache(result),
        )
    return result


def _effective_max_output_tokens(
    options: AnalyzeOptions | dict | None,
    opts: AnalyzeOptions,
    preset: Any,
) -> int:
    """Honor preset max_output_tokens unless the caller overrode it."""
    if options is None:
        return preset.max_output_tokens
    if isinstance(options, AnalyzeOptions):
        if options.max_output_tokens != 8192:
            return options.max_output_tokens
        return preset.max_output_tokens
    if isinstance(options, dict) and "max_output_tokens" in options:
        return opts.max_output_tokens
    return preset.max_output_tokens


def _result_to_cache(result: AnalysisResult) -> dict:
    return {
        "task": result.task,
        "data": _jsonable(result.data),
        "observations": [
            obs.model_dump(mode="json") for obs in result.observations
        ],
        "evidence": [ev.model_dump(mode="json") for ev in result.evidence],
        "synthesis": result.synthesis,
        "usage": result.usage.model_dump(mode="json"),
        "media": result.media.model_dump(mode="json"),
        "errors": result.errors,
    }


def _result_from_cache(
    cached: dict, *, task: str, stats: MediaStats
) -> AnalysisResult:
    return AnalysisResult(
        task=task,
        data=cached.get("data"),
        observations=[
            Observation.model_validate(obs)
            for obs in cached.get("observations", [])
        ],
        evidence=[
            Evidence.model_validate(ev) for ev in cached.get("evidence", [])
        ],
        synthesis=cached.get("synthesis"),
        usage=UsageSummary.model_validate(cached.get("usage", {})),
        media=MediaStats.model_validate(cached.get("media", {}))
        if cached.get("media")
        else stats,
        errors=cached.get("errors", []),
        meta={"from_cache": True},
    )


async def _run_single_batch(
    *,
    frames: list[MediaFrame],
    preset_task_instructions: str,
    preset_system_prompt: str,
    instructions: str | None,
    schema: type[BaseModel] | None,
    temperature: float | None,
    max_output_tokens: int,
    detail: str,
    user_id: str | None,
    task: str,
    stats: MediaStats,
    tracker: UsageTracker,
    emitter: EventEmitter,
    cache: Cache | None,
    api_key: str,
    base_url: str,
    model: str,
    opts: AnalyzeOptions,
) -> AnalysisResult:
    full_prompt = build_analysis_prompt(
        preset_task_instructions, instructions, format_frames(frames)
    )

    cache_key: str | None = None
    if cache is not None:
        cache_key = cache.key(
            {
                "media_hashes": cache.match_media(frames),
                "model": model,
                "task": task,
                "instructions": full_prompt,
                "schema_digest": schema_digest(schema),
                "detail": detail,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        cached = cache.get(cache_key)
        if cached is not None:
            usage = UsageSummary.model_validate(cached.get("usage", {}))
            data = cached.get("data")
            text = cached.get("text", _serialize(data))
            return _build_single_result(
                task=task,
                frames=frames,
                data=data,
                text=text,
                usage=usage,
                stats=stats,
                errors=[],
                meta={"from_cache": True},
            )

    client = DeepSeekMultimodalClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries=opts.max_retries,
        timeout_seconds=opts.timeout_seconds,
        retry_base_delay=opts.retry_base_delay,
        max_concurrency=opts.max_concurrency,
        event_emitter=emitter,
        usage_tracker=tracker,
    )
    try:
        parsed = await client.analyze(
            frames,
            prompt=full_prompt,
            system=preset_system_prompt,
            output_schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            detail=detail,
            user_id=user_id,
        )
    finally:
        await client.aclose()

    result = _build_single_result(
        task=task,
        frames=frames,
        data=parsed["data"],
        text=parsed["text"],
        usage=tracker.to_summary(),
        stats=stats,
        errors=[],
        meta={},
    )
    if cache is not None and cache_key is not None:
        cache.put(
            cache_key,
            {
                "data": _jsonable(parsed["data"]),
                "text": parsed["text"],
                "usage": tracker.to_summary().model_dump(mode="json"),
            },
        )
    return result


def _build_single_result(
    *,
    task: str,
    frames: list[MediaFrame],
    data: Any,
    text: str,
    usage: UsageSummary,
    stats: MediaStats,
    errors: list[str],
    meta: dict,
) -> AnalysisResult:
    serialized = text if data is None else _serialize(data)
    evidence = _extract_evidence_from_frames(frames, serialized)
    observation = Observation(
        text=serialized,
        source_refs=sorted({frame.source for frame in frames}),
        confidence=None,
        evidence=evidence,
    )
    return AnalysisResult(
        task=task,
        data=data,
        observations=[observation],
        evidence=evidence,
        synthesis=text,
        usage=usage,
        media=stats,
        errors=errors,
        meta=meta,
    )


def analyze_media_sync(*args: Any, **kwargs: Any) -> AnalysisResult:
    """Synchronous wrapper around :func:`analyze_media`."""
    return asyncio.run(analyze_media(*args, **kwargs))
