"""Case execution, concurrency, and report writing for the eval harness."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deepseek_vision.media.images import load_collection
from deepseek_vision.media.pdf import render_pages
from deepseek_vision.media.video import extract_frames
from deepseek_vision.models import (
    DocumentOptions,
    ImageCollectionInput,
    ImageInput,
    MediaFrame,
    VideoOptions,
)
from deepseek_vision.presets import get_preset
from deepseek_vision.prompts import schema_to_prompt

from .cases import SCHEMAS, VisionEvalCase
from .evaluators import (
    EVALUATORS,
    grounding_score,
    llm_judge,
    structured_valid,
    success_latency_token_metrics,
)


class EvalResult(BaseModel):
    """Outcome of running one case against one provider."""

    case_id: str
    provider: str
    success: bool
    success_detail: str
    structured_valid: bool
    grounding: float
    latency_s: float | None = None
    requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error: str | None = None


async def _load_frames(case: VisionEvalCase, media_dir: Path) -> list[MediaFrame]:
    """Load a case's media into frames per its media tag."""
    frames: list[MediaFrame] = []
    video_options = VideoOptions(**case.options.get("video", {}))
    document_options = DocumentOptions(**case.options.get("document", {}))
    for rel in case.media:
        path = str(media_dir / rel)
        if "video" in case.tags:
            frames.extend(await extract_frames(path, video_options))
        elif "document" in case.tags:
            frames.extend(await render_pages(path, document_options))
        else:
            frames.extend(
                await load_collection(
                    ImageCollectionInput(images=[ImageInput(path=path)])
                )
            )
    frames.sort(
        key=lambda f: f.timestamp if f.timestamp is not None else float("inf")
    )
    return frames


def resolve_schema(case: VisionEvalCase) -> type[BaseModel] | None:
    """Resolve the output schema: case-named schema or the preset's."""
    if case.output_schema_name and case.output_schema_name in SCHEMAS:
        return SCHEMAS[case.output_schema_name]
    if case.task != "custom":
        preset = get_preset(case.task)
        return preset.output_schema
    return None


def build_prompt(case: VisionEvalCase, schema: type[BaseModel] | None = None) -> str:
    """Compose the provider prompt: preset instructions + case prompt.

    When a structured schema is in play, the JSON schema and an example are
    embedded so the model returns schema-conforming JSON.
    """
    if case.task == "custom":
        prompt = case.prompt
    else:
        preset = get_preset(case.task)
        prompt = "\n\n".join(
            part for part in (preset.task_instructions, case.prompt) if part
        )
    if schema is not None:
        schema_str, example_str = schema_to_prompt(schema)
        prompt += (
            "\n\nRespond with a single JSON object that matches the following "
            f"schema exactly. No markdown fences, no commentary.\n"
            f"JSON Schema:\n{schema_str}\n\nExample JSON:\n{example_str}"
        )
    return prompt


def build_system(case: VisionEvalCase) -> str | None:
    """Resolve the system prompt: case override or preset default."""
    if case.task == "custom":
        return case.system
    preset = get_preset(case.task)
    return case.system or preset.system_prompt


async def run_case(
    case: VisionEvalCase, provider: Any, media_dir: Path
) -> EvalResult:
    """Run one case against one provider and evaluate the result."""
    started = time.monotonic()
    try:
        frames = await _load_frames(case, media_dir)
        if not frames:
            return EvalResult(
                case_id=case.id,
                provider=provider.name,
                success=False,
                success_detail="no frames could be loaded from media",
                structured_valid=False,
                grounding=0.0,
                error="no frames loaded",
            )

        schema = resolve_schema(case)
        response = await provider.analyze_frames(
            frames,
            prompt=build_prompt(case, schema),
            system=build_system(case),
            output_schema=schema,
        )

        result = {
            "data": response.data,
            "synthesis": response.text,
            "observations": [],
            "evidence": [],
        }
        evaluator = EVALUATORS.get(case.evaluator, llm_judge)
        success, detail = evaluator(result, case)
        valid = structured_valid(result, case)
        evidence = [
            {
                "source": frame.source,
                "page": frame.metadata.get("page"),
                "timestamp_seconds": frame.timestamp,
            }
            for frame in frames
        ]
        metrics = success_latency_token_metrics(response, case)
        return EvalResult(
            case_id=case.id,
            provider=provider.name,
            success=success,
            success_detail=detail,
            structured_valid=valid,
            grounding=grounding_score(evidence, case),
            **metrics,
        )
    except Exception as exc:  # noqa: BLE001 - per-case failures are recorded
        elapsed = time.monotonic() - started
        return EvalResult(
            case_id=case.id,
            provider=provider.name,
            success=False,
            success_detail="case raised an exception",
            structured_valid=False,
            grounding=0.0,
            latency_s=elapsed,
            requests=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_all(
    cases: list[VisionEvalCase],
    providers: list[Any],
    media_dir: Path,
    max_concurrency: int = 2,
) -> list[EvalResult]:
    """Run every case against every provider, bounded by a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(case: VisionEvalCase, provider: Any) -> EvalResult:
        async with semaphore:
            return await run_case(case, provider, media_dir)

    tasks = [
        bounded(case, provider)
        for case in cases
        for provider in providers
    ]
    return await asyncio.gather(*tasks)


def write_report(results: list[EvalResult], out: Path) -> None:
    """Write report.md (summary table + details) and report.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.json").write_text(
        json.dumps(
            [result.model_dump(mode="json") for result in results], indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    lines = ["# Evaluation report", ""]
    lines.append(
        "| case | provider | success | grounding | latency (s) | "
        "input tokens | output tokens | cost (USD) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.provider} | "
            f"{'PASS' if result.success else 'FAIL'} | {result.grounding:.2f} | "
            f"{result.latency_s if result.latency_s is not None else '-'} | "
            f"{result.input_tokens if result.input_tokens is not None else '-'} | "
            f"{result.output_tokens if result.output_tokens is not None else '-'} | "
            f"{result.estimated_cost_usd if result.estimated_cost_usd is not None else '-'} |"
        )
    lines.append("")
    lines.append("## Per-case details")
    lines.append("")
    for result in results:
        status = "PASS" if result.success else "FAIL"
        lines.append(f"### {result.case_id} [{result.provider}] - {status}")
        lines.append("")
        lines.append(f"- success_detail: {result.success_detail}")
        if result.error:
            lines.append(f"- error: {result.error}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
