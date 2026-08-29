"""Command-line interface for deepseek-vision.

Subcommands: ``analyze``, ``compare``, ``video``, ``document``. Global flags
(``--json``, ``--usage``, ``--verbose``, ``--no-cache``, ``--model``,
``--timeout``, ``--concurrency``, ``--max-images-per-request``,
``--temperature``, ``--max-tokens``) apply to every subcommand.

Options are resolved exactly like :func:`deepseek_vision.skill.analyze_media`
does: environment variables (``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL``,
``DEEPSEEK_MULTIMODAL_MODEL``) fill in anything the caller does not set
explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from pydantic import BaseModel

from .errors import ConfigurationError, DeepSeekVisionError
from .events import Event
from .models import (
    AnalysisResult,
    AnalyzeOptions,
    DocumentOptions,
    VideoOptions,
)
from .skill import analyze_media


class InvoiceExtraction(BaseModel):
    """CLI example schema: structured invoice extraction."""

    vendor: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    total: float | None = None
    currency: str | None = None


class TableExtraction(BaseModel):
    """CLI example schema: structured table extraction."""

    headers: list[str]
    rows: list[list[str]]


_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "table": TableExtraction,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepseek-vision",
        description=(
            "Provider-independent multimodal media analysis powered by "
            "DeepSeek V4 Flash Vision Exp."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_global_flags(common)

    analyze = subparsers.add_parser(
        "analyze",
        parents=[common],
        help="analyze one or more images (or a single PDF/video/URL)",
        description=(
            "Analyze one or more media inputs. Multiple inputs must all be "
            "images (they are sent as one collection); pass a single PDF or "
            "video path for documents/videos."
        ),
    )
    analyze.add_argument("media", nargs="+", help="image path(s) or URL")
    analyze.add_argument(
        "--task",
        default=None,
        help="task preset name (default: describe)",
    )
    analyze.add_argument(
        "--instructions", default=None, help="extra task instructions"
    )
    analyze.add_argument(
        "--schema",
        choices=sorted(_SCHEMAS),
        default=None,
        help=(
            "example output schema (invoice|table); when given, the default "
            "task becomes 'extract'"
        ),
    )

    compare = subparsers.add_parser(
        "compare",
        parents=[common],
        help="compare two images",
        description="Compare two images with the compare or visual_regression preset.",
    )
    compare.add_argument("image_a", help="first image (reference in visual_regression)")
    compare.add_argument("image_b", help="second image (implementation in visual_regression)")
    compare.add_argument(
        "--task",
        choices=["compare", "visual_regression"],
        default="compare",
        help="task preset (default: compare)",
    )

    video = subparsers.add_parser(
        "video",
        parents=[common],
        help="analyze a video file",
        description="Analyze a local video file with a video task preset.",
    )
    video.add_argument("path", help="video file path (mp4/mov/mkv/webm/avi/m4v)")
    video.add_argument(
        "--task",
        choices=["video_summary", "temporal_analysis", "movement_analysis"],
        default="video_summary",
        help="task preset (default: video_summary)",
    )
    video.add_argument(
        "--sampling",
        choices=["uniform", "scene_change", "keyframes", "adaptive"],
        default=None,
        help="frame sampling strategy (default: adaptive)",
    )
    video.add_argument("--fps", type=float, default=None, help="uniform sampling rate")
    video.add_argument("--max-frames", type=int, default=None, help="maximum frames to extract")
    video.add_argument(
        "--scene-threshold", type=float, default=None, help="scene-change detection threshold"
    )
    video.add_argument(
        "--window-seconds", type=float, default=None, help="temporal window size for batching"
    )
    video.add_argument("--start", type=float, default=None, help="start time in seconds")
    video.add_argument("--end", type=float, default=None, help="end time in seconds")

    document = subparsers.add_parser(
        "document",
        parents=[common],
        help="analyze a PDF document",
        description="Analyze a local PDF with a document task preset.",
    )
    document.add_argument("path", help="PDF file path")
    document.add_argument(
        "--task",
        choices=["document_analysis", "page_extraction"],
        default="document_analysis",
        help="task preset (default: document_analysis)",
    )
    document.add_argument(
        "--pages", default=None, help='page range, e.g. "1-10" or "1,3,5-8"'
    )
    document.add_argument("--dpi", type=int, default=None, help="render resolution (default: 150)")
    document.add_argument(
        "--max-pages-per-batch",
        type=int,
        default=None,
        help="cap pages per API request",
    )

    return parser


def _add_global_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full analysis result as JSON",
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="print the usage/cost summary block to stderr at the end",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print pipeline events (name + key data) to stderr",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="disable the result cache"
    )
    parser.add_argument("--model", default=None, help="model id override")
    parser.add_argument(
        "--timeout", type=float, default=None, help="request timeout in seconds"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="max concurrent API requests"
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=None,
        help="max images per API request",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="max output tokens"
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="override thinking mode (presets default to disabled)",
    )


async def _verbose_handler(event: Event) -> None:
    """Print every pipeline event with its key data to stderr."""
    pairs = " ".join(f"{k}={v}" for k, v in sorted(event.data.items()))
    print(f"[{event.name}] {pairs}", file=sys.stderr)


def _build_options(args: argparse.Namespace) -> AnalyzeOptions:
    """Build AnalyzeOptions from CLI flags; unset values stay None so the
    skill resolves env defaults exactly as it would with no options."""
    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    if args.timeout is not None:
        kwargs["timeout_seconds"] = args.timeout
    if args.concurrency is not None:
        kwargs["max_concurrency"] = args.concurrency
    if args.max_images_per_request is not None:
        kwargs["max_images_per_request"] = args.max_images_per_request
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_output_tokens"] = args.max_tokens
    if args.thinking is not None:
        kwargs["thinking"] = args.thinking
    if args.no_cache:
        kwargs["use_cache"] = False
    if args.verbose:
        kwargs["event_handler"] = _verbose_handler
    if args.command == "video":
        kwargs["video"] = _video_options(args)
    if args.command == "document":
        kwargs["document"] = _document_options(args)
    return AnalyzeOptions(**kwargs)


def _video_options(args: argparse.Namespace) -> VideoOptions:
    fields: dict[str, Any] = {}
    if args.sampling is not None:
        fields["sampling"] = args.sampling
    if args.fps is not None:
        fields["fps"] = args.fps
    if args.max_frames is not None:
        fields["max_frames"] = args.max_frames
    if args.scene_threshold is not None:
        fields["scene_threshold"] = args.scene_threshold
    if args.window_seconds is not None:
        fields["window_seconds"] = args.window_seconds
    if args.start is not None:
        fields["start_seconds"] = args.start
    if args.end is not None:
        fields["end_seconds"] = args.end
    return VideoOptions(**fields)


def _document_options(args: argparse.Namespace) -> DocumentOptions:
    fields: dict[str, Any] = {}
    if args.pages is not None:
        fields["pages"] = args.pages
    if args.dpi is not None:
        fields["dpi"] = args.dpi
    if args.max_pages_per_batch is not None:
        fields["max_pages_per_batch"] = args.max_pages_per_batch
    return DocumentOptions(**fields)


def _task_for(args: argparse.Namespace) -> str:
    """Resolve the effective task, honoring the --schema default."""
    if args.command == "analyze":
        return args.task or ("extract" if args.schema else "describe")
    return args.task


async def _run_command(args: argparse.Namespace) -> AnalysisResult:
    """Dispatch to analyze_media based on the selected subcommand."""
    options = _build_options(args)
    task = _task_for(args)
    schema: type[BaseModel] | None = None
    if getattr(args, "schema", None):
        schema = _SCHEMAS[args.schema]

    if args.command == "analyze":
        media: Any = args.media[0] if len(args.media) == 1 else args.media
        return await analyze_media(
            media,
            task=task,
            instructions=args.instructions,
            output_schema=schema,
            options=options,
        )
    if args.command == "compare":
        return await analyze_media(
            [args.image_a, args.image_b], task=task, options=options
        )
    if args.command in ("video", "document"):
        return await analyze_media(args.path, task=task, options=options)
    raise ConfigurationError(f"unknown command: {args.command}")


def _render_human(result: AnalysisResult) -> str:
    """Human-readable rendering of an AnalysisResult."""
    lines: list[str] = []
    data = result.data
    if isinstance(data, BaseModel):
        lines.append(json.dumps(data.model_dump(mode="json"), indent=2, default=str))
    elif isinstance(data, dict):
        lines.append(json.dumps(data, indent=2, default=str))
    elif data is not None:
        lines.append(str(data))
    if result.synthesis:
        lines.append(f"synthesis: {result.synthesis}")
    for ev in result.evidence:
        provenance: list[str] = [f"src={ev.source}"]
        if ev.page is not None:
            provenance.append(f"page={ev.page}")
        if ev.timestamp_seconds is not None:
            provenance.append(f"t={ev.timestamp_seconds:.2f}")
        lines.append(f"{' '.join(provenance)} | {ev.observation}")
    for error in result.errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def _render_json(result: AnalysisResult) -> str:
    """JSON rendering of the full AnalysisResult."""
    payload = {
        "task": result.task,
        "data": (
            result.data.model_dump(mode="json")
            if isinstance(result.data, BaseModel)
            else result.data
        ),
        "observations": [
            obs.model_dump(mode="json") for obs in result.observations
        ],
        "evidence": [ev.model_dump(mode="json") for ev in result.evidence],
        "synthesis": result.synthesis,
        "usage": result.usage.model_dump(mode="json"),
        "media": result.media.model_dump(mode="json"),
        "errors": result.errors,
    }
    return json.dumps(payload, indent=2, default=str)


def _render_usage(result: AnalysisResult) -> str:
    """Telemetry/usage summary block."""
    u = result.usage
    return (
        "usage:\n"
        f"  requests: {u.requests}\n"
        f"  images processed: {u.images_processed}\n"
        f"  input tokens: {u.input_tokens}\n"
        f"  output tokens: {u.output_tokens}\n"
        f"  cache hit tokens: {u.cache_hit_tokens}\n"
        f"  cache miss tokens: {u.cache_miss_tokens}\n"
        f"  estimated cost (USD): {u.estimated_cost_usd}\n"
        f"  latency (s): {u.latency_seconds:.2f}\n"
        f"  retries: {u.retries}\n"
        f"  from cache: {u.from_cache}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit codes: 0 on success, 1 on any DeepSeekVisionError, 2 on argparse
    usage errors (raised by argparse itself).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run_command(args))
    except DeepSeekVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = _render_json(result) if args.json else _render_human(result)
    print(output)
    if args.usage:
        print(_render_usage(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
