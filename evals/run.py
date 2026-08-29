"""Evaluation harness CLI: ``python -m evals.run``.

Generates the media fixtures, runs the selected cases against the selected
providers, and writes ``report.md`` / ``report.json`` under ``--out``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from evals.cases import BUILTIN_CASES, VisionEvalCase, generate_media
from evals.providers import get_provider
from evals.runner import run_all, write_report

DEFAULT_MEDIA_DIR = "./.eval-media"
DEFAULT_OUT = "./eval-report"
DEFAULT_PROVIDERS = "deepseek"


class ConfigError(Exception):
    """Configuration error: exits with code 2."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description=(
            "Provider-independent evaluation harness for multimodal analysis."
        ),
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated case ids or 'all' (default: all)",
    )
    parser.add_argument(
        "--providers",
        default=DEFAULT_PROVIDERS,
        help="comma-separated provider names or 'all' (default: deepseek)",
    )
    parser.add_argument(
        "--media-dir",
        default=DEFAULT_MEDIA_DIR,
        help="media fixture directory, auto-generated (default: ./.eval-media)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="report output directory (default: ./eval-report)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="max concurrent case executions (default: 2)",
    )
    return parser


def _select_cases(spec: str, media_dir: Path) -> list[VisionEvalCase]:
    by_id = {case.id: case for case in BUILTIN_CASES}
    if spec == "all":
        return [
            case
            for case in BUILTIN_CASES
            if all((media_dir / rel).exists() for rel in case.media)
        ]
    selected: list[VisionEvalCase] = []
    for raw in spec.split(","):
        case_id = raw.strip()
        if not case_id:
            continue
        case = by_id.get(case_id)
        if case is None:
            raise ConfigError(
                f"unknown case id {case_id!r}; available: "
                f"{', '.join(sorted(by_id))}"
            )
        missing = [
            rel for rel in case.media if not (media_dir / rel).exists()
        ]
        if missing:
            raise ConfigError(
                f"case {case_id!r} needs missing media files: {missing}"
            )
        selected.append(case)
    return selected


def _select_providers(spec: str) -> list[Any]:
    if spec == "all":
        names = ["deepseek"]
    else:
        names = [raw.strip() for raw in spec.split(",") if raw.strip()]
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if "deepseek" in names and not api_key:
        raise ConfigError(
            "DEEPSEEK_API_KEY is required for the deepseek provider; "
            "export it before running the eval"
        )
    providers: list[Any] = []
    for name in names:
        try:
            providers.append(get_provider(name, api_key=api_key))
        except KeyError as exc:
            raise ConfigError(str(exc)) from exc
    return providers


async def _run(args: argparse.Namespace) -> int:
    media_dir = Path(args.media_dir)
    generate_media(media_dir)
    cases = _select_cases(args.cases, media_dir)
    providers = _select_providers(args.providers)
    try:
        results = await run_all(
            cases, providers, media_dir, max_concurrency=args.concurrency
        )
    finally:
        for provider in providers:
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()

    out = Path(args.out)
    write_report(results, out)
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(
            f"{status:4s} {result.case_id} [{result.provider}] "
            f"{result.success_detail}"
        )
    print(
        f"report written to {out / 'report.md'} and {out / 'report.json'} "
        f"({len(results)} result(s))"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
