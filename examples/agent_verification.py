"""Verify visual claims about a UI reference screenshot."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from make_fixtures import make_all

from deepseek_vision import analyze_media

HERE = Path(__file__).resolve().parent

CLAIMS = (
    "Verify these claims:\n"
    "1. A blue primary button is visible\n"
    "2. The label reads 'Sign Up'\n"
    "3. A footer exists"
)


async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set — skipping (set it to run live)")
        return
    fixtures = make_all(HERE / "fixtures")
    path = fixtures["design_reference"]
    result = await analyze_media(
        str(path), task="verification", instructions=CLAIMS
    )
    data = result.data
    for check in data.checks:
        print(
            f"[{check.status}] {check.claim} "
            f"(confidence: {check.confidence}, evidence: {check.evidence})"
        )
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
