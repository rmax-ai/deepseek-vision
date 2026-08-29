"""Debug a UI screenshot: severity, symptoms, cause, and suggested fix."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from make_fixtures import make_all

from deepseek_vision import analyze_media

HERE = Path(__file__).resolve().parent


async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set — skipping (set it to run live)")
        return
    fixtures = make_all(HERE / "fixtures")
    path = fixtures["ui_screenshot"]
    result = await analyze_media(str(path), task="bug_report")
    data = result.data
    print(f"severity: {data.severity}")
    print(f"symptoms: {data.symptoms}")
    print(f"likely_cause: {data.likely_cause}")
    print(f"suggested_fix: {data.suggested_fix}")
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
