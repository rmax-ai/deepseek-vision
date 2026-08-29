"""Visual regression: compare a design reference against the implementation."""

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
    reference = fixtures["design_reference"]
    actual = fixtures["rendered_actual"]
    result = await analyze_media(
        [str(reference), str(actual)], task="visual_regression"
    )
    data = result.data
    print(f"differences ({len(data.differences)}):")
    for difference in data.differences:
        print(f"  - {difference}")
    print(f"implementation_notes: {data.implementation_notes}")
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
