"""Analyze an architecture diagram: components, relationships, data flow."""

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
    path = fixtures["architecture"]
    result = await analyze_media(str(path), task="diagram_analysis")
    data = result.data
    print(f"components ({len(data.components)}):")
    for component in data.components:
        print(f"  - {component}")
    print(f"relationships ({len(data.relationships)}):")
    for relationship in data.relationships:
        print(f"  - {relationship}")
    print(f"data_flow: {data.data_flow}")
    print(f"risks: {data.potential_risks}")
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
