"""Analyze a long PDF with page-cited observations and targeted answers."""

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
    path = fixtures["report"]
    result = await analyze_media(
        str(path),
        task="document_analysis",
        instructions=(
            "Answer: when is the migration scheduled, and what is the "
            "current churn rate?"
        ),
        options={"document": {"max_pages_per_batch": 2}},
    )
    data = result.data
    print(f"summary: {data.summary}")
    cited = [obs for obs in data.observations if obs.page is not None]
    print(f"page-cited observations: {len(cited)}")
    for obs in cited[:10]:
        print(f"  p{obs.page}: {obs.text}")
    answer = data.answer or ""
    print(f"answer: {answer}")
    for needle in ("Q4", "2.3%"):
        print(f"  contains {needle!r}: {needle.lower() in answer.lower()}")
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
