"""Movement analysis of a synthetic video: phases and coaching feedback."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from make_fixtures import make_all

from deepseek_vision import VideoOptions, analyze_media

HERE = Path(__file__).resolve().parent


async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set — skipping (set it to run live)")
        return
    fixtures = make_all(HERE / "fixtures")
    path = fixtures["session"]
    if not path.exists():
        print("session.mp4 fixture missing (ffmpeg unavailable) — skipping")
        return
    result = await analyze_media(
        str(path),
        task="movement_analysis",
        options={
            "video": VideoOptions(
                sampling="adaptive",
                fps=0.5,
                max_frames=48,
                scene_threshold=0.3,
            )
        },
    )
    data = result.data
    print(f"phases ({len(data.phases)}):")
    for phase in data.phases:
        print(f"  - {phase}")
    print(f"patterns: {data.patterns}")
    print(f"issues: {data.issues}")
    print(f"coaching_feedback: {data.coaching_feedback}")
    print(f"cited_timestamps: {data.cited_timestamps}")
    print(f"usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
