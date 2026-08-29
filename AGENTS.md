# AGENTS.md

Project: deepseek-vision — provider-independent multimodal media analysis built on
DeepSeek V4 Flash Vision Exp (`deepseek-v4-flash-vision-exp`).

## Layout

- `src/deepseek_vision/` — library package (import name `deepseek_vision`)
  - `skill.py` — `analyze_media()`: the single public orchestration primitive
  - `client.py` — DeepSeek API transport ONLY (httpx). No media processing here.
  - `media/` — deterministic preprocessing: images, video (ffmpeg), pdf (PyMuPDF), batching
  - `presets/` — task presets as prompt/config data. No business logic.
  - `models.py`, `errors.py`, `usage.py`, `cache.py`, `synthesis.py`, `events.py`, `prompts.py`
- `tests/` — unit tests; deterministic parts mocked with respx; NO live API calls
- `tests/integration/` — live-API tests, skipped unless `DEEPSEEK_API_KEY` is set
- `evals/` — provider-independent evaluation harness
- `examples/` — runnable end-to-end examples

## Conventions

- Python >= 3.10 syntax (`X | None` unions). Pydantic v2 for all public surfaces.
- Async I/O throughout. Bounded concurrency (asyncio.Semaphore) — never unbounded gather.
- All provider limits/prices live in `usage.py` / `models.py` as constants, configurable.
- Deterministic work (decode, frame extraction, PDF render, batching, caching, validation)
  must never depend on the LLM. LLM sits at the semantic boundary only.
- Never log API keys, auth headers, or sensitive media bytes.
- Unit tests: no network, no API keys. Integration tests live in `tests/integration/`
  and skip without `DEEPSEEK_API_KEY`.

## Tooling

- uv for dependency management: `uv sync`, `uv run pytest`
- lint: `uvx ruff check src/ tests/`
- Python 3.13 available on this machine; keep `requires-python >= 3.10`
