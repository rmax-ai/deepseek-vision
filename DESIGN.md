# deepseek-vision — Design Document

## Goal

One multimodal primitive, not a collection of bespoke vision scripts.

```text
media ──► deterministic preprocessing ──► batched DeepSeek inference ──► structured observations ──► optional synthesis
   │            (PIL / ffmpeg / PyMuPDF)          (bounded concurrency)          (schemas + evidence)          (map/reduce)
```

An agent says *"understand this visual artifact for this purpose"* — `analyze_media()` decides
how to preprocess, batch, call the model, preserve provenance, and synthesize a result.
A second agent (or an eval harness) gets machine-readable evidence back.

## Decision 1: The model sits at the semantic boundary only

Deterministic code owns everything deterministic:

| Task | Owner | Why not the LLM |
|---|---|---|
| media decoding / EXIF / resize | PIL | lossless, free |
| video duration / fps | ffprobe | exact, free |
| frame selection / timestamps | ffmpeg | reproducible, free |
| PDF page rendering | PyMuPDF | deterministic, free |
| batch-size math | `media/batching.py` | arithmetic |
| caching | `cache.py` | hashing |
| schema validation | Pydantic | correctness |
| visual interpretation & reasoning | DeepSeek | the actual capability |

Asking the model to do what code does reliably costs tokens, adds failure modes, and breaks
reproducibility. The prompt layer is built so the model only ever receives *preprocessed frames
with provenance labels*.

## Decision 2: httpx transport instead of the `openai` SDK

DeepSeek's API is OpenAI Chat Completions-compatible, so the `openai` SDK would work.
We deliberately use plain httpx:

- **Ownership of retry semantics.** DeepSeek's rate limiting is concurrency-based (HTTP 429 +
  `Retry-After`), and its JSON mode has a documented "occasionally empty content" quirk that
  must be treated as retryable. SDK retry policies are opaque and version-churny.
- **Provider isolation.** The eval harness defines a provider protocol; future
  `GeminiMultimodalClient` / `OpenAIMultimodalClient` / `AnthropicMultimodalClient` implement
  the same interface without touching media processing or eval cases. A single HTTP layer we
  own keeps that boundary clean.
- **Small dependency surface** on a constrained machine (this box has 3.7 GiB RAM).

Trade-off: we own response parsing. That parsing is ~40 lines and is unit-tested with `respx`
mocks of the real wire format.

## Decision 3: Structured output = JSON mode + Pydantic, not tool calls

DeepSeek documents `response_format={"type": "json_object"}` for the vision model; there is no
strict `json_schema` enforcement. Therefore the structured-output contract is three layers:

1. Prompt-embedded JSON schema + concrete example (required by the API for JSON mode).
2. Pydantic v2 validation of the parsed response.
3. Exactly **one** retry with validation-feedback appended, then hard failure
   (`ResponseValidationError`) — two chances, bounded cost, no infinite loops.

Tool/function calling is intentionally not used for extraction: single-shot JSON is cheaper,
simpler, and sufficient for the extraction/verification/comparison surface. Function calling
remains available at the raw API level for agentic loops built on top of this skill.

## Decision 4: Context-aware batching, conservative defaults

Batch size is computed, not hard-coded:

```text
batch = min( n, ⌊(ctx·0.5 − prompt_tokens − reserved_output)/384⌋, ⌊64 MiB / max_frame_bytes⌋, 600, max_images_per_request )
```

- **384 tokens/image** is the documented upper bound (images are auto-resized to ~800×800;
  larger images bill identically).
- **Safety factor 0.5** on the 1M context: prompt-token estimation is chars/3 — deliberately
  crude, so we never build a batch that risks truncation.
- **64 MiB inline cap** and **600-image cap** are provider limits.
- **Default `max_images_per_request=16`** is a practical ceiling below all provider limits:
  keeps per-request latency and retry-cost bounded, and avoids the dimension-drop rule
  (≥15 images → 8192→4096 px).

All constants are configurable (`AnalyzeOptions`), because provider docs drift.

## Decision 5: Hierarchical map/reduce with provenance threaded through

Long media (video, multi-page documents) is never sent raw. Pipeline:

```text
frames/pages → batches → per-batch observations → global synthesis
                                └─────────────── evidence survives every stage
```

`Evidence(source, page, timestamp_seconds, batch, observation, confidence)` is emitted at the
batch stage and *merged, never dropped*, into `AnalysisResult.evidence`. The global synthesis
stage may only re-read and cite; it cannot erase. Every global claim is traceable to
source file + page/frame + timestamp + batch.

Trade-off: global synthesis can omit or blur details. Mitigation: the full evidence list is
always returned alongside the synthesis, so downstream agents can trust `evidence` even when
`synthesis` is prose.

## Decision 6: Video strategy — ffmpeg sampling, never blind frame dumps

The DeepSeek API has no native video input. `media/video.py` provides four sampling
strategies, all producing sorted, timestamped JPEG frames:

- `uniform` — fps-based; deterministic timestamp recomputation (no stderr parsing).
- `scene_change` — ffmpeg `select='gt(scene,θ)'` + synthetic bookends (t=0, last frame).
- `keyframes` — `-skip_frame nokey` with pts from `-frame_pts 1`.
- `adaptive` (default) — union of scene-change + sparse uniform, timestamp-deduped, capped at
  `max_frames` while preserving temporal spread and always keeping t=0 + last frame.

`max_frames=128` default keeps a long video to ≤128 API images (~49K tokens worst case).
Long videos are additionally split into temporal windows (`window_seconds`) so analysis is
hierarchical rather than one giant context dump.

## Decision 7: Documents — render pages, keep page provenance

The Files API accepts images only (no PDF ingestion). PyMuPDF renders selected pages at 150
dpi to JPEG (q85). Page ranges (`"1-10,15,20-25"`) are parsed deterministically. Every page
frame carries `metadata.page` and `total_pages`; batch prompts instruct the model to cite page
numbers, and `Evidence.page` preserves them.

Trade-off: rendering discards embedded text layer and OCR-free text extraction. That is
deliberate — this is a *visual* understanding primitive. When exact text is the goal, the
skill documentation explicitly redirects to OCR/deterministic extraction instead.

## Decision 8: Content-addressed caching

Cache key = sha256 over: media bytes + per-frame timestamps, model, task, instructions,
schema digest (JSON schema, sorted), and inference parameters (detail, temperature,
max_output_tokens). Two granularities:

- **Full-pipeline cache** for hierarchical runs (video/document): the entire final
  `AnalysisResult` is cached — this is the big cost saver for repeated video analysis.
- **Per-batch cache**: partial reuse when only some windows/pages changed.

Trade-off: manual invalidation on model upgrades (model id is in the key; changing
`DEEPSEEK_MULTIMODAL_MODEL` or bumping schema naturally misses). Cache dir defaults to
`~/.cache/deepseek-vision/`.

## Decision 9: Usage accounting — actuals first, conservative estimates otherwise

`usage.py` prefers the API's real `usage` fields (`prompt_tokens`, `completion_tokens`,
`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`). When cache fields are absent, input
tokens are priced at the *cache-miss* rate (conservative). Peak pricing (Mon–Fri 01:00–04:00
and 06:00–10:00 UTC) is modeled from the official pricing table. All prices live in one
module so provider price changes are one-line edits. The skill reports cost on every run —
agents should know what vision costs before they spend it.

## Decision 10: Bounded concurrency, backoff with jitter

- `asyncio.Semaphore(max_concurrency)` — default 4, configurable. Never unbounded
  `asyncio.gather`.
- Exponential backoff (base 1s, ×2, ±20% jitter) on 429/5xx/timeouts, honoring `Retry-After`.
- Provider concurrency limit is 2500 — we self-limit far below it for stability on small
  machines and fairness; the limit is documented, not emulated.

## Decision 11: Presets are data, not code

Each preset is a `Preset` model: system prompt, task instructions, optional batch/final
schemas, temperature, hierarchical flag. Adding a task = adding a data record. Business logic
lives only in `skill.py` (orchestration) and `synthesis.py` (map/reduce), never per-task code.

## Decision 12: Skill ergonomics — progressive disclosure + "when not to use"

SKILL.md is deliberately small (<100 lines): what it does, when to use it, the primary
interface, a few examples. Deep workflows live in `references/`. The skill explicitly tells
agents *not* to use vision when deterministic tools suffice — JSON parser for JSON, DOM for
HTML structure, ffprobe for duration, OCR for exact text. Vision is for semantics; misuse is
expensive (≤384 tokens/image is cheap per image but not free at agent scale).

## Decision 13: Thinking mode — disabled by default at the preset layer

DeepSeek V4 models have chain-of-thought **enabled by default** (effort `high`).
Live probing of `deepseek-v4-flash-vision-exp` confirmed:

- With thinking on, `reasoning_content` is returned and `reasoning_tokens` are
  billed inside `completion_tokens` — a trivial describe call produced **246
  reasoning tokens vs 17 answer tokens** (~15x output inflation).
- While thinking is on, `temperature`/`top_p` are silently ignored.
- `{"thinking": {"type": "disabled"}}` is accepted; temperature then works, and
  `response_format=json_object` coexists with thinking either way.

Therefore every preset defaults to `thinking="disabled"`: preset temperatures
(0.0–0.3) become meaningful, structured extraction is deterministic and cheap,
and reasoning-heavy workloads can opt back in via `options.thinking="enabled"`.
The cache key includes thinking mode.

## Model pinning

`deepseek-v4-flash-vision-exp` is experimental. The model id is a constant resolved from
`DEEPSEEK_MULTIMODAL_MODEL` → default, so a rename or a GA release is a one-line/env-var
change. The eval harness is provider-agnostic by construction for the same reason.

## Open risks

1. **Experimental model drift** — vision-exp is explicitly experimental; behavior/limits may
   change. Mitigation: all provider constants centralized; integration tests run only with a
   key present; `deepseek-api.md` reference documents the verified contract + date.
2. **JSON-mode empty content** — documented API quirk; handled as retryable, but at low
   `max_tokens` truncation can still occur. Mitigation: default `max_output_tokens=8192`,
   16384 for extraction.
3. **Small-machine constraints** — PDF rendering of very large documents and 100+ frame
   videos are memory-heavy; defaults (150 dpi, 128 frames) are sized for this box.
