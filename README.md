# deepseek-vision

Provider-independent multimodal media analysis built on DeepSeek V4 Flash
Vision Exp (`deepseek-v4-flash-vision-exp`). One async primitive,
`analyze_media()`, accepts images, image collections, videos, and PDFs,
preprocesses them deterministically (decode, frame extraction, page
rendering, batching), sends the result to the DeepSeek API, and returns a
structured `AnalysisResult` carrying observations, provenance-anchored
evidence, token/cost accounting, and optional hierarchical synthesis for
long documents and videos. The evaluation harness (`evals/`) exercises the
same media processing against any provider implementing a small async
protocol, so cases never change per provider.

## Quick start

```bash
uv sync
export DEEPSEEK_API_KEY="sk-..."
# One CLI command
uv run deepseek-vision analyze screenshot.png --task describe --usage
# One Python snippet
uv run python -c "
import asyncio
from deepseek_vision import analyze_media

result = asyncio.run(analyze_media('screenshot.png', task='describe'))
print(result.synthesis)
print(result.usage)
"
```

`DEEPSEEK_API_KEY` is read from the environment at runtime; it is never
stored or logged. `DEEPSEEK_BASE_URL` and `DEEPSEEK_MULTIMODAL_MODEL` are
optional overrides.

## Capabilities

- **Image understanding** — precise descriptions, object and layout analysis.
- **Structured extraction** — extract any caller-supplied Pydantic schema
  from images, documents, or screens (invoice/table schemas ship in the CLI).
- **Comparison** — structured diffs between two images, including a
  reference-vs-implementation visual regression mode.
- **Documents** — multi-page PDF analysis with page-cited observations and
  hierarchical synthesis when pages span multiple requests.
- **Video** — deterministic frame extraction (uniform, scene-change,
  keyframes, adaptive) with timestamped evidence and hierarchical synthesis.
- **Verification** — verify visual claims and get per-claim pass/fail
  status with evidence and confidence.
- **Software-engineering analysis** — UI bug triage, screenshot/terminal
  analysis, architecture diagram analysis, visual regression.

## Architecture

Deterministic work (decode, frame extraction, PDF rendering, batching,
validation, caching) never depends on the LLM. The LLM sits only at the
semantic boundary.

```
media ──► deterministic preprocessing ──► batching ──► DeepSeek API
             (decode, frames, pages)                     │
                                                         ▼
                                            structured observations
                                                         │
                                                         ▼
                                       optional hierarchical synthesis
                                                         │
                                                         ▼
                                                    AnalysisResult
```

| Module | Responsibility |
| --- | --- |
| `skill.py` | `analyze_media()` / `analyze_media_sync()`: the single public orchestration primitive |
| `cli.py` | argparse CLI (`deepseek-vision`) with `analyze`, `compare`, `video`, `document` subcommands |
| `client.py` | DeepSeek API transport only (httpx); retries, rate-limit backoff, JSON-schema validation |
| `media/images.py` | image decode, EXIF transpose, downscale, re-encode |
| `media/video.py` | ffmpeg/ffprobe frame extraction (uniform, scene-change, keyframes, adaptive) |
| `media/pdf.py` | PyMuPDF page rendering with page-spec parsing |
| `media/batching.py` | context-safe batch sizing and grouping |
| `presets/` | 15 task presets: prompt/config data only, no logic |
| `synthesis.py` | hierarchical multi-batch + global synthesis |
| `models.py` | Pydantic v2 public surface (inputs, options, results) |
| `errors.py` | exception hierarchy rooted at `DeepSeekVisionError` |
| `events.py` | observable pipeline events (media loaded, batches, requests) |
| `usage.py` | token/cost/latency accounting with peak/off-peak pricing |
| `cache.py` | content-addressed result cache |
| `prompts.py` | prompt construction and token estimation |

## Presets

| Name | Purpose | Schema | Hierarchical |
| --- | --- | --- | --- |
| `describe` | precise description of visible content | none | no |
| `question_answer` | answer questions grounded in media | none | no |
| `extract` | extract caller-supplied structured data | caller-provided | no |
| `compare` | structured image comparison | `CompareResult` | no |
| `diagram_analysis` | architecture/technical diagram analysis | `DiagramAnalysis` | no |
| `ui_analysis` | UI state inspection | `UiAnalysis` | no |
| `document_analysis` | multi-page document analysis | `DocumentObservations` -> `DocumentAnalysis` | yes |
| `page_extraction` | extract structured data from pages | caller-provided | no |
| `visual_regression` | reference vs implementation UI diff | `VisualRegressionResult` | no |
| `bug_report` | UI bug triage from screenshots | `BugReport` | no |
| `verification` | verify visual claims | `VerificationResult` | no |
| `screenshot_analysis` | terminal/error screenshot inspection | `ScreenAnalysis` | no |
| `video_summary` | timestamped video summarization | `WindowObservations` -> `VideoSummary` | yes |
| `temporal_analysis` | temporal structure extraction | `WindowObservations` -> `TimelineAnalysis` | yes |
| `movement_analysis` | biomechanical movement analysis | `WindowObservations` -> `MovementAnalysis` | yes |

## CLI reference

Global flags (every subcommand): `--json`, `--usage`, `--verbose`,
`--no-cache`, `--model`, `--timeout FLOAT`, `--concurrency INT`,
`--max-images-per-request INT`, `--temperature FLOAT`, `--max-tokens INT`.

Exit codes: `0` success, `1` any `DeepSeekVisionError` (message printed as
`error: ...` to stderr), `2` argparse usage errors.

### analyze

```bash
uv run deepseek-vision analyze screenshot.png --task describe
uv run deepseek-vision analyze a.png b.png c.png --task compare --json
uv run deepseek-vision analyze invoice.png --schema invoice --usage
uv run deepseek-vision analyze table.png --schema table --task extract
uv run deepseek-vision analyze photo.png --instructions "Focus on the foreground" --verbose
```

`--schema invoice|table` defines example output schemas and defaults the
task to `extract` when `--task` is omitted.

### compare

```bash
uv run deepseek-vision compare reference.png actual.png --task visual_regression
uv run deepseek-vision compare a.png b.png
```

### video

```bash
uv run deepseek-vision video clip.mp4 --task video_summary
uv run deepseek-vision video clip.mp4 --task movement_analysis --sampling adaptive --fps 0.5 --max-frames 48
uv run deepseek-vision video clip.mp4 --start 10 --end 60 --window-seconds 5
```

`--sampling` is one of `uniform|scene_change|keyframes|adaptive`.

### document

```bash
uv run deepseek-vision document report.pdf --task document_analysis
uv run deepseek-vision document report.pdf --pages "1-10" --dpi 150 --max-pages-per-batch 2
```

## Python API

```python
async def analyze_media(
    media: MediaLike,            # path, URL, MediaInput, or list of images
    *,
    task: str = "describe",      # any preset name
    instructions: str | None = None,
    output_schema: type[BaseModel] | None = None,
    options: AnalyzeOptions | dict | None = None,
) -> AnalysisResult
```

Input models: `ImageInput`, `ImageCollectionInput`, `VideoInput`,
`DocumentInput`. Options model: `AnalyzeOptions` (`VideoOptions`,
`DocumentOptions`, batching/token budgets, retries, concurrency, caching,
`event_handler`). Result model: `AnalysisResult` with `data`,
`observations`, `evidence`, `synthesis`, `usage` (`UsageSummary`), `media`
(`MediaStats`), and `errors`. `analyze_media_sync(...)` is the synchronous
wrapper.

## Provider contract (verified DeepSeek facts)

| Fact | Value |
| --- | --- |
| Model id | `deepseek-v4-flash-vision-exp` |
| Context window | 1,000,000 tokens (configurable via `AnalyzeOptions`) |
| Max images per request | 600 (configurable via `max_images_per_request`) |
| Max inline image bytes per request | 64 MiB |
| Token accounting | `prompt_tokens`, `completion_tokens`, `prompt_cache_hit_tokens` per response |
| Input pricing (per 1M tokens) | cache hit: $0.007 off-peak / $0.014 peak; cache miss: $0.22 off-peak / $0.44 peak |
| Output pricing (per 1M tokens) | $0.66 off-peak / $1.32 peak |
| Peak hours (UTC) | Mon-Fri 01:00-04:00 and 06:00-10:00 |
| Native video/document input | No: videos and PDFs are deterministically preprocessed into image frames |

Pricing constants live in `usage.py` and are configurable.

## Caching

Results are cached content-addressed (media frame hashes + model + task +
prompt + schema + options) under `~/.cache/deepseek-vision/cache` by
default. Disable with `--no-cache`, `options={"use_cache": False}`, or
`DEEPSEEK_CACHE_DIR` to relocate.

## Telemetry and usage

With `--usage` (CLI) or `result.usage` (Python) you get an accounting block:

```json
{
  "requests": 3,
  "images_processed": 5,
  "input_tokens": 4210,
  "output_tokens": 890,
  "cache_hit_tokens": 0,
  "cache_miss_tokens": 4210,
  "estimated_cost_usd": 0.0015,
  "latency_seconds": 12.4,
  "retries": 0,
  "from_cache": 0
}
```

## Evaluation harness

```bash
uv run python -m evals.run --help
uv run python -m evals.run --cases all --providers deepseek --concurrency 2
uv run python -m evals.run --cases invoice_case,compare_case --out ./eval-report
```

`evals/` generates deterministic synthetic fixtures (PIL/PyMuPDF/ffmpeg)
into `./.eval-media`, runs each case against each provider through the
`MultimodalProvider` protocol, evaluates deterministically
(`structured_exact`, `verification_exact`, `llm_judge`), and writes
`report.md` + `report.json`. `DEEPSEEK_API_KEY` is required for the
`deepseek` provider. Exit code is `0` when cases fail (the report carries
results) and `2` on configuration errors.

## Testing

```bash
uv run pytest -q                    # unit tests (no network, no API key)
uv export DEEPSEEK_API_KEY=... && uv run pytest -q tests/integration
uvx ruff check src/ tests/ evals/ examples/
```

Unit tests mock the API with respx. Integration tests (`tests/integration/`)
skip unless `DEEPSEEK_API_KEY` is set and assert structural properties only;
LLM outputs are never asserted semantically.

## Repository layout

```
src/deepseek_vision/    library package (CLI lives here too)
tests/                  unit tests (respx-mocked, no live calls)
tests/integration/      live API tests (skipped without DEEPSEEK_API_KEY)
evals/                  provider-independent evaluation harness
examples/               runnable end-to-end examples + fixture generator
```

## Design decisions

See `DESIGN.md` for the design decisions and trade-offs behind this layout.
Notable choices: deterministic preprocessing never depends on the LLM; all
provider limits and prices live as constants in `usage.py` / `models.py`;
bounded concurrency via `asyncio.Semaphore`; unit tests never touch the
network.

## License

MIT — see `LICENSE`.
