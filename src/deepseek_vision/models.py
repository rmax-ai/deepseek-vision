"""Pydantic v2 models for the deepseek_vision public surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MediaFrame(BaseModel):
    """A single preprocessed, re-encoded image ready for the API.

    ``image`` bytes are ALWAYS one of the supported re-encoded formats
    (JPEG/PNG/GIF/WebP), so ``metadata["format"]`` can be trusted for mime
    types when building data URLs.
    """

    image: bytes
    timestamp: float | None = None
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Evidence(BaseModel):
    """A provenance-carrying observation anchored to a media source."""

    source: str
    page: int | None = None
    timestamp_seconds: float | None = None
    batch: int | None = None
    observation: str
    confidence: float | None = None


class Observation(BaseModel):
    """A top-level observation surfaced in :class:`AnalysisResult`."""

    text: str
    source_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class ImageInput(BaseModel):
    """A single image: exactly one of ``path``, ``url``, or ``data``."""

    kind: Literal["image"] = "image"
    path: str | None = None
    url: str | None = None
    data: bytes | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImageInput:
        provided = sum(
            1 for v in (self.path, self.url, self.data) if v is not None
        )
        if provided != 1:
            raise ValueError(
                "ImageInput requires exactly one of path, url, or data "
                f"(got {provided})"
            )
        return self


class ImageCollectionInput(BaseModel):
    """An ordered collection of images."""

    kind: Literal["collection"] = "collection"
    images: list[ImageInput]
    name: str | None = None


class VideoOptions(BaseModel):
    """Options controlling deterministic video frame extraction."""

    sampling: Literal["uniform", "scene_change", "keyframes", "adaptive"] = (
        "adaptive"
    )
    fps: float = 0.5
    max_frames: int = 128
    scene_threshold: float = 0.3
    start_seconds: float | None = None
    end_seconds: float | None = None
    window_seconds: float | None = None


class VideoInput(BaseModel):
    """A local video file; frames are extracted with ffmpeg."""

    kind: Literal["video"] = "video"
    path: str
    options: VideoOptions = Field(default_factory=VideoOptions)


class DocumentOptions(BaseModel):
    """Options controlling deterministic PDF page rendering."""

    pages: str | None = None
    dpi: int = 150
    max_pages_per_batch: int | None = None


class DocumentInput(BaseModel):
    """A local PDF file; pages are rendered with PyMuPDF."""

    kind: Literal["document"] = "document"
    path: str
    options: DocumentOptions = Field(default_factory=DocumentOptions)


MediaInput = Annotated[
    ImageInput | ImageCollectionInput | VideoInput | DocumentInput,
    Field(discriminator="kind"),
]


class AnalyzeOptions(BaseModel):
    """Runtime options resolved from env vars when fields are None."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_images_per_request: int = 16
    max_concurrency: int = 4
    max_retries: int = 3
    timeout_seconds: float = 120.0
    retry_base_delay: float = 1.0
    temperature: float | None = None
    max_output_tokens: int = 8192
    detail: str = "original"
    use_cache: bool = True
    cache_dir: str | None = None
    image_token_estimate: int = 384
    context_window: int = 1_000_000
    context_safety_factor: float = 0.5
    reserved_output_tokens: int = 16384
    video: VideoOptions = Field(default_factory=VideoOptions)
    document: DocumentOptions = Field(default_factory=DocumentOptions)
    user_id: str | None = None
    event_handler: object | None = Field(default=None, exclude=True)


class UsageSummary(BaseModel):
    """Aggregated token/cost/latency accounting for a run."""

    requests: int = 0
    images_processed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_seconds: float = 0.0
    retries: int = 0
    from_cache: int = 0


class MediaStats(BaseModel):
    """Statistics describing what media was processed."""

    sources: list[str] = Field(default_factory=list)
    frames: int = 0
    pages: int = 0
    batches: int = 0
    batch_sizes: list[int] = Field(default_factory=list)
    duration_seconds: float | None = None


class AnalysisResult(BaseModel):
    """The single public result of an analysis run."""

    task: str
    data: Any | None = None
    observations: list[Observation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    synthesis: str | None = None
    usage: UsageSummary
    media: MediaStats
    errors: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompareResult(BaseModel):
    """Structured output for the ``compare`` preset."""

    common_elements: list[str] = Field(default_factory=list)
    differences: list[dict[str, Any]] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)


class VerificationCheck(BaseModel):
    """One claim/status/evidence tuple for the ``verification`` preset."""

    claim: str
    status: Literal["pass", "fail", "inconclusive"]
    evidence: str = ""
    confidence: float | None = None


class VerificationResult(BaseModel):
    """Structured output for the ``verification`` preset."""

    checks: list[VerificationCheck] = Field(default_factory=list)
