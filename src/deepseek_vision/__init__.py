"""deepseek-vision: one multimodal analysis primitive on DeepSeek V4 Flash.

Lazy re-exports: heavy submodules (client, media, presets) are only imported
on first attribute access via PEP 562 ``__getattr__``.
"""

from __future__ import annotations

from .errors import (
    CacheError,
    ConfigurationError,
    DeepSeekVisionError,
    MediaError,
    PreprocessingError,
    ProviderAPIError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseValidationError,
    SynthesisError,
)
from .models import (
    AnalysisResult,
    AnalyzeOptions,
    CompareResult,
    DocumentInput,
    DocumentOptions,
    Evidence,
    ImageCollectionInput,
    ImageInput,
    MediaFrame,
    Observation,
    UsageSummary,
    VerificationCheck,
    VerificationResult,
    VideoInput,
    VideoOptions,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "AnalyzeOptions",
    "CacheError",
    "CompareResult",
    "ConfigurationError",
    "DeepSeekMultimodalClient",
    "DeepSeekVisionError",
    "DocumentInput",
    "DocumentOptions",
    "Evidence",
    "ImageCollectionInput",
    "ImageInput",
    "MediaError",
    "MediaFrame",
    "Observation",
    "PreprocessingError",
    "ProviderAPIError",
    "ProviderTimeoutError",
    "RateLimitError",
    "ResponseValidationError",
    "SynthesisError",
    "UsageSummary",
    "VerificationCheck",
    "VerificationResult",
    "VideoInput",
    "VideoOptions",
    "__version__",
    "analyze_media",
    "analyze_media_sync",
]

_LAZY_IMPORTS = {
    "analyze_media": ("deepseek_vision.skill", "analyze_media"),
    "analyze_media_sync": ("deepseek_vision.skill", "analyze_media_sync"),
    "DeepSeekMultimodalClient": ("deepseek_vision.client", "DeepSeekMultimodalClient"),
}


def __getattr__(name: str):
    """PEP 562 lazy import for the heavy public entry points."""
    spec = _LAZY_IMPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = spec
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
