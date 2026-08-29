"""Provider contract for the evaluation harness.

Future adapters (GeminiMultimodalClient, OpenAIMultimodalClient,
AnthropicMultimodalClient) implement this same protocol so eval cases and
media processing never change per provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ProviderResponse:
    """Normalized response from any multimodal provider adapter."""

    text: str
    data: Any | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    requests: int = 1


class MultimodalProvider(Protocol):
    """Async provider contract implemented by every adapter."""

    name: str

    async def analyze_frames(
        self,
        frames: list,
        prompt: str,
        *,
        system: str | None = None,
        output_schema: type | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 8192,
    ) -> ProviderResponse:
        """Analyze a list of MediaFrame-like objects and return a response.

        ``data`` carries the parsed schema instance (or dict/None) and
        ``usage`` carries the provider token accounting dict, when available.
        """
        ...
