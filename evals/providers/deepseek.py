"""DeepSeek adapter implementing the evals MultimodalProvider protocol."""

from __future__ import annotations

import os
import time
from typing import Self

from deepseek_vision.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekMultimodalClient,
)

from ..protocol import ProviderResponse


class DeepSeekProvider:
    """Wraps :class:`DeepSeekMultimodalClient` behind the evals protocol.

    Model and base URL resolve from environment exactly like the package
    does (``DEEPSEEK_MULTIMODAL_MODEL`` / ``DEEPSEEK_BASE_URL``) unless
    overridden in the constructor.
    """

    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model or os.environ.get(
            "DEEPSEEK_MULTIMODAL_MODEL", DEFAULT_MODEL
        )
        self._base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", DEFAULT_BASE_URL
        )
        self._client = DeepSeekMultimodalClient(
            api_key=api_key,
            base_url=self._base_url,
            model=self._model,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

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
        """Analyze frames via the DeepSeek client.

        Media frames are converted with ``media.images.frame_to_data_url``
        by the client itself, so raw MediaFrame objects are passed through.
        """
        start = time.monotonic()
        result = await self._client.analyze(
            frames,
            prompt=prompt,
            system=system,
            output_schema=output_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        elapsed = time.monotonic() - start
        return ProviderResponse(
            text=result["text"],
            data=result["data"],
            usage=result.get("usage") or {},
            latency_s=elapsed,
            requests=1,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
