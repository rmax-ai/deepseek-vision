"""DeepSeek API transport only.

This module performs HTTP requests against the DeepSeek chat completions
endpoint. It does not import ``media/`` or ``presets/`` modules (except the
pure ``frame_to_data_url`` helper) and never processes media itself.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Self

import httpx
from pydantic import BaseModel, ValidationError

from .errors import (
    ProviderAPIError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseValidationError,
)
from .events import (
    REQUEST_COMPLETED,
    REQUEST_FAILED,
    REQUEST_STARTED,
    EventEmitter,
)
from .media.images import frame_to_data_url
from .models import MediaFrame
from .usage import UsageTracker

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class _RetryableFailure(Exception):
    """Internal signal: the attempt failed but may be retried."""

    def __init__(
        self,
        status: int | None,
        retry_after: float | None = None,
        message: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.message = message


class _ValidationFailed(Exception):
    """Internal signal: valid JSON but schema validation failed."""

    def __init__(self, content: str, errors: list[dict]) -> None:
        super().__init__("schema validation failed")
        self.content = content
        self.errors = errors


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences if present."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class DeepSeekMultimodalClient:
    """Async HTTP client for the DeepSeek multimodal chat completions API.

    The API key is used only to build the Authorization header; it is never
    stored as an attribute and never logged.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout_seconds: float = 120.0,
        retry_base_delay: float = 1.0,
        max_concurrency: int = 4,
        event_emitter: EventEmitter | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=30.0, read=timeout_seconds, write=60.0, pool=30.0
            ),
        )
        self._base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.retry_base_delay = retry_base_delay
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.event_emitter = event_emitter or EventEmitter()
        self.usage_tracker = usage_tracker or UsageTracker()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def analyze(
        self,
        frames: list[MediaFrame],
        *,
        prompt: str,
        system: str | None = None,
        output_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 8192,
        detail: str = "original",
        user_id: str | None = None,
    ) -> dict:
        """Run one (possibly multi-image) analysis request.

        Returns ``{"data": ..., "text": ..., "usage": {...}}`` where ``data``
        is a parsed schema instance, a parsed dict, or raw text.
        """
        num_frames = len(frames)
        body = self._build_body(
            prompt, system, frames, output_schema, temperature,
            max_output_tokens, detail, user_id,
        )
        try:
            data, content, usage = await self._request_loop(
                body, output_schema, num_frames
            )
        except _ValidationFailed as vf:
            new_prompt = (
                prompt
                + "\n\nPREVIOUS RESPONSE FAILED VALIDATION:\n"
                + json.dumps(vf.errors)[:2000]
                + "\nRespond again with valid JSON."
            )
            body = self._build_body(
                new_prompt, system, frames, output_schema, temperature,
                max_output_tokens, detail, user_id,
            )
            data, content, usage = await self._request_loop(
                body, output_schema, num_frames, validation_failure_mode="raise"
            )
        return {"data": data, "text": content, "usage": usage}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Request orchestration
    # ------------------------------------------------------------------ #
    async def _request_loop(
        self,
        body: dict,
        output_schema: type[BaseModel] | None,
        num_frames: int,
        validation_failure_mode: str = "retry",
    ) -> tuple[Any, str, dict]:
        total_attempts = self.max_retries + 1
        retries = 0
        for attempt in range(total_attempts):
            await self.event_emitter.emit(REQUEST_STARTED, attempt=attempt)
            try:
                data, content, usage, latency = await self._attempt(
                    body, output_schema, validation_failure_mode
                )
            except _RetryableFailure as failure:
                retries += 1
                await self.event_emitter.emit(
                    REQUEST_FAILED,
                    attempt=attempt,
                    status=failure.status,
                    retryable=True,
                )
                if attempt == total_attempts - 1:
                    self._record(
                        {}, 0.0, retries=retries, images=num_frames
                    )
                    if failure.status == 429:
                        raise RateLimitError(
                            429,
                            f"rate limit exceeded: {failure.message}",
                            failure.message[:2000],
                            failure.retry_after,
                        )
                    if failure.status is None:
                        raise ProviderTimeoutError(
                            f"provider request timed out: {failure.message}"
                        )
                    if failure.status == 200:
                        raise ResponseValidationError(
                            failure.message, payloads=[]
                        )
                    raise ProviderAPIError(
                        failure.status,
                        f"provider error: {failure.message}",
                        failure.message[:2000],
                    )
                await self._backoff(attempt, failure.retry_after)
                continue
            except ProviderAPIError:
                raise
            except _ValidationFailed:
                raise

            self._record(usage, latency, retries=retries, images=num_frames)
            await self.event_emitter.emit(
                REQUEST_COMPLETED,
                latency_ms=round(latency * 1000, 2),
                tokens=usage.get("completion_tokens", 0),
                attempt=attempt,
            )
            return data, content, usage

        raise ProviderAPIError(None, "request loop exhausted unexpectedly")

    async def _attempt(
        self,
        body: dict,
        output_schema: type[BaseModel] | None,
        validation_failure_mode: str,
    ) -> tuple[Any, str, dict, float]:
        """Perform one POST and parse the result.

        Returns ``(data, content, usage, latency_s)`` on success. Raises
        ``_RetryableFailure`` for retryable failures, ``_ValidationFailed``
        for schema mismatches (when mode is ``"retry"``),
        ``ResponseValidationError`` (when mode is ``"raise"``), or
        ``ProviderAPIError`` for hard failures.
        """
        start = time.monotonic()
        try:
            async with self._semaphore:
                response = await self._client.post(
                    self._base_url + "/chat/completions", json=body
                )
            latency = time.monotonic() - start
        except httpx.TimeoutException as exc:
            raise _RetryableFailure(
                status=None, message=f"request timed out: {exc}"
            ) from exc

        status = response.status_code
        if status == 429:
            raise _RetryableFailure(
                status=429,
                retry_after=self._parse_retry_after(response),
                message=response.text[:500],
            )
        if status in _RETRYABLE_STATUSES:
            raise _RetryableFailure(
                status=status, message=response.text[:500]
            )
        if status != 200:
            raise ProviderAPIError(
                status,
                f"provider returned HTTP {status}: {response.text[:500]}",
                response.text[:2000],
            )

        try:
            r_json = response.json()
        except json.JSONDecodeError:
            raw = response.text
            if output_schema is None:
                return raw, raw, {}, latency
            raise _RetryableFailure(
                status=200, message="response body was not valid JSON"
            )

        content = r_json.get("choices", [{}])[0].get("message", {}).get("content")
        usage = r_json.get("usage", {})
        if content is None or content == "":
            if output_schema is None:
                return "", "", usage, latency
            raise _RetryableFailure(
                status=200, message="provider returned empty content"
            )

        cleaned = _strip_markdown_fences(content)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            if output_schema is None:
                return content, content, usage, latency
            raise _RetryableFailure(
                status=200, message="response content was not valid JSON"
            )

        if output_schema is not None:
            try:
                parsed = output_schema.model_validate(obj)
            except ValidationError as exc:
                if validation_failure_mode == "raise":
                    raise ResponseValidationError(
                        "response failed schema validation",
                        [content[:2000]],
                    )
                raise _ValidationFailed(content, exc.errors())
            return parsed, content, usage, latency
        return obj, content, usage, latency

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_body(
        self,
        prompt: str,
        system: str | None,
        frames: list[MediaFrame],
        output_schema: type[BaseModel] | None,
        temperature: float | None,
        max_output_tokens: int,
        detail: str,
        user_id: str | None,
    ) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": frame_to_data_url(frame, detail),
            }
            for frame in frames
        )
        messages.append({"role": "user", "content": content})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if output_schema is not None:
            body["response_format"] = {"type": "json_object"}
        if user_id:
            body["user_id"] = user_id
        return body

    def _record(
        self, usage: dict, latency_s: float, retries: int, images: int
    ) -> None:
        self.usage_tracker.add_request(
            usage, latency_s, retries=retries, images=images
        )

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            delta = (parsed - datetime.now(timezone.utc)).total_seconds()
            return max(delta, 0.0)
        except (TypeError, ValueError):
            return None

    async def _backoff(
        self, attempt: int, retry_after: float | None
    ) -> None:
        delay = self.retry_base_delay * (2**attempt) * random.uniform(0.8, 1.2)
        if retry_after is not None:
            delay = max(delay, retry_after)
        await asyncio.sleep(delay)
