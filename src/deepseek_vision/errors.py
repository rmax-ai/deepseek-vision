"""Exception hierarchy for deepseek_vision.

All exceptions inherit from :class:`DeepSeekVisionError`. Each subclass has a
clean ``__str__`` implementation so failures surface as actionable messages.
"""

from __future__ import annotations

from typing import Any


class DeepSeekVisionError(Exception):
    """Base class for every error raised by deepseek_vision."""


class ConfigurationError(DeepSeekVisionError):
    """Invalid configuration, preset name, or caller-supplied options."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"ConfigurationError: {self.message}"


class MediaError(DeepSeekVisionError):
    """Unreadable, missing, or unsupported media input."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"MediaError: {self.message}"


class PreprocessingError(DeepSeekVisionError):
    """Failure in deterministic preprocessing (ffmpeg/ffprobe/PyMuPDF)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"PreprocessingError: {self.message}"


class ProviderAPIError(DeepSeekVisionError):
    """A non-retryable (or exhausted) failure from the provider API.

    Attributes:
        status_code: HTTP status code, or None when no response was received.
        message: Human readable error message.
        response_body: Raw response body snippet, if available.
    """

    def __init__(
        self,
        status_code: int | None,
        message: str,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response_body = response_body

    def __str__(self) -> str:
        base = f"ProviderAPIError(status={self.status_code}): {self.message}"
        if self.response_body:
            base += f"\nResponse body: {self.response_body}"
        return base


class RateLimitError(ProviderAPIError):
    """HTTP 429 after retries were exhausted (or immediately raised)."""

    def __init__(
        self,
        status_code: int,
        message: str,
        response_body: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(status_code, message, response_body)
        self.retry_after = retry_after

    def __str__(self) -> str:
        base = f"RateLimitError(status=429): {self.message}"
        if self.retry_after is not None:
            base += f"\nRetry-After: {self.retry_after:.1f}s"
        if self.response_body:
            base += f"\nResponse body: {self.response_body}"
        return base


class ProviderTimeoutError(ProviderAPIError):
    """The provider request timed out and retries were exhausted."""

    def __init__(self, message: str, response_body: str | None = None) -> None:
        super().__init__(None, message, response_body)

    def __str__(self) -> str:
        base = f"ProviderTimeoutError: {self.message}"
        if self.response_body:
            base += f"\nResponse body: {self.response_body}"
        return base


class ResponseValidationError(DeepSeekVisionError):
    """Response failed JSON parsing or Pydantic validation.

    Attributes:
        message: Human readable error description.
        payloads: The attempted response payloads that failed validation.
    """

    def __init__(self, message: str, payloads: list[Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.payloads = payloads or []

    def __str__(self) -> str:
        base = f"ResponseValidationError: {self.message}"
        if self.payloads:
            base += f"\nAttempted payloads: {len(self.payloads)}"
        return base


class SynthesisError(DeepSeekVisionError):
    """Failure during hierarchical (multi-batch) synthesis."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"SynthesisError: {self.message}"


class CacheError(DeepSeekVisionError):
    """Failure reading or writing the content-addressed cache."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"CacheError: {self.message}"
