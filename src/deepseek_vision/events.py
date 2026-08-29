"""Event emission for observability across the analysis pipeline."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# Canonical event names used across the pipeline.
MEDIA_LOADED = "media_loaded"
MEDIA_PREPROCESSED = "media_preprocessed"
BATCH_CREATED = "batch_created"
REQUEST_STARTED = "request_started"
REQUEST_COMPLETED = "request_completed"
REQUEST_FAILED = "request_failed"
SYNTHESIS_STARTED = "synthesis_started"
SYNTHESIS_COMPLETED = "synthesis_completed"

EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass(frozen=True)
class Event:
    """An immutable, timestamped event payload."""

    name: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


async def _noop_handler(event: Event) -> None:
    """Default handler: swallow everything."""


class EventEmitter:
    """Fan-out emitter that never breaks the pipeline.

    Holds an optional async callable handler. Errors raised by the handler
    are caught and discarded.
    """

    def __init__(self, handler: EventHandler | None = None) -> None:
        self._handler: EventHandler = handler or _noop_handler

    @property
    def handler(self) -> EventHandler:
        return self._handler

    async def emit(self, name: str, **data: Any) -> None:
        """Emit an event, swallowing any handler errors."""
        try:
            event = Event(name=name, ts=time.time(), data=dict(data))
            await self._handler(event)
        except Exception:  # noqa: BLE001, S110 - handler errors never break the pipeline
            pass
