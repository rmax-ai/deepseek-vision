"""Content-addressed file cache for analysis results.

Pure, unit-testable functions first (``key``, ``frame_sha``,
``schema_digest``), with a thin synchronous IO class (:class:`Cache`) on top.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import MediaFrame

_CACHE_KEYS = (
    "model",
    "task",
    "instructions",
    "schema_digest",
    "detail",
    "temperature",
    "max_output_tokens",
    "thinking",
)


def default_cache_dir() -> str:
    """Resolve the default cache directory from env or home."""
    return os.environ.get("DEEPSEEK_CACHE_DIR") or str(
        Path.home() / ".cache" / "deepseek-vision" / "cache"
    )


def frame_sha(frame: MediaFrame) -> str:
    """Content hash of a frame's bytes and timestamp."""
    digest = hashlib.sha256()
    digest.update(frame.image)
    digest.update(repr(frame.timestamp).encode("utf-8"))
    return digest.hexdigest()


def schema_digest(schema: type | None) -> str:
    """Stable digest of a Pydantic schema, or ``"none"``."""
    if schema is None:
        return "none"
    payload = json.dumps(
        schema.model_json_schema(mode="validation"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def key(inputs: dict) -> str:
    """Content-addressed key derived from canonical input fields.

    ``inputs`` must contain ``media_hashes`` (list of frame sha256 strings;
    order does not matter) plus any of: model, task, instructions,
    schema_digest, detail, temperature, max_output_tokens.
    """
    canonical: dict[str, Any] = {}
    for field in _CACHE_KEYS:
        if field in inputs:
            canonical[field] = inputs[field]
    media_hashes = inputs.get("media_hashes", [])
    canonical["media_hashes"] = sorted(media_hashes)
    payload = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    """Thin content-addressed JSON file cache.

    Synchronous IO by design; callers inside async code should wrap calls
    with ``asyncio.to_thread``.
    """

    def __init__(self, directory: str | None = None) -> None:
        self._dir = Path(directory or default_cache_dir())

    @property
    def dir(self) -> str:
        return str(self._dir)

    def _path_for(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Read a cached entry; return None on any read/parse failure."""
        path = self._path_for(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                value = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(value, dict):
            return None
        return value

    def put(self, key: str, value: dict) -> None:
        """Write an entry atomically (tmp file + os.replace)."""
        directory = self._path_for(key).parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(directory), prefix=".tmp-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh)
            os.replace(tmp_path, self._path_for(key))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def match_media(self, frames: list[MediaFrame]) -> list[str]:
        """Return the content hash of every frame, in order."""
        return [frame_sha(frame) for frame in frames]
