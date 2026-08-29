"""Tests for the content-addressed cache."""

from __future__ import annotations

from pydantic import BaseModel

from deepseek_vision.cache import Cache, frame_sha, key, schema_digest
from deepseek_vision.models import MediaFrame


class _DummySchema(BaseModel):
    summary: str


def _base_inputs(**overrides) -> dict:
    inputs = {
        "media_hashes": ["a", "b", "c"],
        "model": "deepseek-v4-flash-vision-exp",
        "task": "describe",
        "instructions": "look carefully",
        "schema_digest": "none",
        "detail": "original",
        "temperature": 0.3,
        "max_output_tokens": 8192,
    }
    inputs.update(overrides)
    return inputs


def test_key_identical_for_identical_inputs() -> None:
    assert key(_base_inputs()) == key(_base_inputs())


def test_key_differs_on_instructions_change() -> None:
    assert key(_base_inputs()) != key(_base_inputs(instructions="other"))


def test_key_differs_on_model_change() -> None:
    assert key(_base_inputs()) != key(_base_inputs(model="other-model"))


def test_key_differs_on_schema_digest_change() -> None:
    assert key(_base_inputs()) != key(
        _base_inputs(schema_digest=schema_digest(_DummySchema))
    )


def test_key_differs_on_temperature_change() -> None:
    assert key(_base_inputs()) != key(_base_inputs(temperature=0.9))


def test_key_media_hashes_order_independent() -> None:
    a = key(_base_inputs())
    b = key(_base_inputs(media_hashes=["c", "a", "b"]))
    assert a == b


def test_frame_sha_stable() -> None:
    frame = MediaFrame(image=b"image-bytes", timestamp=1.5, source="s.mp4")
    assert frame_sha(frame) == frame_sha(frame)
    other = MediaFrame(image=b"image-bytes", timestamp=1.6, source="s.mp4")
    assert frame_sha(frame) != frame_sha(other)


def test_get_put_roundtrip(tmp_path) -> None:
    cache = Cache(str(tmp_path))
    k = key(_base_inputs())
    value = {"data": {"summary": "ok"}, "text": '{"summary": "ok"}'}
    cache.put(k, value)
    assert cache.get(k) == value
    assert cache.get(k) is not None


def test_get_missing_returns_none(tmp_path) -> None:
    cache = Cache(str(tmp_path))
    assert cache.get("deadbeef") is None


def test_get_corrupted_file_returns_none(tmp_path) -> None:
    cache = Cache(str(tmp_path))
    k = "abcdef0123456789"
    path = tmp_path / k[:2] / f"{k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json!!!")
    assert cache.get(k) is None


def test_schema_digest_none() -> None:
    assert schema_digest(None) == "none"


def test_schema_digest_stable() -> None:
    assert schema_digest(_DummySchema) == schema_digest(_DummySchema)
    assert schema_digest(_DummySchema) != "none"


def test_match_media_returns_per_frame_shas() -> None:
    cache = Cache("/tmp/irrelevant")
    frames = [
        MediaFrame(image=b"x", source="a"),
        MediaFrame(image=b"y", source="b"),
    ]
    assert cache.match_media(frames) == [frame_sha(f) for f in frames]


def test_cache_dir_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_CACHE_DIR", str(tmp_path / "env-cache"))
    cache = Cache()
    assert cache.dir == str(tmp_path / "env-cache")


def test_default_cache_dir_home(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPSEEK_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    cache = Cache()
    assert cache.dir == str(tmp_path / ".cache" / "deepseek-vision" / "cache")


def test_key_differs_on_thinking() -> None:
    base = _base_inputs()
    assert key({**base, "thinking": "disabled"}) != key(
        {**base, "thinking": "enabled"}
    )
