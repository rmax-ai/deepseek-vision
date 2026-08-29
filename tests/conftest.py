"""Shared pytest fixtures. No live API calls happen here."""

from __future__ import annotations

import io
import subprocess

import httpx
import pytest
import respx
from PIL import Image, ImageDraw

from deepseek_vision.client import DeepSeekMultimodalClient
from deepseek_vision.events import EventEmitter
from deepseek_vision.usage import UsageTracker

BASE_URL = "https://api.deepseek.com"


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg; skip tests when ffmpeg is unavailable."""
    try:
        check = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    if check.returncode != 0:
        pytest.skip("ffmpeg not available")
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed: {stderr}")


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    """320x240 RGBA gradient with shapes."""
    img = Image.new("RGBA", (320, 240), (10, 10, 10, 255))
    draw = ImageDraw.Draw(img)
    for y in range(240):
        for x in range(320):
            img.putpixel(
                (x, y), (x % 256, y % 256, (x + y) % 256, 255)
            )
    draw.rectangle([20, 20, 100, 100], fill=(255, 0, 0, 255))
    draw.ellipse([140, 60, 220, 180], fill=(0, 255, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_jpeg_bytes(sample_png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(sample_png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_webp_bytes(sample_png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(sample_png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture(scope="session")
def invalid_image_bytes() -> bytes:
    return b"not an image"


@pytest.fixture
def sample_video(tmp_path):
    """4s 320x240 10fps two-scene video (black then red)."""
    out = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "10",
        str(out),
    ]
    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        fallback = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
            "-c:v", "mpeg4", "-q:v", "3", "-r", "10",
            str(out),
        ]
        run_ffmpeg(fallback)
    return str(out)


@pytest.fixture
def sample_pdf(tmp_path):
    """3-page PDF with page numbers rendered as text."""
    import fitz  # pymupdf, lazy import

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    out = tmp_path / "sample.pdf"
    doc.save(str(out))
    doc.close()
    return str(out)


def _json_response(content: str = '{"summary": "ok"}', **usage: int) -> httpx.Response:
    payload = {"prompt_tokens": 100, "completion_tokens": 50}
    payload.update(usage)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": payload,
        },
    )


@pytest.fixture
def mock_api():
    """Respx mock returning a fixed 200 JSON response."""
    with respx.mock(base_url=BASE_URL) as router:
        router.post("/chat/completions").mock(
            return_value=_json_response()
        )
        yield router


@pytest.fixture
def mock_api_json_sequence():
    """Install a route returning responses in order (last repeats).

    Entries may be ``httpx.Response`` objects or exception instances.
    """
    routers = []

    def _install(responses: list):
        router = respx.mock(base_url=BASE_URL)
        iterator = iter(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            try:
                item = next(iterator)
            except StopIteration:
                item = responses[-1]
            if isinstance(item, Exception):
                raise item
            return item

        router.post("/chat/completions").mock(side_effect=handler)
        router.start()
        routers.append(router)
        return router

    yield _install
    for router in routers:
        router.stop()


@pytest.fixture
def collect_events():
    """EventEmitter that records every emitted event."""

    class RecordingEmitter(EventEmitter):
        def __init__(self) -> None:
            super().__init__()
            self.recorded: list[tuple[str, dict]] = []

        async def emit(self, name: str, **data) -> None:
            self.recorded.append((name, data))
            await super().emit(name, **data)

    return RecordingEmitter()


@pytest.fixture
def client(collect_events):
    """DeepSeekMultimodalClient with placeholder key and recording emitter."""
    return DeepSeekMultimodalClient(
        api_key="sk-test-placeholder",
        event_emitter=collect_events,
        usage_tracker=UsageTracker(),
    )
