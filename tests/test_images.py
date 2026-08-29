"""Tests for media/image loading, format detection, and data URLs."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from deepseek_vision.errors import MediaError
from deepseek_vision.media.images import (
    detect_format,
    frame_to_data_url,
    load_collection,
    load_image,
)
from deepseek_vision.models import ImageCollectionInput, ImageInput


def test_detect_format_png(sample_png_bytes: bytes) -> None:
    assert detect_format(sample_png_bytes) == "png"


def test_detect_format_jpeg(sample_jpeg_bytes: bytes) -> None:
    assert detect_format(sample_jpeg_bytes) == "jpeg"


def test_detect_format_webp(sample_webp_bytes: bytes) -> None:
    assert detect_format(sample_webp_bytes) == "webp"


def test_detect_format_invalid(invalid_image_bytes: bytes) -> None:
    with pytest.raises(MediaError):
        detect_format(invalid_image_bytes)


async def test_exif_transpose_applied(tmp_path) -> None:
    img = Image.new("RGB", (10, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation 6: rotate 90 CW for display
    path = tmp_path / "oriented.jpg"
    img.save(path, exif=exif)
    frame = await load_image(ImageInput(path=str(path)))
    assert frame.metadata["width"] == 20
    assert frame.metadata["height"] == 10


async def test_oversized_image_downscaled(tmp_path) -> None:
    img = Image.new("RGB", (5000, 1000), (0, 128, 0))
    path = tmp_path / "big.png"
    img.save(path)
    frame = await load_image(ImageInput(path=str(path)), downscale_above=1024)
    assert frame.metadata["width"] <= 1024
    assert frame.metadata["height"] <= 1024
    assert frame.metadata["width"] == 1024  # aspect ratio preserved
    assert frame.metadata["original_size_bytes"] > 0


async def test_collection_preserves_order(tmp_path) -> None:
    paths: list[str] = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (8, 8), color)
        p = tmp_path / f"img{i}.png"
        img.save(p)
        paths.append(str(p))
    spec = ImageCollectionInput(images=[ImageInput(path=p) for p in paths])
    frames = await load_collection(spec)
    assert len(frames) == 3
    assert [f.source for f in frames] == paths


async def test_frame_to_data_url_mime_per_format(
    sample_png_bytes: bytes,
    sample_jpeg_bytes: bytes,
    sample_webp_bytes: bytes,
) -> None:
    cases = [
        (sample_png_bytes, "data:image/png;base64,"),
        (sample_jpeg_bytes, "data:image/jpeg;base64,"),
        (sample_webp_bytes, "data:image/webp;base64,"),
    ]
    for data, prefix in cases:
        frame = await load_image(ImageInput(data=data))
        part = frame_to_data_url(frame, detail="high")
        assert part["url"].startswith(prefix), frame.metadata["format"]
        assert part["detail"] == "high"


def test_image_input_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        ImageInput()  # none of path/url/data
    with pytest.raises(ValueError):
        ImageInput(path="a.png", url="https://example.com/a.png")


def test_image_bytes_roundtrip(sample_png_bytes: bytes) -> None:
    buf = io.BytesIO(sample_png_bytes)
    img = Image.open(buf)
    assert img.size == (320, 240)
