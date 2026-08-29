"""Image loading, format detection, and re-encoding (Pillow)."""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from ..errors import MediaError
from ..models import ImageCollectionInput, ImageInput, MediaFrame

_DOWNSCALE_ALGORITHM = Image.Resampling.LANCZOS


def detect_format(data: bytes) -> str:
    """Detect image format from magic bytes (not extension).

    Returns one of ``"jpeg"``, ``"png"``, ``"gif"``, ``"webp"``.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise MediaError("unsupported image format")


def _read_image_bytes(src: ImageInput) -> tuple[bytes, str]:
    """Read raw bytes from path/url/data; returns (bytes, source_label)."""
    if src.path is not None:
        try:
            return Path(src.path).read_bytes(), src.path
        except OSError as exc:
            raise MediaError(f"cannot read image file {src.path}: {exc}")
    if src.url is not None:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(src.url)
        except httpx.HTTPError as exc:
            raise MediaError(f"failed to fetch image URL {src.url}: {exc}")
        if resp.status_code != 200:
            raise MediaError(
                f"failed to fetch image URL {src.url}: HTTP {resp.status_code}"
            )
        return resp.content, src.url
    if src.data is not None:
        return src.data, src.name or "<data>"
    raise MediaError("ImageInput has no path, url, or data")


def _process_image_bytes(data: bytes, downscale_above: int) -> tuple[bytes, dict]:
    """Decode, transpose, downscale, and re-encode image bytes."""
    fmt = detect_format(data)
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if fmt == "png":
            if "A" in img.getbands():
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        if max(img.size) > downscale_above:
            img.thumbnail((downscale_above, downscale_above), _DOWNSCALE_ALGORITHM)

        output = io.BytesIO()
        if fmt == "png" and img.mode == "RGBA":
            img.save(output, format="PNG")
            out_format = "png"
        elif fmt == "gif":
            img.save(output, format="GIF")
            out_format = "gif"
        elif fmt == "webp":
            img.save(output, format="WEBP")
            out_format = "webp"
        else:
            img.save(output, format="JPEG", quality=92)
            out_format = "jpeg"
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as MediaError
        raise MediaError(f"failed to decode image: {exc}")

    metadata = {
        "width": img.size[0],
        "height": img.size[1],
        "format": out_format,
        "original_size_bytes": len(data),
    }
    return output.getvalue(), metadata


async def load_image(
    src: ImageInput, downscale_above: int = 4096
) -> MediaFrame:
    """Load one image into a re-encoded :class:`MediaFrame`."""
    data, source = await asyncio.to_thread(_read_image_bytes, src)
    image_bytes, metadata = await asyncio.to_thread(
        _process_image_bytes, data, downscale_above
    )
    return MediaFrame(image=image_bytes, source=source, metadata=metadata)


async def load_collection(
    spec: ImageCollectionInput, downscale_above: int = 4096
) -> list[MediaFrame]:
    """Load all images in an :class:`ImageCollectionInput`, preserving order."""
    return [
        await load_image(item, downscale_above=downscale_above)
        for item in spec.images
    ]


def frame_to_data_url(frame: MediaFrame, detail: str = "original") -> dict:
    """Build an ``image_url`` content part for a frame."""
    fmt = frame.metadata.get("format", "jpeg")
    encoded = base64.b64encode(frame.image).decode("ascii")
    return {"url": f"data:image/{fmt};base64,{encoded}", "detail": detail}
