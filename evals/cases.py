"""Evaluation case definitions and deterministic synthetic fixtures.

``generate_media`` creates all fixtures (PIL / PyMuPDF / ffmpeg) in one
function, idempotently (existing files are left untouched). Every built-in
case carries a ``tags`` entry of ``image`` | ``document`` | ``video`` so the
runner knows how to load its media.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

try:
    _FONT = ImageFont.load_default(size=28)
except TypeError:  # Pillow < 10.1: load_default has no size argument
    _FONT = ImageFont.load_default()


class InvoiceExtraction(BaseModel):
    """Structured extraction schema for invoices."""

    vendor: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    total: float | None = None
    currency: str | None = None


SCHEMAS: dict[str, type[BaseModel]] = {"invoice": InvoiceExtraction}


class VisionEvalCase(BaseModel):
    """One evaluation case: media, prompt, expectations, and evaluator."""

    id: str
    description: str
    media: list[str] = Field(description="relative paths under the case media dir")
    prompt: str
    system: str | None = None
    task: str = "describe"
    output_schema_name: str | None = None
    expected: dict | None = Field(default=None, description="lightweight expected facts")
    evaluator: str = "llm_judge"
    tags: list[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Fixture generation (PIL / PyMuPDF / ffmpeg, idempotent)
# --------------------------------------------------------------------------- #


def _skip_if_exists(path: Path) -> bool:
    """Return True (and do nothing) when the fixture already exists."""
    return path.exists()


def _make_objects_png(path: Path) -> None:
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    # Red square on the left (x 50-150).
    draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
    # Blue circle in the middle-right (x 220-320).
    draw.ellipse([220, 60, 320, 160], fill=(0, 0, 255))
    # Green triangle bottom center (x 140-240, y 160-260).
    draw.polygon([(140, 160), (240, 160), (190, 260)], fill=(0, 128, 0))
    img.save(path)


def _make_invoice_png(path: Path) -> None:
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (500, 400), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "INVOICE",
        "Invoice No: INV-2026-0042",
        "Date: 2026-08-20",
        "Vendor: Acme Widgets Ltd",
        "Total: $1,234.50",
        "Currency: USD",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=_FONT)
        y += 45
    img.save(path)


def _make_ui(path: Path, *, title: str, button_color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    # Shared header bar.
    draw.rectangle([0, 0, 400, 50], fill=(60, 60, 60))
    draw.text((160, 18), "My App", fill="white", font=_FONT)
    # Title text (differs between the two screenshots).
    draw.text((30, 85), title, fill="black", font=_FONT)
    # Placeholder content rows.
    for i in range(3):
        draw.rectangle([30, 130 + i * 40, 370, 158 + i * 40], fill=(230, 230, 230))
    # Shared "Submit" button, bottom-right (color differs).
    draw.rectangle([270, 230, 370, 270], fill=button_color)
    draw.text((290, 236), "Submit", fill="white", font=_FONT)
    img.save(path)


def _make_ui_a_png(path: Path) -> None:
    if _skip_if_exists(path):
        return
    _make_ui(path, title="Dashboard", button_color=(0, 180, 0))


def _make_ui_b_png(path: Path) -> None:
    if _skip_if_exists(path):
        return
    _make_ui(path, title="Dashbord", button_color=(200, 0, 0))


def _make_doc3_pdf(path: Path) -> None:
    if _skip_if_exists(path):
        return
    import pymupdf  # lazy import: only needed for documents

    facts = [
        "The server fleet comprises 42 machines in Amsterdam.",
        "PostgreSQL 16 is the primary datastore.",
        "Deployments run on Tuesdays at 14:00 CET.",
    ]
    doc = pymupdf.open()
    for index, fact in enumerate(facts, start=1):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {index}")
        page.insert_text((72, 120), fact)
    doc.save(str(path))
    doc.close()


def _run_ffmpeg(cmd: list[str], fallback: list[str] | None = None) -> None:
    """Run ffmpeg; raise RuntimeError with stderr tail on failure."""
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode == 0:
        return
    if fallback is not None:
        result = subprocess.run(fallback, capture_output=True, check=False)
        if result.returncode == 0:
            return
    stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
    raise RuntimeError(f"ffmpeg failed: {stderr}")


def _make_video3_mp4(path: Path) -> None:
    if _skip_if_exists(path):
        return
    if shutil.which("ffmpeg") is None:
        print(
            "warning: ffmpeg not available; skipping video3.mp4 fixture",
            file=sys.stderr,
        )
        return
    base = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=yellow:s=320x240:d=2",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1",
        "-r", "10",
        str(path),
    ]
    fallback = base[:-1] + ["-c:v", "mpeg4", "-q:v", "3", str(path)]
    _run_ffmpeg(base, fallback)


def _make_verify_png(path: Path) -> None:
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 400, 50], fill=(60, 60, 60))
    # Clearly visible red "Submit" button, bottom-right.
    draw.rectangle([270, 230, 370, 270], fill=(200, 0, 0))
    draw.text((292, 236), "Submit", fill="white", font=_FONT)
    img.save(path)


def generate_media(media_dir: str | Path) -> dict[str, str]:
    """Create all synthetic fixtures under ``media_dir`` (idempotent).

    Returns a mapping of fixture id -> absolute path.
    """
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    _make_objects_png(media_dir / "objects.png")
    _make_invoice_png(media_dir / "invoice.png")
    _make_ui_a_png(media_dir / "ui_a.png")
    _make_ui_b_png(media_dir / "ui_b.png")
    _make_doc3_pdf(media_dir / "doc3.pdf")
    _make_video3_mp4(media_dir / "video3.mp4")
    _make_verify_png(media_dir / "verify.png")

    return {
        "objects": str(media_dir / "objects.png"),
        "invoice": str(media_dir / "invoice.png"),
        "ui_a": str(media_dir / "ui_a.png"),
        "ui_b": str(media_dir / "ui_b.png"),
        "doc3": str(media_dir / "doc3.pdf"),
        "video3": str(media_dir / "video3.mp4"),
        "verify": str(media_dir / "verify.png"),
    }


# --------------------------------------------------------------------------- #
# Built-in cases
# --------------------------------------------------------------------------- #

BUILTIN_CASES: list[VisionEvalCase] = [
    VisionEvalCase(
        id="objects_case",
        description="Describe distinct shapes and their colors in a synthetic image.",
        media=["objects.png"],
        prompt="List the distinct shapes and their colors.",
        task="describe",
        evaluator="llm_judge",
        tags=["image"],
        expected={"answers_include": ["red", "blue", "green"]},
    ),
    VisionEvalCase(
        id="invoice_case",
        description="Extract invoice fields from a synthetic invoice image.",
        media=["invoice.png"],
        prompt="Extract the invoice fields.",
        task="extract",
        output_schema_name="invoice",
        expected={
            "invoice_number": "INV-2026-0042",
            "vendor": "Acme Widgets Ltd",
            "total": 1234.5,
            "currency": "USD",
        },
        evaluator="structured_exact",
        tags=["image"],
    ),
    VisionEvalCase(
        id="compare_case",
        description="Compare two mock UI screenshots with subtle differences.",
        media=["ui_a.png", "ui_b.png"],
        prompt=(
            "Compare the two screenshots and list every difference, "
            "including color changes and typos (misspelled text). Use the "
            "words 'color' and 'typo' in your answer when they apply."
        ),
        task="compare",
        evaluator="llm_judge",
        tags=["image"],
        expected={"differences_include": ["color", "typo"]},
    ),
    VisionEvalCase(
        id="doc_case",
        description="Answer a question grounded in a 3-page PDF.",
        media=["doc3.pdf"],
        prompt="How many machines are in the fleet? Where?",
        task="document_analysis",
        evaluator="llm_judge",
        tags=["document"],
        expected={"answers_include": ["42", "Amsterdam"]},
    ),
    VisionEvalCase(
        id="video_case",
        description="Identify the order of colors in a synthetic video.",
        media=["video3.mp4"],
        prompt="What is the order of colors in this video?",
        task="video_summary",
        evaluator="llm_judge",
        tags=["video"],
        expected={"answers_include": ["blue", "green", "yellow"]},
        options={"video": {"sampling": "uniform", "fps": 1.0, "max_frames": 12}},
    ),
    VisionEvalCase(
        id="verify_case",
        description="Verify visual claims about a synthetic UI.",
        media=["verify.png"],
        prompt=(
            'Verify: {"claims": [{"claim": "A Submit button is visible", '
            '"want": "pass"}, {"claim": "The button is blue", "want": "fail"}]}'
        ),
        task="verification",
        evaluator="verification_exact",
        tags=["image"],
        expected={
            "statuses": {
                "Submit button is visible": "pass",
                "The button is blue": "fail",
            }
        },
    ),
]
