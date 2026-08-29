"""Generate deterministic synthetic fixtures for the runnable examples.

Idempotent: existing fixture files are left untouched. ffmpeg is required
for ``session.mp4``; PyMuPDF for ``report.pdf``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    _FONT = ImageFont.load_default(size=24)
except TypeError:  # Pillow < 10.1: load_default has no size argument
    _FONT = ImageFont.load_default()


def _skip_if_exists(path: Path) -> bool:
    """Return True (and skip creation) when the fixture already exists."""
    return path.exists()


def _make_ui_screenshot(path: Path) -> None:
    """Mock UI: header, sidebar, submit button, error dialog."""
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 60], fill=(50, 50, 50))
    draw.text((20, 20), "App Header", fill="white", font=_FONT)
    draw.rectangle([0, 60, 140, 420], fill=(225, 225, 225))
    draw.text((20, 80), "Nav item 1", fill="black", font=_FONT)
    draw.text((20, 105), "Nav item 2", fill="black", font=_FONT)
    draw.rectangle([500, 350, 620, 390], fill=(0, 120, 200))
    draw.text((535, 362), "Submit", fill="white", font=_FONT)
    draw.rectangle([180, 140, 560, 300], fill=(255, 245, 245), outline=(180, 0, 0))
    draw.rectangle([180, 140, 560, 172], fill=(180, 0, 0))
    draw.text((200, 148), "Error", fill="white", font=_FONT)
    draw.text(
        (200, 190),
        "TypeError: cannot read property 'user' of undefined",
        fill="black",
        font=_FONT,
    )
    img.save(path)


def _make_architecture(path: Path) -> None:
    """Boxes-and-arrows architecture diagram."""
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(img)
    boxes = {
        "API Gateway": (30, 170, 190, 230),
        "Auth Service": (270, 60, 430, 120),
        "DB": (500, 60, 620, 120),
        "Worker Queue": (270, 290, 430, 350),
    }
    for label, (x0, y0, x1, y1) in boxes.items():
        draw.rectangle([x0, y0, x1, y1], fill=(220, 235, 250), outline="black")
        draw.text((x0 + 8, (y0 + y1) // 2 - 6), label, fill="black", font=_FONT)
    # API Gateway -> Auth Service -> DB
    draw.line([(190, 200), (270, 90)], fill="black", width=2)
    draw.line([(430, 90), (500, 90)], fill="black", width=2)
    # API Gateway -> Worker Queue
    draw.line([(190, 200), (270, 320)], fill="black", width=2)
    draw.polygon([(270, 90), (260, 82), (260, 98)], fill="black")
    draw.polygon([(500, 90), (490, 82), (490, 98)], fill="black")
    draw.polygon([(270, 320), (260, 312), (260, 328)], fill="black")
    img.save(path)


def _make_design_reference(path: Path) -> None:
    """Reference mock screen: blue primary button, correct label."""
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 400, 50], fill=(60, 60, 60))
    draw.text((150, 18), "Sign Up", fill="white", font=_FONT)
    draw.rectangle([30, 90, 370, 130], fill=(230, 230, 230))
    draw.rectangle([30, 150, 370, 190], fill=(230, 230, 230))
    draw.rectangle([270, 230, 370, 270], fill=(0, 90, 220))
    draw.text((295, 243), "Sign Up", fill="white", font=_FONT)
    img.save(path)


def _make_rendered_actual(path: Path) -> None:
    """Actual implementation: green button and 'Signup' label."""
    if _skip_if_exists(path):
        return
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 400, 50], fill=(60, 60, 60))
    draw.text((150, 18), "Sign Up", fill="white", font=_FONT)
    draw.rectangle([30, 90, 370, 130], fill=(230, 230, 230))
    draw.rectangle([30, 150, 370, 190], fill=(230, 230, 230))
    draw.rectangle([270, 230, 370, 270], fill=(0, 150, 60))
    draw.text((300, 243), "Signup", fill="white", font=_FONT)
    img.save(path)


def _make_report_pdf(path: Path) -> None:
    """5-page report: title, paragraph, and a small table per page."""
    if _skip_if_exists(path):
        return
    import pymupdf  # lazy import: only needed for PDF fixtures

    facts = {
        2: "The migration is scheduled for Q4.",
        4: "Churn rate is 2.3%.",
    }
    doc = pymupdf.open()
    for page_number in range(1, 6):
        page = doc.new_page()
        page.insert_text((72, 72), f"Quarterly Report - Page {page_number}")
        page.insert_text(
            (72, 110),
            "This report summarizes the current operational status.",
        )
        table_rows = [
            ("Metric", "Value", "Status"),
            ("Uptime", "99.9%", "Good"),
            ("Latency", "42ms", "Good"),
            ("Errors", "0.1%", "Watch"),
        ]
        y = 160
        for col, value, status in table_rows:
            page.insert_text((72, y), f"{col:<12}{value:<12}{status}")
            y += 22
        if page_number in facts:
            page.insert_text((72, y + 20), facts[page_number])
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


def _make_session_mp4(path: Path) -> None:
    """8s synthetic video: 4 scene colors with a moving white square."""
    if _skip_if_exists(path):
        return
    if shutil.which("ffmpeg") is None:
        print(
            "warning: ffmpeg not available; skipping session.mp4 fixture",
            file=sys.stderr,
        )
        return
    base = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=yellow:s=320x240:d=2",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-filter_complex",
        (
            "[0:v][1:v][2:v][3:v]concat=n=4:v=1[v];"
            "[v]drawbox=x='mod(t*40,280)':y=100:w=40:h=40:color=white:t=fill[vout]"
        ),
        "-map", "[vout]",
        "-r", "10",
        str(path),
    ]
    fallback = base[:-1] + ["-c:v", "mpeg4", "-q:v", "3", str(path)]
    _run_ffmpeg(base, fallback)


def make_all(target: str | Path) -> dict[str, Path]:
    """Create all example fixtures under ``target``; return id -> path."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    _make_ui_screenshot(target / "ui_screenshot.png")
    _make_architecture(target / "architecture.png")
    _make_design_reference(target / "design_reference.png")
    _make_rendered_actual(target / "rendered_actual.png")
    _make_report_pdf(target / "report.pdf")
    _make_session_mp4(target / "session.mp4")

    return {
        "ui_screenshot": target / "ui_screenshot.png",
        "architecture": target / "architecture.png",
        "design_reference": target / "design_reference.png",
        "rendered_actual": target / "rendered_actual.png",
        "report": target / "report.pdf",
        "session": target / "session.mp4",
    }


if __name__ == "__main__":
    fixtures = make_all(Path(__file__).resolve().parent / "fixtures")
    print("generated fixtures:")
    for fixture_id, path in fixtures.items():
        print(f"  {fixture_id}: {path}")
