"""Deterministic video frame extraction via ffmpeg/ffprobe.

All ffmpeg/ffprobe invocations use ``asyncio.create_subprocess_exec`` with
captured stderr; ``shell=True`` is never used.
"""

from __future__ import annotations

import asyncio
import glob
import json
import math
import os
import re
import tempfile
from pathlib import Path

from ..errors import MediaError, PreprocessingError
from ..models import MediaFrame, VideoOptions

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")
_FILENAME_PTS_RE = re.compile(r"_(\d+)\.jpg$")


async def _run(cmd: list[str], label: str) -> tuple[str, str]:
    """Run a subprocess, capturing stdout/stderr; raise on nonzero exit."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise PreprocessingError(f"{label}: executable not found: {cmd[0]}") from exc
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode("utf-8", errors="replace")[-2000:]
        raise PreprocessingError(f"{label} failed (exit {proc.returncode}): {tail}")
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _parse_rate(rate: str | None) -> float | None:
    """Parse ffprobe ``num/den`` frame rate strings."""
    if not rate:
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/")
            num, den = float(num), float(den)
            return num / den if den else None
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return None


async def probe(path: str) -> dict:
    """Probe a media file with ffprobe; raise MediaError if missing."""
    if not Path(path).exists():
        raise MediaError(f"media file not found: {path}")
    stdout, _ = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ],
        "ffprobe",
    )
    try:
        info = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PreprocessingError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = info.get("streams", []) or []
    video_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    stream = video_stream or {}

    format_info = info.get("format", {}) or {}
    duration_raw = stream.get("duration") or format_info.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
        stream.get("r_frame_rate")
    )
    nb_frames_raw = stream.get("nb_frames")
    try:
        nb_frames = int(nb_frames_raw) if nb_frames_raw is not None else None
    except (TypeError, ValueError):
        nb_frames = None

    width = stream.get("width")
    height = stream.get("height")

    time_base: str | None = stream.get("time_base")
    tb_num = tb_den = None
    if time_base and "/" in time_base:
        try:
            num_s, den_s = time_base.split("/")
            tb_num, tb_den = int(num_s), int(den_s)
        except ValueError:
            tb_num = tb_den = None

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "nb_frames": nb_frames,
        "time_base": (tb_num, tb_den) if tb_num is not None else None,
    }


def _clip_window(
    duration: float | None, options: VideoOptions
) -> tuple[float, float]:
    """Return (start, end) seconds honoring start_seconds/end_seconds."""
    start = options.start_seconds if options.start_seconds is not None else 0.0
    end = options.end_seconds if options.end_seconds is not None else (
        duration or 0.0
    )
    end = max(end, start)
    return start, end


async def _extract_uniform(
    path: str,
    start: float,
    end: float,
    options: VideoOptions,
    tmpdir: str,
    duration: float | None,
    fps_in: float | None = None,
) -> list[MediaFrame]:
    """Uniform fps sampling with deterministic recomputed timestamps."""
    clip_len = end - start
    effective_fps = options.fps
    if duration and duration > 0:
        effective_fps = min(options.fps, options.max_frames / duration)

    expected = math.ceil(clip_len * effective_fps) if clip_len > 0 else 0
    if not (duration and duration > 0):
        expected = options.max_frames
    expected = min(options.max_frames, max(expected, 0))

    frames: list[MediaFrame] = []
    if expected == 0:
        return frames

    # Deterministic path: select exact input frame indices nearest to the
    # target timestamps (avoids ffmpeg fps-filter frame-count quirks).
    if fps_in and fps_in > 0 and clip_len > 0 and effective_fps > 0:
        indices: list[int] = []
        for j in range(expected):
            target = start + j / effective_fps
            index = round(target * fps_in)
            if index not in indices:
                indices.append(index)
        if indices:
            select_expr = "+".join(f"eq(n,{idx})" for idx in indices)
            cmd = ["ffmpeg", "-y", "-i", path]
            if start > 0:
                cmd += ["-ss", f"{start}"]
            if clip_len > 0:
                cmd += ["-t", f"{clip_len}"]
            cmd += [
                "-vf",
                f"select='{select_expr}'",
                "-fps_mode",
                "vfr",
                "-qscale:v",
                "3",
                os.path.join(tmpdir, "f_%06d.jpg"),
            ]
            await _run(cmd, "ffmpeg uniform extraction")
            pattern = os.path.join(tmpdir, "f_*.jpg")
            files = sorted(glob.glob(pattern))[:expected]
            for i, fpath in enumerate(files):
                data = Path(fpath).read_bytes()
                ts = start + i / effective_fps
                frames.append(
                    MediaFrame(
                        image=data,
                        timestamp=ts,
                        source=path,
                        metadata={"format": "jpeg"},
                    )
                )
            return frames

    # Fallback: rely on the fps filter, capped at max_frames.
    cmd = ["ffmpeg", "-y", "-i", path]
    if start > 0:
        cmd += ["-ss", f"{start}"]
    if clip_len > 0:
        cmd += ["-t", f"{clip_len}"]
    cmd += [
        "-vf",
        f"fps={effective_fps}",
        "-qscale:v",
        "3",
        os.path.join(tmpdir, "f_%06d.jpg"),
    ]
    await _run(cmd, "ffmpeg uniform extraction")

    pattern = os.path.join(tmpdir, "f_*.jpg")
    files = sorted(glob.glob(pattern))[:expected]
    for i, fpath in enumerate(files):
        data = Path(fpath).read_bytes()
        ts = start + i / effective_fps if effective_fps > 0 else start
        frames.append(
            MediaFrame(
                image=data,
                timestamp=ts,
                source=path,
                metadata={"format": "jpeg"},
            )
        )
    return frames


async def _extract_scene(
    path: str,
    start: float,
    end: float,
    options: VideoOptions,
    tmpdir: str,
    duration: float | None,
    probe_info: dict | None = None,
) -> tuple[list[MediaFrame], bool]:
    """Scene-change detection. Returns (frames incl. bookends, found_flag)."""
    clip_len = end - start
    cmd = ["ffmpeg", "-y", "-i", path]
    if start > 0:
        cmd += ["-ss", f"{start}"]
    if clip_len > 0:
        cmd += ["-t", f"{clip_len}"]
    cmd += [
        "-vf",
        f"select='gt(scene,{options.scene_threshold})',showinfo",
        "-fps_mode",
        "vfr",
        "-qscale:v",
        "3",
        os.path.join(tmpdir, "sc_%06d.jpg"),
    ]
    _, stderr = await _run(cmd, "ffmpeg scene_change extraction")

    all_pts = [float(m) for m in _PTS_RE.findall(stderr)]
    # showinfo reports every selected frame before output trimming; keep only
    # the ones inside the requested window (they are the ones written to disk).
    pts = [
        pts_time
        for pts_time in all_pts
        if start - 0.001 <= pts_time <= end + 0.001
    ]
    files = sorted(glob.glob(os.path.join(tmpdir, "sc_*.jpg")))

    frames: list[MediaFrame] = []
    for fpath, pts_time in zip(files, pts):
        data = Path(fpath).read_bytes()
        frames.append(
            MediaFrame(
                image=data,
                timestamp=pts_time,
                source=path,
                metadata={"format": "jpeg"},
            )
        )

    found = len(frames) > 0

    # Synthetic bookends: first frame at start and last frame near end.
    book_start = os.path.join(tmpdir, "book_start.jpg")
    await _run(
        ["ffmpeg", "-y", "-ss", f"{start}", "-i", path, "-frames:v", "1", book_start],
        "ffmpeg scene bookend (start)",
    )
    frames.append(
        MediaFrame(
            image=Path(book_start).read_bytes(),
            timestamp=start,
            source=path,
            metadata={"format": "jpeg"},
        )
    )

    # End bookend: last frame near the end. Prefer the exact last frame time
    # derived from nb_frames/fps (a plain seek to duration-0.05 can land past
    # the final frame on some encoders).
    last_frame_ts: float | None = None
    nb_frames = probe_info.get("nb_frames") if probe_info else None
    fps = probe_info.get("fps") if probe_info else None
    if nb_frames and fps:
        last_frame_ts = (nb_frames - 1) / float(fps)
    end_ts = min(end - 0.05, last_frame_ts) if last_frame_ts else end - 0.05
    end_ts = max(end_ts, start)
    if end > start:
        book_end = os.path.join(tmpdir, "book_end.jpg")
        try:
            await _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{end_ts}",
                    "-i",
                    path,
                    "-frames:v",
                    "1",
                    book_end,
                ],
                "ffmpeg scene bookend (end)",
            )
            frames.append(
                MediaFrame(
                    image=Path(book_end).read_bytes(),
                    timestamp=end_ts,
                    source=path,
                    metadata={"format": "jpeg"},
                )
            )
        except PreprocessingError:
            # Final-frame extraction is best-effort; scene frames still stand.
            pass

    frames = _dedupe_frames(frames, tolerance=0.1)
    return frames, found


async def _extract_keyframes(
    path: str,
    start: float,
    end: float,
    options: VideoOptions,
    tmpdir: str,
    time_base: tuple[int, int] | None,
) -> list[MediaFrame]:
    """Keyframe extraction; timestamps from filename pts or positional."""
    clip_len = end - start
    cmd = ["ffmpeg", "-y", "-skip_frame", "nokey", "-i", path]
    if start > 0:
        cmd += ["-ss", f"{start}"]
    if clip_len > 0:
        cmd += ["-t", f"{clip_len}"]
    cmd += [
        "-fps_mode",
        "passthrough",
        "-frame_pts",
        "1",
        os.path.join(tmpdir, "k_%06d.jpg"),
    ]
    await _run(cmd, "ffmpeg keyframe extraction")

    files = sorted(glob.glob(os.path.join(tmpdir, "k_*.jpg")))
    parsed_pts: list[int] = []
    for fpath in files:
        match = _FILENAME_PTS_RE.search(fpath)
        parsed_pts.append(int(match.group(1)) if match else -1)

    # -frame_pts writes the frame pts; but some builds write the sequential
    # index instead. Detect the sequential case and fall back positionally.
    sequential = parsed_pts == list(range(len(parsed_pts)))

    frames: list[MediaFrame] = []
    for i, fpath in enumerate(files):
        data = Path(fpath).read_bytes()
        if not sequential and parsed_pts[i] >= 0 and time_base is not None:
            pts = parsed_pts[i]
            num, den = time_base
            ts = pts * num / den if den else i / (options.fps or 25)
        else:
            ts = i / (options.fps or 25)
        frames.append(
            MediaFrame(
                image=data,
                timestamp=ts,
                source=path,
                metadata={"format": "jpeg"},
            )
        )
    return frames


def _dedupe_frames(
    frames: list[MediaFrame], tolerance: float
) -> list[MediaFrame]:
    """Drop frames whose timestamp is within ``tolerance`` of a kept frame."""
    ordered = sorted(
        frames, key=lambda f: f.timestamp if f.timestamp is not None else 0.0
    )
    result: list[MediaFrame] = []
    for frame in ordered:
        ts = frame.timestamp if frame.timestamp is not None else 0.0
        if not result:
            result.append(frame)
            continue
        last_ts = result[-1].timestamp if result[-1].timestamp is not None else 0.0
        if ts - last_ts >= tolerance:
            result.append(frame)
    return result


def _resample_uniform(frames: list[MediaFrame], max_frames: int) -> list[MediaFrame]:
    """Evenly re-sample a sorted frame list while keeping both ends."""
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return frames[:1]
    indices = {
        round(i * (len(frames) - 1) / (max_frames - 1))
        for i in range(max_frames)
    }
    return [frames[i] for i in sorted(indices)]


async def extract_frames(path: str, options: VideoOptions) -> list[MediaFrame]:
    """Extract frames per the requested sampling strategy.

    Frames are sorted ascending by timestamp; every frame carries
    ``source=path`` and a float ``timestamp``.
    """
    if not Path(path).exists():
        raise MediaError(f"media file not found: {path}")
    info = await probe(path)
    duration = info.get("duration")
    start, end = _clip_window(duration, options)
    time_base = info.get("time_base")
    fps_in = info.get("fps")

    with tempfile.TemporaryDirectory() as tmpdir:
        if options.sampling == "uniform":
            frames = await _extract_uniform(
                path, start, end, options, tmpdir, duration, fps_in
            )
        elif options.sampling == "scene_change":
            scene_frames, found = await _extract_scene(
                path, start, end, options, tmpdir, duration, info
            )
            if not found:
                frames = await _extract_uniform(
                    path, start, end, options, tmpdir, duration, fps_in
                )
            else:
                frames = scene_frames
        elif options.sampling == "keyframes":
            frames = await _extract_keyframes(
                path, start, end, options, tmpdir, time_base
            )
        else:  # adaptive
            scene_frames, _ = await _extract_scene(
                path, start, end, options, tmpdir, duration, info
            )
            sparse = VideoOptions(
                **{
                    **options.model_dump(),
                    "sampling": "uniform",
                    "fps": max(0.2, options.fps),
                }
            )
            uniform_frames = await _extract_uniform(
                path, start, end, sparse, tmpdir, duration, fps_in
            )
            frames = _dedupe_frames(
                scene_frames + uniform_frames, tolerance=0.25
            )
            frames = _resample_uniform(frames, options.max_frames)

    frames.sort(key=lambda f: f.timestamp if f.timestamp is not None else 0.0)
    return frames


def window_frames(
    frames: list[MediaFrame], window_seconds: float
) -> list[list[MediaFrame]]:
    """Partition frames into consecutive temporal windows.

    A frame belongs to the window whose ``[start, end)`` interval contains its
    timestamp. Always returns at least one window for non-empty input.
    """
    if not frames:
        return []
    if window_seconds is None or window_seconds <= 0:
        return [sorted(frames, key=lambda f: f.timestamp or 0.0)]

    ordered = sorted(
        frames, key=lambda f: f.timestamp if f.timestamp is not None else 0.0
    )
    windows: list[list[MediaFrame]] = [[]]
    window_end = window_seconds
    for frame in ordered:
        ts = frame.timestamp if frame.timestamp is not None else 0.0
        while ts >= window_end:
            windows.append([])
            window_end += window_seconds
        windows[-1].append(frame)

    while windows and not windows[-1]:
        windows.pop()
    if not windows:
        windows = [[]]
    return windows


def frames_within_duration(
    frames: list[MediaFrame], start: float, end: float
) -> list[MediaFrame]:
    """Return frames whose timestamp falls within ``[start, end]``."""
    return [
        frame
        for frame in frames
        if frame.timestamp is not None and start <= frame.timestamp <= end
    ]
