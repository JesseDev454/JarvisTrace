"""Safe bounded MP4 validation and immutable file placement for Stage 12."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import uuid

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_DURATION_SECONDS = 4 * 60 * 60


def safe_filename(value: str) -> str:
    name = Path(str(value).replace("\\", "/")).name
    name = "".join(c for c in name if c.isprintable() and c not in '<>:"|?*')
    name = name.strip(" .")[:180]
    if not name or not name.lower().endswith(".mp4"):
        raise ValueError("Choose a file with the .mp4 extension")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_mp4(path: Path, *, ffprobe: str | None = None, ffmpeg: str | None = None, timeout=20) -> dict:
    local=Path(__file__).resolve().parents[1]/'scene-search-prototype/.env'
    try:
        from dotenv import dotenv_values
        configured=dotenv_values(local)
    except Exception:
        configured={}
    executable = ffprobe or os.environ.get("FFPROBE") or configured.get('FFPROBE') or "ffprobe"
    ffmpeg_executable = ffmpeg or os.environ.get("FFMPEG") or configured.get('FFMPEG') or "ffmpeg"
    command = [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        raise ValueError("The MP4 could not be verified. Re-export as a playable H.264 MP4 and try again") from None
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError("The recording must contain exactly one video stream")
    format_name = str(data.get("format", {}).get("format_name", "")).lower()
    if not {"mp4", "mov"}.intersection(format_name.split(",")):
        raise ValueError("Use an H.264 video in an MP4 container")
    stream = streams[0]
    if stream.get("codec_name") != "h264":
        raise ValueError("Use H.264 video in an MP4 container so recordings play in the browser")
    try:
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(data["format"]["duration"])
        fps = _fraction(stream["avg_frame_rate"])
        nominal_fps = _fraction(stream["r_frame_rate"])
        time_base = _fraction(stream["time_base"])
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        raise ValueError("The video is missing usable frame-rate, dimensions, or duration metadata") from None
    # Fragmented MP4 duration metadata can be short by two frame ticks. Use a
    # little more rate tolerance, then verify the actual timestamp intervals.
    rate_tolerance = max(1e-4, nominal_fps * 2e-3)
    if (min(width, height) < 32 or max(width, height) > 16384 or not 0 < fps <= 120
            or not 0 < duration <= MAX_DURATION_SECONDS or abs(fps - nominal_fps) > rate_tolerance):
        raise ValueError("The video must be a constant-frame-rate recording up to four hours and 120 FPS")
    _verify_constant_frame_rate(path, nominal_fps, time_base, ffmpeg_executable, timeout)
    return {"width": width, "height": height, "fps": fps, "duration_seconds": duration,
            "format_name": data.get("format", {}).get("format_name", "")}


def _verify_constant_frame_rate(path: Path, nominal_fps: float, time_base: float,
                                ffmpeg: str, timeout: int) -> None:
    """Check decoded frame timestamps, allowing only time-base rounding jitter."""
    command = [ffmpeg, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-vf", "vfrdet",
               "-an", "-f", "null", "-"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        raise ValueError("The MP4 could not be verified. Re-export as a playable H.264 MP4 and try again") from None
    variation = re.search(r"\bVFR:\s*[\d.]+\s*\((\d+)/(\d+)\)", result.stderr or "")
    if not variation:
        raise ValueError("The MP4 frame timing could not be verified")
    varying_intervals = int(variation.group(1))
    if varying_intervals == 0:
        return
    report = re.search(r"\bVFR:\s*[\d.]+\s*\(\d+/\d+\)\s*min:\s*(\d+)\s*max:\s*(\d+)",
                       result.stderr or "")
    if not report:
        raise ValueError("The MP4 frame timing could not be verified")
    min_ticks, max_ticks = (int(value) for value in report.groups())
    expected_ticks = 1 / (nominal_fps * time_base)
    # FFmpeg's timestamp time base may quantize a frame interval to adjacent
    # integer ticks. For common high-resolution clocks, tolerate one extra tick
    # for MediaMTX muxer jitter; never tolerate a full missing frame interval.
    if expected_ticks >= 100:
        lower_ticks, upper_ticks = expected_ticks - 1.000001, expected_ticks + 1.000001
    else:
        lower_ticks, upper_ticks = int(expected_ticks), int(expected_ticks + 0.999999)
    if min_ticks < lower_ticks or max_ticks > upper_ticks:
        raise ValueError("The video must be a constant-frame-rate recording up to four hours and 120 FPS")


def _fraction(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:/\d+)?", value):
        raise ValueError("invalid frame rate")
    numerator, *denominator = value.split("/")
    return float(numerator) / float(denominator[0] if denominator else 1)


def store_upload(upload, *, filename: str, storage_root: Path, max_bytes=MAX_UPLOAD_BYTES, video_id=None,
                 probe=None, expected_sha256=None):
    """Stream an uploaded file to a unique private temp path, validate, then publish atomically."""
    filename = safe_filename(filename)
    storage_root = Path(storage_root).resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    temp_id = uuid.uuid4().hex
    video_id = video_id or f"upload_{temp_id}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", video_id):
        raise ValueError("Invalid recording ID")
    final_path = storage_root / f"{video_id}.mp4"
    temp_path = storage_root / f".{temp_id}.upload"
    final_path_created = False
    digest = hashlib.sha256()
    total = 0
    stream = getattr(upload, "file", upload)
    try:
        with temp_path.open("xb") as output:
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > max_bytes:
                    raise ValueError("The file is larger than the 1 GB upload limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if total <= 0:
            raise ValueError("The recording is empty")
        source_sha256 = digest.hexdigest()
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise ValueError("The MediaMTX segment changed while it was being copied")
        metadata = (probe or inspect_mp4)(temp_path)
        try:
            os.link(temp_path, final_path)
            final_path_created = True
            temp_path.unlink()
        except FileExistsError:
            if sha256_file(final_path) != source_sha256:
                raise ValueError("This immutable recording ID already refers to different content") from None
            temp_path.unlink()
        return {"video_id": video_id, "filename": filename, "path": final_path,
                "size_bytes": total, "source_sha256": source_sha256, **metadata}
    except BaseException:
        temp_path.unlink(missing_ok=True)
        if final_path_created:
            final_path.unlink(missing_ok=True)
        raise


def parse_browser_datetime(local_value: str | None, timezone_name: str | None):
    from src.stage12_workspace import parse_capture_start
    return parse_capture_start(local_value, timezone_name)


def segment_ids(camera_id: str, relative_path: str, source_sha256: str):
    event_id = hashlib.sha256(f"{camera_id}\0{relative_path}\0{source_sha256}".encode()).hexdigest()
    return event_id, f"mtx_{event_id[:32]}"


def segment_start_from_record_path(path: str):
    """Read the explicit Unix-epoch prefix configured in mediamtx.yml."""
    match = re.match(r"^(\d{10})(?:-(\d{6}))?$", Path(path).stem)
    if not match:
        return None
    seconds = int(match.group(1))
    micros = int(match.group(2) or 0)
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=micros)
    except (OverflowError, OSError, ValueError):
        return None
