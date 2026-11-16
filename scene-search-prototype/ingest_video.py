"""Stage 2: index representative frames from a real, local, constant-FPS video.

This does NOT detect objects or generate descriptions. It stores timestamped
frame candidates for manual annotation and later VLM processing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def image_signature(frame: np.ndarray) -> np.ndarray:
    """Small grayscale representation used by our *heuristic* change filter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(gray, (256, 144)), (5, 5), 0)


def changed_pixel_fraction(current: np.ndarray, previous: np.ndarray, pixel_delta: int = 25) -> float:
    """Fraction of pixels differing from the last *saved* candidate."""
    return float(np.mean(cv2.absdiff(current, previous) >= pixel_delta))


def choose_candidate(change_fraction: float | None, seconds_since_saved: float,
                     threshold: float, guard_seconds: float) -> str | None:
    """Return why a candidate is kept, or None if it is skipped."""
    if change_fraction is None:
        return 'first-frame'
    if change_fraction >= threshold:
        return 'visual-change'
    if seconds_since_saved >= guard_seconds - 1e-6:
        return 'guard-sample'
    return None


def content_id(path: Path) -> str:
    """Stable short identifier from file size plus first/last chunks.

    This is for local duplicate detection, NOT a cryptographic proof of identity.
    """
    digest = hashlib.sha256()
    digest.update(str(path.stat().st_size).encode())
    with path.open('rb') as file:
        digest.update(file.read(1024 * 1024))
        if path.stat().st_size > 1024 * 1024:
            file.seek(max(0, path.stat().st_size - 1024 * 1024))
            digest.update(file.read(1024 * 1024))
    return f'{path.stem[:48]}_{digest.hexdigest()[:12]}'


@dataclass
class Candidate:
    timestamp_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    frame_path: str
    change_fraction: float | None
    reason: str


def ingest(video: Path, camera_id: str, every: float, guard: float,
           threshold: float, max_minutes: float, dry_run: bool) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f'Video file does not exist: {video}')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('OpenCV could not read this video. Convert to MP4/H.264 with FFmpeg and retry.')

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frames_total <= 0:
            raise RuntimeError('Missing usable FPS/frame count. This prototype expects constant-frame-rate video.')
        duration = frames_total / fps
        time_limit = min(duration, max_minutes * 60)
        stride = max(1, round(every * fps))
        video_id = content_id(video)
        frame_dir = HERE / 'data' / 'frames' / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        previous_saved = None
        last_saved_at = -float('inf')
        frame_index = 0
        sampled_count = 0
        candidates = []
        while (frame_index / fps) < time_limit:
            if not cap.grab():
                break
            if frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    print(f'WARNING: failed to decode frame {frame_index}; skipping')
                    frame_index += 1
                    continue
                sampled_count += 1
                timestamp = frame_index / fps
                signature = image_signature(frame)
                change_fraction = (changed_pixel_fraction(signature, previous_saved)
                                   if previous_saved is not None else None)
                reason = choose_candidate(change_fraction, timestamp - last_saved_at,
                                          threshold, guard)
                if reason:
                    filename = f'frame_{frame_index:09d}_{timestamp:011.3f}.jpg'
                    out_path = frame_dir / filename
                    if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                        raise OSError(f'Failed to save {out_path}')
                    # Around the observation time; these are NOT validated scene boundaries.
                    clip_start = max(0, timestamp - every / 2)
                    clip_end = min(duration, timestamp + every / 2)
                    if clip_end <= clip_start:
                        clip_end = min(duration, clip_start + 0.1)
                    candidates.append(Candidate(
                        timestamp_seconds=round(timestamp, 3),
                        clip_start_seconds=round(clip_start, 3),
                        clip_end_seconds=round(clip_end, 3),
                        frame_path=str(out_path),
                        change_fraction=round(change_fraction, 4) if change_fraction is not None else None,
                        reason=reason,
                    ))
                    previous_saved = signature
                    last_saved_at = timestamp
            frame_index += 1
    finally:
        cap.release()

    result = {
        'video_id': video_id,
        'video_path': str(video.resolve()),
        'camera_id': camera_id,
        'fps': round(fps, 4),
        'duration_seconds': round(duration, 3),
        'processed_up_to_seconds': round(min(time_limit, frame_index / fps), 3),
        'sample_interval_seconds': every,
        'guard_seconds': guard,
        'change_threshold_fraction': threshold,
        'samples_examined': sampled_count,
        'candidates_saved': len(candidates),
        'candidates': [asdict(x) for x in candidates],
    }
    manifest_path = frame_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not dry_run:
        store_metadata(result)
    return result


def store_metadata(result: dict) -> None:
    """Store metadata only, leaving video bytes on disk."""
    from app import open_database

    with open_database() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS video_recordings (
            video_id TEXT PRIMARY KEY,
            local_video_path TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            fps DOUBLE PRECISION NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS frame_candidates (
            id BIGSERIAL PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES video_recordings(video_id),
            timestamp_seconds NUMERIC(12,3) NOT NULL,
            clip_start_seconds NUMERIC(12,3) NOT NULL,
            clip_end_seconds NUMERIC(12,3) NOT NULL,
            frame_path TEXT NOT NULL,
            change_fraction DOUBLE PRECISION,
            reason TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(video_id, timestamp_seconds)
        )''')
        conn.execute('''INSERT INTO video_recordings
            (video_id, local_video_path, camera_id, fps, duration_seconds)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT(video_id) DO UPDATE
            SET local_video_path=EXCLUDED.local_video_path,
                camera_id=EXCLUDED.camera_id,
                fps=EXCLUDED.fps, duration_seconds=EXCLUDED.duration_seconds''',
            (result['video_id'], result['video_path'], result['camera_id'],
             result['fps'], result['duration_seconds']))
        for c in result['candidates']:
            conn.execute('''INSERT INTO frame_candidates
                (video_id, timestamp_seconds, clip_start_seconds, clip_end_seconds,
                 frame_path, change_fraction, reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (video_id, timestamp_seconds) DO UPDATE
                SET frame_path=EXCLUDED.frame_path,
                    change_fraction=EXCLUDED.change_fraction, reason=EXCLUDED.reason,
                    clip_start_seconds=EXCLUDED.clip_start_seconds,
                    clip_end_seconds=EXCLUDED.clip_end_seconds''',
                (result['video_id'], c['timestamp_seconds'], c['clip_start_seconds'],
                 c['clip_end_seconds'], c['frame_path'], c['change_fraction'], c['reason']))


def main():
    parser = argparse.ArgumentParser(description='Extract representative frames and timestamps from a real recording')
    parser.add_argument('--video', type=Path, required=True, help='Path to real video e.g. videos\\test01.mp4')
    parser.add_argument('--camera', required=True, help='Camera ID e.g. CAM-TEST')
    parser.add_argument('--every', type=float, default=1.0, help='Sampling interval (seconds), default 1')
    parser.add_argument('--guard', type=float, default=5.0, help='Max interval without saving a frame, default 5')
    parser.add_argument('--threshold', type=float, default=0.08, help='Changed pixel fraction threshold, default 0.08')
    parser.add_argument('--max-minutes', type=float, default=5.0, help='Process at most this many minutes, default 5')
    parser.add_argument('--dry-run', action='store_true', help='Extract images + manifest without DB')
    args = parser.parse_args()
    if args.every <= 0 or args.guard <= 0 or args.max_minutes <= 0 or not 0 <= args.threshold <= 1:
        parser.error('--every, --guard and --max-minutes must be > 0; --threshold must be 0..1')
    try:
        result = ingest(args.video.resolve(), args.camera, args.every, args.guard,
                        args.threshold, args.max_minutes, args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'ERROR: {error}\n')
    print(f"Video ID: {result['video_id']}")
    print(f"Duration: {result['duration_seconds']} sec; sampled {result['samples_examined']} times")
    print(f"Saved {result['candidates_saved']} frame candidates")
    print(f"Manifest + JPGs: {HERE / 'data' / 'frames' / result['video_id']}")
    if args.dry_run:
        print('DRY RUN: no PostgreSQL records saved')
    else:
        print('Video and frame metadata stored in PostgreSQL')
        print(f"Next: python label_candidates.py --video-id {result['video_id']} --limit 10")


if __name__ == '__main__':
    main()
