"""Stage 9: import one local tracking export and describe image-space trajectories.

No model, database, surface inference or physical-identity matching is involved.
Input contract: constant FPS, stride-one video export with save_conf=True.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('FPS must be finite and positive; no fallback is inferred')
        for value in (self.width, self.height, self.frame_count):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Dimensions and frame count must be positive integers')


def _ranges(indices) -> list[list[int]]:
    """Inclusive integer ranges, for compact reporting of observation gaps."""
    result = []
    for n in sorted(set(indices)):
        if result and result[-1][1] + 1 == n:
            result[-1][1] = n
        else:
            result.append([n, n])
    return result


def _integer(token: str, name: str, minimum: int = 0) -> int:
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f'Invalid {name}') from exc
    if not value.is_finite() or value != value.to_integral_value() or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def load_yolo_labels(labels_dir: Path, video_stem: str, video: VideoInfo,
                     class_id: int = 24) -> dict:
    """Read normalized detection boxes, retaining source line and optional track ID.

    Six columns are explicitly confidence-bearing *untracked* detections, never
    guessed to be confidence-free tracks. The caller must confirm save_conf=True.
    Reject duplicate exports instead of silently counting appended runs as data.
    """
    paths = sorted(Path(labels_dir).glob('*.txt'))
    if not paths:
        raise ValueError('No label TXT files found')
    pattern = re.compile(re.escape(video_stem) + r'_(\d+)\.txt', re.IGNORECASE)
    files = {}
    digest = hashlib.sha256()
    observations = []
    tracked_keys = set()
    raw_keys = set()
    all_rows = 0
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f'Unexpected label filename: {path.name}; use one video export')
        frame = int(match[1]) - 1
        if not 0 <= frame < video.frame_count:
            raise ValueError(f'{path.name}: frame outside video; suffixes must be one-based')
        if frame in files:
            raise ValueError(f'{path.name}: duplicate frame file')
        files[frame] = path.name
        payload = path.read_bytes()
        # Name + content hashes give a stable, boundary-unambiguous export identity.
        digest.update(path.name.encode('utf-8') + b'\0' + hashlib.sha256(payload).digest())
        for line_number, line in enumerate(payload.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                if len(parts) not in (6, 7):
                    raise ValueError('Expected class cx cy width height confidence [track_id]')
                cls = _integer(parts[0], 'class ID')
                numeric = [float(x) for x in parts[1:6]]
                if not all(math.isfinite(x) for x in numeric):
                    raise ValueError('Nonfinite number')
                cx, cy, width, height, confidence = numeric
                track_id = _integer(parts[6], 'track ID') if len(parts) == 7 else None
                if not 0 <= confidence <= 1 or not (0 < width <= 1 and 0 < height <= 1):
                    raise ValueError('Confidence or box size outside valid range')
                xyxy = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
                # YOLO saves rounded normalized floats; tolerate only rounding at edges.
                if any(x < -1e-5 or x > 1+1e-5 for x in xyxy):
                    raise ValueError('Box outside image')
                raw_key = (frame, cls, *numeric, track_id)
                tracked_key = (frame, track_id)
                if raw_key in raw_keys or (track_id is not None and tracked_key in tracked_keys):
                    raise ValueError('duplicate observation; possible appended/repeated export')
                raw_keys.add(raw_key)
                if track_id is not None:
                    tracked_keys.add(tracked_key)
                all_rows += 1
                if cls != class_id:
                    continue
                bbox = [round(min(1, max(0, x))*size, 6)
                        for x, size in zip(xyxy, (video.width, video.height)*2)]
                observations.append({
                    'frame_index': frame, 'timestamp_seconds': frame / video.fps,
                    'class_id': cls, 'local_track_id': track_id, 'confidence': confidence,
                    'bbox_xyxy': bbox,
                    'center_xy': [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2],
                    'source_label': path.name, 'source_line': line_number,
                })
            except ValueError as exc:
                raise ValueError(f'{path.name}:{line_number}: {exc}') from exc
    observations.sort(key=lambda r: (r['frame_index'], r['source_line']))
    return {
        'observations': observations, 'label_file_count': len(files),
        'all_class_row_count': all_rows, 'labels_sha256': digest.hexdigest(),
        'frames_without_label_file': _ranges(set(range(video.frame_count)) - files.keys()),
    }


def _spatial_metrics(rows: list[dict], video: VideoInfo) -> dict:
    xs, ys = zip(*(r['center_xy'] for r in rows))
    # Axis-aligned center-envelope diagonal, NOT accumulated detector jitter.
    extent = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
    net = math.dist(rows[0]['center_xy'], rows[-1]['center_xy'])
    return {'center_extent_pixels': extent,
            'center_extent_diagonal_fraction': extent/math.hypot(video.width, video.height),
            'net_center_displacement_pixels': net}


def analyze_observations(observations: list[dict], video: VideoInfo, *,
                         assume_static_camera: bool = False, window_seconds: float = 1.0,
                         low_extent: float = .01, movement_extent: float = .03,
                         minimum_span_seconds: float = .5,
                         fragment_gap_seconds: float = 1.0) -> dict:
    """Summarize a *single recording/run/class*; never link IDs or fill missing frames.

    Thresholds are configurable exploratory heuristics, not calibrated confidence.
    The explicit static-camera flag is a caller assumption, not motion estimation.
    """
    settings = (window_seconds, low_extent, movement_extent,
                minimum_span_seconds, fragment_gap_seconds)
    if (not all(math.isfinite(x) for x in settings) or window_seconds <= 0
            or not 0 <= low_extent < movement_extent <= 1
            or minimum_span_seconds < 0 or fragment_gap_seconds < 0):
        raise ValueError('Invalid analysis settings')
    rows = sorted(observations, key=lambda r: (r['frame_index'], r['source_line']))
    grouped = defaultdict(list)
    frame_tracks = defaultdict(set)
    for row in rows:
        if row['local_track_id'] is not None:
            grouped[row['local_track_id']].append(row)
            frame_tracks[row['frame_index']].add(row['local_track_id'])
    tracks = []
    for track_id, group in sorted(grouped.items()):
        segments = []
        gaps = []
        for row in group:
            if not segments or row['frame_index'] != segments[-1][-1]['frame_index'] + 1:
                if segments:
                    gaps.append([segments[-1][-1]['frame_index']+1, row['frame_index']-1])
                segments.append([])
            segments[-1].append(row)
        windows = []
        for segment in segments:
            buckets = defaultdict(list)
            for row in segment:
                elapsed = (row['frame_index']-segment[0]['frame_index']) / video.fps
                buckets[math.floor(elapsed/window_seconds + 1e-9)].append(row)
            previous = None
            for bucket_rows in buckets.values():
                # Overlap by one observed frame so changes at bucket edges survive.
                # This never crosses a missing-frame gap or another track ID.
                bucket = ([previous] if previous is not None else []) + bucket_rows
                previous = bucket_rows[-1]
                metrics = _spatial_metrics(bucket, video)
                span = (bucket[-1]['frame_index']-bucket[0]['frame_index'])/video.fps
                extent = metrics['center_extent_diagonal_fraction']
                if len(bucket) < 2:
                    state = 'insufficient_duration'
                elif not assume_static_camera:
                    state = 'camera_motion_unassessed'
                elif extent >= movement_extent:
                    state = 'image_displacement_candidate'
                elif span + 1e-9 < minimum_span_seconds:
                    state = 'insufficient_duration'
                elif extent <= low_extent:
                    state = 'low_image_displacement_candidate'
                else:
                    state = 'indeterminate_image_displacement'
                windows.append({
                    'first_frame': bucket[0]['frame_index'], 'last_frame': bucket[-1]['frame_index'],
                    'start_seconds': bucket[0]['timestamp_seconds'],
                    'end_seconds': bucket[-1]['timestamp_seconds'],
                    'observation_count': len(bucket), 'state': state, **metrics,
                })
        tracks.append({
            'local_track_id': track_id, 'observation_count': len(group),
            'first_observation': group[0], 'last_observation': group[-1],
            'min_detector_confidence': min(r['confidence'] for r in group),
            'max_detector_confidence': max(r['confidence'] for r in group),
            'gaps': gaps, 'spatial_windows': windows, **_spatial_metrics(group, video),
        })
    pairs = []
    for before in tracks:
        for after in tracks:
            end, start = before['last_observation'], after['first_observation']
            gap = start['timestamp_seconds']-end['timestamp_seconds']
            if 0 < gap <= fragment_gap_seconds:
                pairs.append({
                    'from_track_id': before['local_track_id'], 'to_track_id': after['local_track_id'],
                    'elapsed_seconds': gap,
                    'unobserved_frames_between': start['frame_index']-end['frame_index']-1,
                    'center_distance_pixels': math.dist(end['center_xy'], start['center_xy']),
                    'identity_status': 'unresolved_not_merged',
                    'basis': 'temporal proximity only; could be different objects or false positives',
                })
    observed_frames = {r['frame_index'] for r in rows}
    return {
        'observation_count': len(rows), 'observed_frame_count': len(observed_frames),
        'untracked_observation_count': sum(r['local_track_id'] is None for r in rows),
        'tracks': tracks, 'possible_fragment_pairs': pairs,
        'concurrent_track_frames': _ranges(f for f, ids in frame_tracks.items() if len(ids) > 1),
        'frames_without_target_observation': _ranges(set(range(video.frame_count))-observed_frames),
        'last_observed_candidates': [r for r in rows if r['frame_index'] == rows[-1]['frame_index']] if rows else [],
        'settings': dict(assume_static_camera=assume_static_camera, window_seconds=window_seconds,
                         low_extent=low_extent, movement_extent=movement_extent,
                         minimum_span_seconds=minimum_span_seconds,
                         fragment_gap_seconds=fragment_gap_seconds),
        'review_status': 'unreviewed_detector_output',
        'limitations': [
            'Image-coordinate displacement is not proof of physical motion or supporting surface.',
            'Low extent is not proof of physical stationarity; pose, jitter and camera motion matter.',
            'Missing exports/observations do not establish object absence or processed coverage.',
            'Local IDs are scoped to this recording/export; fragments are never merged.',
            'Latest detections are not verified identity matches or the current physical location.',
            'Thresholds are exploratory heuristics; detector scores are not calibrated probabilities.',
        ],
    }
