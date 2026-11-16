"""Stage 9: Deterministic object-state analysis from YOLO/ByteTrack label outputs.

Parses frame-level YOLO tracking labels (supporting both 6-column untracked
detections and 7-column tracked detections), detects transient tracker class
flips during occlusion, computes per-track and sub-track motion states
(stationary spans vs. movement onset), measures spatial person-object bbox
overlap, flags track-fragmentation candidates without silently merging IDs,
summarizes arbitrary time windows (e.g. Stage 6/7 windows), and identifies
the last supported object sighting.

Note: Normalized bounding-box displacement alone does not prove physical
surface transfer (e.g. chair to desk) or physical object identity across gaps.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

COCO_CLASS_NAMES = {
    0: 'person',
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
    39: 'bottle',
    56: 'chair',
    63: 'laptop',
    67: 'cell phone',
}

DEFAULT_EVAL_WINDOWS = (
    (1, 0.0, 4.0),
    (2, 3.0, 7.0),
    (3, 6.0, 10.0),
)

_FRAME_NUM_RE = re.compile(r'_(\d+)$')


@dataclass(frozen=True)
class Detection:
    frame_index: int
    timestamp_seconds: float
    class_id: int
    xc: float
    yc: float
    width: float
    height: float
    confidence: float
    track_id: int | None


def parse_frame_index(stem: str) -> int:
    """Extract 1-based frame index from a YOLO label filename stem."""
    match = _FRAME_NUM_RE.search(stem)
    if not match:
        raise ValueError(f'Cannot parse frame index from label filename stem: {stem!r}')
    return int(match.group(1))


def parse_tracking_labels(
    labels_dir: Path,
    fps: float,
    class_ids: set[int] | None = None,
) -> list[Detection]:
    """Load YOLO/ByteTrack .txt label files sorted chronologically.

    Supports both:
    - 6 columns: class_id xc yc w h conf (untracked detection)
    - 7 columns: class_id xc yc w h conf track_id (tracked detection)
    """
    if fps <= 0:
        raise ValueError('fps must be positive')
    if not labels_dir.is_dir():
        raise FileNotFoundError(f'Labels directory does not exist: {labels_dir}')

    files = sorted(labels_dir.glob('*.txt'), key=lambda p: parse_frame_index(p.stem))
    detections: list[Detection] = []
    for path in files:
        frame_index = parse_frame_index(path.stem)
        timestamp = round((frame_index - 1) / fps, 3)
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            continue
        for line_no, raw in enumerate(text.splitlines(), 1):
            parts = raw.split()
            if len(parts) not in (6, 7):
                raise ValueError(
                    f'Expected 6 or 7 columns in {path.name}:{line_no}, got {len(parts)}'
                )
            cls_id = int(parts[0])
            if class_ids is not None and cls_ids_filter(cls_id, class_ids):
                continue
            xc, yc, w, h, conf = (float(v) for v in parts[1:6])
            track_id = int(parts[6]) if len(parts) == 7 else None
            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    class_id=cls_id,
                    xc=xc,
                    yc=yc,
                    width=w,
                    height=h,
                    confidence=conf,
                    track_id=track_id,
                )
            )
    return detections


def cls_ids_filter(cls_id: int, allowed: set[int]) -> bool:
    return cls_id not in allowed


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def bbox_iou(a: Detection, b: Detection) -> float:
    """Compute normalized bounding-box IoU between two detections."""
    ax1, ay1 = a.xc - a.width / 2.0, a.yc - a.height / 2.0
    ax2, ay2 = a.xc + a.width / 2.0, a.yc + a.height / 2.0
    bx1, by1 = b.xc - b.width / 2.0, b.yc - b.height / 2.0
    bx2, by2 = b.xc + b.width / 2.0, b.yc + b.height / 2.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = (a.width * a.height) + (b.width * b.height) - inter
    return inter / union if union > 0.0 else 0.0


def _classify_motion(
    net_displacement: float,
    max_anchor_drift: float,
    observation_count: int,
) -> str:
    if observation_count < 2:
        return 'single_observation'
    if net_displacement >= 0.04 or max_anchor_drift >= 0.05:
        return 'movement_candidate'
    if net_displacement <= 0.025 and max_anchor_drift <= 0.03:
        return 'stationary'
    return 'uncertain_motion'


def _detect_motion_onset(
    ordered: list[Detection],
    drift_threshold: float = 0.025,
) -> dict | None:
    """If a track starts stationary and later drifts from its initial anchor, report the split."""
    if len(ordered) < 4:
        return None
    anchor = (ordered[0].xc, ordered[0].yc)
    for idx, det in enumerate(ordered):
        drift = _euclidean(anchor, (det.xc, det.yc))
        if drift >= drift_threshold:
            if idx == 0:
                return None
            prev = ordered[idx - 1]
            return {
                'stationary_until_frame': prev.frame_index,
                'stationary_until_seconds': prev.timestamp_seconds,
                'movement_onset_frame': det.frame_index,
                'movement_onset_seconds': det.timestamp_seconds,
                'anchor_center': [round(anchor[0], 4), round(anchor[1], 4)],
                'drift_at_onset': round(drift, 4),
            }
    return None


def summarize_observation_sequence(
    class_id: int,
    track_id: int | None,
    observations: list[Detection],
    class_flip_frames: list[int] | None = None,
) -> dict:
    """Compute deterministic spatial, temporal, and confidence metrics for a sequence."""
    ordered = sorted(observations, key=lambda d: d.frame_index)
    first = ordered[0]
    last = ordered[-1]
    frame_span = (last.frame_index - first.frame_index) + 1
    coverage_ratio = round(len(ordered) / frame_span, 4)

    centers = [(d.xc, d.yc) for d in ordered]
    steps = [_euclidean(centers[i - 1], centers[i]) for i in range(1, len(centers))]
    anchor_drifts = [_euclidean(centers[0], c) for c in centers]
    net_disp = _euclidean(centers[0], centers[-1])
    total_path = sum(steps)
    max_step = max(steps) if steps else 0.0
    max_drift = max(anchor_drifts) if anchor_drifts else 0.0

    confs = [d.confidence for d in ordered]
    mean_conf = sum(confs) / len(confs)

    present_frames = {d.frame_index for d in ordered}
    missing_frames = [
        f for f in range(first.frame_index, last.frame_index + 1) if f not in present_frames
    ]

    motion_state = _classify_motion(net_disp, max_drift, len(ordered))
    onset = _detect_motion_onset(ordered) if motion_state == 'movement_candidate' else None

    result = {
        'class_id': class_id,
        'class_name': COCO_CLASS_NAMES.get(class_id, f'class_{class_id}'),
        'track_id': track_id,
        'start_frame': first.frame_index,
        'end_frame': last.frame_index,
        'start_seconds': first.timestamp_seconds,
        'end_seconds': last.timestamp_seconds,
        'observation_count': len(ordered),
        'frame_span': frame_span,
        'coverage_ratio': coverage_ratio,
        'internal_missing_frames': len(missing_frames),
        'mean_confidence': round(mean_conf, 4),
        'max_confidence': round(max(confs), 4),
        'start_center': [round(first.xc, 4), round(first.yc, 4)],
        'end_center': [round(last.xc, 4), round(last.yc, 4)],
        'net_displacement': round(net_disp, 4),
        'max_anchor_drift': round(max_drift, 4),
        'total_path_length': round(total_path, 4),
        'max_step_displacement': round(max_step, 4),
        'motion_state': motion_state,
    }
    if onset is not None:
        result['motion_phase_split'] = onset
    if class_flip_frames:
        result['class_flip_frames'] = sorted(class_flip_frames)
    return result


def group_tracks_with_class_flip_audit(
    detections: list[Detection],
    target_class_id: int,
) -> tuple[list[dict], list[dict], list[Detection]]:
    """Group tracked detections while auditing transient ByteTrack class flips.

    In ByteTrack, a single track_id can briefly flip its class label during
    occlusion (e.g. track 17 in test01 is class 24 for 38 frames and class 0
    for 2 low-confidence frames 76..77 at the exact same bounding box).
    Only merge a minority-class observation into a dominant-class track when the
    boxes in adjacent frames spatially overlap (IoU >= 0.5); if two distinct
    objects in the same frame share a numeric track_id across classes, keep them
     strictly separated by (class_id, track_id).
    """
    by_track_id: dict[int, list[Detection]] = defaultdict(list)
    untracked_target: list[Detection] = []

    for det in detections:
        if det.track_id is None:
            if det.class_id == target_class_id:
                untracked_target.append(det)
        else:
            by_track_id[det.track_id].append(det)

    target_tracks: list[dict] = []
    person_tracks: list[dict] = []

    for tid, obs in sorted(by_track_id.items()):
        by_cls: dict[int, list[Detection]] = defaultdict(list)
        for d in obs:
            by_cls[d.class_id].append(d)

        if len(by_cls) == 1:
            cls_id = next(iter(by_cls))
            summary = summarize_observation_sequence(cls_id, tid, obs)
            if cls_id == target_class_id:
                target_tracks.append(summary)
            elif cls_id == 0:
                person_tracks.append(summary)
            continue

        # Multiple classes appeared under the same numeric track_id.
        # Check if they occur on disjoint frames with high spatial IoU (true class flip)
        # versus simultaneous different objects in the same frame.
        frame_counts = Counter(d.frame_index for d in obs)
        has_same_frame_collision = any(c > 1 for c in frame_counts.values())

        cls_counts = Counter(d.class_id for d in obs)
        dominant_cls, dominant_count = cls_counts.most_common(1)[0]
        minority_obs = [d for d in obs if d.class_id != dominant_cls]
        dominant_obs = [d for d in obs if d.class_id == dominant_cls]

        is_spatial_continuation = False
        if not has_same_frame_collision and dominant_obs and minority_obs:
            # Verify minority detections overlap spatially with nearest dominant detection
            nearest_ious = []
            for m in minority_obs:
                nearest = min(dominant_obs, key=lambda d: abs(d.frame_index - m.frame_index))
                nearest_ious.append(bbox_iou(m, nearest))
            is_spatial_continuation = min(nearest_ious) >= 0.5

        if is_spatial_continuation and dominant_count >= len(minority_obs):
            flip_frames = [d.frame_index for d in minority_obs]
            # Keep only target-class detections for primary trajectory metrics, or
            # include flip frames in audit list while summarizing dominant observations.
            summary = summarize_observation_sequence(
                dominant_cls, tid, dominant_obs, class_flip_frames=flip_frames
            )
            if dominant_cls == target_class_id:
                target_tracks.append(summary)
            elif dominant_cls == 0:
                person_tracks.append(summary)
        else:
            # Separate tracks that happen to share a numeric ID across classes
            for cls_id, cls_obs in sorted(by_cls.items()):
                summary = summarize_observation_sequence(cls_id, tid, cls_obs)
                if cls_id == target_class_id:
                    target_tracks.append(summary)
                elif cls_id == 0:
                    person_tracks.append(summary)

    return target_tracks, person_tracks, untracked_target


def compute_person_object_interactions(
    detections: list[Detection],
    target_class_id: int,
    fps: float,
) -> dict:
    """Identify frames where a person bounding box spatially overlaps the target object."""
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in detections:
        by_frame[d.frame_index].append(d)

    overlap_frames: list[dict] = []
    for frame_idx, items in sorted(by_frame.items()):
        persons = [d for d in items if d.class_id == 0]
        targets = [d for d in items if d.class_id == target_class_id]
        for p in persons:
            for t in targets:
                if p.track_id is not None and p.track_id == t.track_id:
                    continue
                iou = bbox_iou(p, t)
                if iou > 0.0:
                    overlap_frames.append({
                        'frame_index': frame_idx,
                        'timestamp_seconds': round((frame_idx - 1) / fps, 3),
                        'person_track_id': p.track_id,
                        'target_track_id': t.track_id,
                        'bbox_iou': round(iou, 4),
                    })

    if not overlap_frames:
        return {
            'overlap_frame_count': 0,
            'first_overlap_seconds': None,
            'last_overlap_seconds': None,
            'person_track_ids': [],
        }

    person_ids = sorted({r['person_track_id'] for r in overlap_frames if r['person_track_id'] is not None})
    return {
        'overlap_frame_count': len(overlap_frames),
        'first_overlap_frame': overlap_frames[0]['frame_index'],
        'last_overlap_frame': overlap_frames[-1]['frame_index'],
        'first_overlap_seconds': overlap_frames[0]['timestamp_seconds'],
        'last_overlap_seconds': overlap_frames[-1]['timestamp_seconds'],
        'max_bbox_iou': max(r['bbox_iou'] for r in overlap_frames),
        'person_track_ids': person_ids,
    }


def _tracks_overlap_window(
    tracks: list[dict],
    start_frame: int,
    end_frame: int,
) -> list[int]:
    return [
        t['track_id']
        for t in tracks
        if t['start_frame'] <= end_frame and t['end_frame'] >= start_frame
    ]


def find_gap_candidates(
    target_tracks: list[dict],
    person_tracks: list[dict],
    fps: float,
    max_gap_seconds: float = 3.0,
) -> list[dict]:
    """Identify gaps between sequential same-class tracks and flag fragment links."""
    ordered = sorted(target_tracks, key=lambda t: (t['start_frame'], t['end_frame']))
    gaps: list[dict] = []
    for idx in range(1, len(ordered)):
        prev_t = ordered[idx - 1]
        next_t = ordered[idx]
        if prev_t['class_id'] != next_t['class_id']:
            continue
        gap_frames = next_t['start_frame'] - prev_t['end_frame'] - 1
        if gap_frames < 0:
            gaps.append({
                'kind': 'overlapping_tracks',
                'prev_track_id': prev_t['track_id'],
                'next_track_id': next_t['track_id'],
                'gap_frames': gap_frames,
                'gap_seconds': round(gap_frames / fps, 3),
                'spatial_jump': round(
                    _euclidean(tuple(prev_t['end_center']), tuple(next_t['start_center'])), 4
                ),
                'overlapping_person_track_ids': _tracks_overlap_window(
                    person_tracks, next_t['start_frame'], prev_t['end_frame']
                ),
                'note': 'Tracks overlap in time; do not assume single object.',
            })
            continue

        gap_seconds = round(gap_frames / fps, 3)
        spatial_jump = round(
            _euclidean(tuple(prev_t['end_center']), tuple(next_t['start_center'])), 4
        )
        person_ids = _tracks_overlap_window(
            person_tracks, prev_t['end_frame'], next_t['start_frame']
        )
        is_candidate = gap_seconds <= max_gap_seconds
        gaps.append({
            'kind': 'fragment_link_candidate' if is_candidate else 'long_absence_gap',
            'prev_track_id': prev_t['track_id'],
            'next_track_id': next_t['track_id'],
            'gap_start_frame': prev_t['end_frame'] + 1,
            'gap_end_frame': next_t['start_frame'] - 1,
            'gap_frames': gap_frames,
            'gap_seconds': gap_seconds,
            'spatial_jump': spatial_jump,
            'overlapping_person_track_ids': person_ids,
            'note': (
                'Same-class track gap with concurrent person presence; candidate for visual re-association.'
                if is_candidate and person_ids
                else 'Same-class track gap; visual check required before linking.'
            ),
        })
    return gaps


def summarize_time_window(
    detections: list[Detection],
    target_class_id: int,
    start_seconds: float,
    end_seconds: float,
    window_number: int | None = None,
) -> dict:
    """Summarize observed target state within a specific [start_seconds, end_seconds] window."""
    in_win = [
        d for d in detections
        if start_seconds - 1e-6 <= d.timestamp_seconds <= end_seconds + 1e-6
    ]
    target_obs = [d for d in in_win if d.class_id == target_class_id]
    person_obs = [d for d in in_win if d.class_id == 0]

    if not target_obs:
        return {
            'window_number': window_number,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'target_observation_count': 0,
            'track_ids': [],
            'motion_state': 'not_observed',
            'net_displacement': None,
            'person_bbox_overlap_frames': 0,
        }

    ordered = sorted(target_obs, key=lambda d: d.frame_index)
    centers = [(d.xc, d.yc) for d in ordered]
    net_disp = _euclidean(centers[0], centers[-1])
    max_drift = max(_euclidean(centers[0], c) for c in centers)
    track_ids = sorted({d.track_id for d in ordered if d.track_id is not None})

    # Count frames in this window where a person bbox overlaps the target bbox
    by_f: dict[int, list[Detection]] = defaultdict(list)
    for d in in_win:
        by_f[d.frame_index].append(d)
    overlap_count = 0
    for f_items in by_f.values():
        ps = [d for d in f_items if d.class_id == 0]
        ts = [d for d in f_items if d.class_id == target_class_id]
        if any(
            (p.track_id != t.track_id or p.track_id is None) and bbox_iou(p, t) > 0.0
            for p in ps for t in ts
        ):
            overlap_count += 1

    return {
        'window_number': window_number,
        'start_seconds': start_seconds,
        'end_seconds': end_seconds,
        'target_observation_count': len(ordered),
        'untracked_count': sum(1 for d in ordered if d.track_id is None),
        'track_ids': track_ids,
        'start_center': [round(ordered[0].xc, 4), round(ordered[0].yc, 4)],
        'end_center': [round(ordered[-1].xc, 4), round(ordered[-1].yc, 4)],
        'net_displacement': round(net_disp, 4),
        'max_anchor_drift': round(max_drift, 4),
        'motion_state': _classify_motion(net_disp, max_drift, len(ordered)),
        'person_present_frames': len({d.frame_index for d in person_obs}),
        'person_bbox_overlap_frames': overlap_count,
    }


def find_last_supported_sighting(
    target_tracks: list[dict],
    min_observations: int = 5,
    min_mean_confidence: float = 0.25,
) -> dict | None:
    """Select the latest track meeting minimum support thresholds."""
    supported = [
        t for t in target_tracks
        if t['observation_count'] >= min_observations
        and t['mean_confidence'] >= min_mean_confidence
    ]
    if not supported:
        return None
    latest = max(supported, key=lambda t: (t['end_frame'], t['observation_count']))
    return {
        'track_id': latest['track_id'],
        'class_id': latest['class_id'],
        'class_name': latest['class_name'],
        'last_frame': latest['end_frame'],
        'last_timestamp_seconds': latest['end_seconds'],
        'last_center': latest['end_center'],
        'observation_count': latest['observation_count'],
        'mean_confidence': latest['mean_confidence'],
        'motion_state': latest['motion_state'],
        'caveat': (
            'Last observed sighting in recorded footage; does not guarantee '
            'current physical location after recording ends.'
        ),
    }


def analyze_object_state(
    labels_dir: Path,
    fps: float,
    target_class_id: int = 24,
    min_observations: int = 5,
    min_mean_confidence: float = 0.25,
    eval_windows: tuple[tuple[int, float, float], ...] = DEFAULT_EVAL_WINDOWS,
) -> dict:
    """Run full Stage 9 object-state analysis for one target class."""
    detections = parse_tracking_labels(
        labels_dir=labels_dir,
        fps=fps,
        class_ids={target_class_id, 0},
    )

    target_tracks, person_tracks, untracked_target = group_tracks_with_class_flip_audit(
        detections, target_class_id=target_class_id
    )

    untracked_summary = (
        summarize_observation_sequence(target_class_id, None, untracked_target)
        if untracked_target else None
    )
    if untracked_summary and target_tracks:
        first_track = target_tracks[0]
        jump = _euclidean(
            tuple(untracked_summary['end_center']),
            tuple(first_track['start_center']),
        )
        untracked_summary['distance_to_first_track'] = round(jump, 4)
        untracked_summary['first_track_id'] = first_track['track_id']
        untracked_summary['pre_track_continuation_candidate'] = jump <= 0.03

    # Check if any single frame had multiple detections of the target class
    target_frame_counts = Counter(
        d.frame_index for d in detections if d.class_id == target_class_id
    )
    multi_detection_frames = sorted(
        f for f, count in target_frame_counts.items() if count > 1
    )

    gaps = find_gap_candidates(target_tracks, person_tracks, fps=fps)
    interactions = compute_person_object_interactions(
        detections, target_class_id=target_class_id, fps=fps
    )
    window_summaries = [
        summarize_time_window(detections, target_class_id, start_s, end_s, window_number=w_num)
        for w_num, start_s, end_s in eval_windows
    ]
    last_supported = find_last_supported_sighting(
        target_tracks,
        min_observations=min_observations,
        min_mean_confidence=min_mean_confidence,
    )

    return {
        'report_format': 'group8-stage9-object-state-v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'labels_dir': str(labels_dir),
        'fps': fps,
        'target_class_id': target_class_id,
        'target_class_name': COCO_CLASS_NAMES.get(target_class_id, f'class_{target_class_id}'),
        'untracked_target_detections': len(untracked_target),
        'untracked_summary': untracked_summary,
        'multi_detection_frames': multi_detection_frames,
        'target_tracks': target_tracks,
        'person_tracks': person_tracks,
        'person_object_interactions': interactions,
        'gap_and_fragment_analysis': gaps,
        'window_state_summaries': window_summaries,
        'last_supported_sighting': last_supported,
        'limitations': [
            'Normalized bounding-box displacement does not establish physical surface support (e.g., chair vs desk).',
            'Track IDs across gaps are flagged as fragment_link_candidate rather than silently merged.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir', type=Path, required=True, help='YOLO tracking labels directory')
    parser.add_argument('--fps', type=float, default=24.0, help='Constant video frame rate')
    parser.add_argument('--class-id', type=int, default=24, help='Target COCO class ID (default: 24=backpack)')
    parser.add_argument('--min-observations', type=int, default=5)
    parser.add_argument('--min-mean-confidence', type=float, default=0.25)
    parser.add_argument('--out', type=Path, default=None, help='Optional JSON output path')
    args = parser.parse_args()

    report = analyze_object_state(
        labels_dir=args.labels_dir,
        fps=args.fps,
        target_class_id=args.class_id,
        min_observations=args.min_observations,
        min_mean_confidence=args.min_mean_confidence,
    )

    print(f"Target: {report['target_class_name']} (class {report['target_class_id']}) @ {report['fps']} fps")
    u = report['untracked_summary']
    if u:
        print(
            f"Untracked pre-track cluster: frames {u['start_frame']}..{u['end_frame']} "
            f"({u['start_seconds']:.2f}s..{u['end_seconds']:.2f}s), n={u['observation_count']}, "
            f"conf={u['mean_confidence']:.3f}, disp={u['net_displacement']:.4f}, state={u['motion_state']}"
        )
    for t in report['target_tracks']:
        split = t.get('motion_phase_split')
        split_note = (
            f" [stationary until {split['stationary_until_seconds']:.2f}s, "
            f"moves at {split['movement_onset_seconds']:.2f}s]"
            if split else ''
        )
        flips = f" [class_flips={t['class_flip_frames']}]" if t.get('class_flip_frames') else ''
        print(
            f"  Track {t['track_id']}: frames {t['start_frame']}..{t['end_frame']} "
            f"({t['start_seconds']:.2f}s..{t['end_seconds']:.2f}s), n={t['observation_count']}, "
            f"conf={t['mean_confidence']:.3f}, disp={t['net_displacement']:.4f}, "
            f"state={t['motion_state']}{split_note}{flips}"
        )
    inter = report['person_object_interactions']
    if inter['overlap_frame_count'] > 0:
        print(
            f"Person-object bbox overlap: {inter['overlap_frame_count']} frames "
            f"({inter['first_overlap_seconds']:.2f}s..{inter['last_overlap_seconds']:.2f}s, "
            f"max_iou={inter['max_bbox_iou']:.3f}, person_tracks={inter['person_track_ids']})"
        )
    for g in report['gap_and_fragment_analysis']:
        print(
            f"  Gap {g['prev_track_id']} -> {g['next_track_id']}: {g['kind']}, "
            f"{g['gap_frames']} frames ({g['gap_seconds']:.2f}s), jump={g['spatial_jump']:.4f}, "
            f"person_overlap={g['overlapping_person_track_ids']}"
        )
    print("Stage 6/7 window state summaries:")
    for w in report['window_state_summaries']:
        print(
            f"  Window {w['window_number']} ({w['start_seconds']:.1f}–{w['end_seconds']:.1f}s): "
            f"n={w['target_observation_count']}, tracks={w['track_ids']}, "
            f"disp={w['net_displacement']}, state={w['motion_state']}, "
            f"person_bbox_overlap_frames={w['person_bbox_overlap_frames']}"
        )
    last = report['last_supported_sighting']
    if last:
        print(
            f"Last supported sighting: track {last['track_id']} at frame {last['last_frame']} "
            f"({last['last_timestamp_seconds']:.2f}s), center={last['last_center']}, state={last['motion_state']}"
        )
    else:
        print('Last supported sighting: NONE (no track met support thresholds)')

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Wrote report to {args.out}')


if __name__ == '__main__':
    main()
