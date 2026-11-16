"""Read saved YOLO/ByteTrack labels and create a fresh Stage 9 report directory.

Run from the repository root: python -m src.stage9 --help
No detector/VLM loading, database access or edits to previous experiments.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from src.object_state import VideoInfo, analyze_observations, load_yolo_labels


def run(video_path: Path, labels_dir: Path, output_dir: Path, camera_id: str,
        video_id: str, assume_static_camera: bool = False, *, class_id: int = 24,
        **analysis_options) -> dict:
    """Import a confirmed save_conf=True, stride-one CFR export of one video.

    Format/timing assumptions are part of this Python API's input contract.
    The CLI requires explicit flags because the old export has no run manifest.
    """
    import cv2  # Metadata only; keep pure analysis usable without OpenCV.

    video_path, labels_dir, output_dir = map(Path, (video_path, labels_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Output directory already exists; select a fresh directory')
    if not camera_id.strip() or not video_id.strip() or class_id < 0:
        raise ValueError('Nonempty camera/video IDs and a nonnegative class ID are required')
    if not video_path.is_file():
        raise FileNotFoundError(f'Video does not exist: {video_path}')
    cap = cv2.VideoCapture(str(video_path.resolve()))
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be opened')
        video = VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
    imported = load_yolo_labels(labels_dir, video_path.stem, video, class_id)
    with video_path.open('rb') as f:
        video_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    scope = hashlib.sha256(json.dumps([camera_id, video_id, video_hash,
                                      imported['labels_sha256']]).encode()).hexdigest()
    rows = imported.pop('observations')
    for row in rows:
        track_id = row['local_track_id']
        row.update(camera_id=camera_id, video_id=video_id, scope_id=scope,
                   review_status='unreviewed_detector_output',
                   scoped_track_key=f'{scope}:class_{class_id}:track_{track_id}' if track_id is not None else None)
    analysis = analyze_observations(rows, video, assume_static_camera=assume_static_camera,
                                    **analysis_options)
    report = {
        'report_format': 'group8-stage9-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'camera_id': camera_id, 'video_id': video_id, 'target_class_id': class_id,
        'database_updated': False, 'video': asdict(video),
        'provenance': {
            'scope_id': scope, 'video_path': str(video_path.resolve()),
            'video_sha256': video_hash, 'labels_directory': str(labels_dir.resolve()),
            **imported,
            'label_format': 'class cx cy width height confidence [track_id]',
            'time_basis': '(one-based label suffix - 1) / FPS; caller-confirmed CFR and stride=1',
            'source_association': 'caller-supplied video matched by filename; original export hash unavailable',
            'original_tracking_settings': 'not recorded in saved export; do not infer from current packages',
            'importer_versions': {'python': platform.python_version(), 'opencv': version('opencv-python')},
            'analysis_code_sha256': {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('object_state.py', 'stage9.py')
            },
        },
        'analysis': analysis,
    }
    # Complete validation before creating outputs. Existing inputs/outputs are never removed.
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'observations.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    fields = ['frame_index', 'timestamp_seconds', 'local_track_id', 'confidence',
              'bbox_xyxy', 'source_label', 'source_line', 'human_object_id',
              'human_bbox_xyxy', 'human_surface', 'human_visible', 'human_motion',
              'reviewer', 'notes']
    with (output_dir / 'human_review.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields if key in row})
    # Written last: analysis.json is the marker of a completed file report.
    (output_dir / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage9') /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--class-id', type=int, default=24, help='COCO backpack is 24')
    parser.add_argument('--confirm-save-conf', action='store_true', required=True,
                        help='Confirm labels include confidence (not confidence-free tracking rows)')
    parser.add_argument('--confirm-cfr-stride-one', action='store_true', required=True,
                        help='Confirm constant FPS, full-video export and vid_stride=1')
    parser.add_argument('--assume-static-camera', action='store_true',
                        help='Explicit caller assumption; no camera-motion estimation is performed')
    parser.add_argument('--window-seconds', type=float, default=1.0)
    parser.add_argument('--low-extent', type=float, default=.01,
                        help='Low displacement envelope / image diagonal; uncalibrated heuristic')
    parser.add_argument('--movement-extent', type=float, default=.03)
    parser.add_argument('--minimum-span-seconds', type=float, default=.5)
    parser.add_argument('--fragment-gap-seconds', type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        report = run(args.video, args.labels, args.output, args.camera, args.video_id,
                     args.assume_static_camera, class_id=args.class_id,
                     window_seconds=args.window_seconds, low_extent=args.low_extent,
                     movement_extent=args.movement_extent,
                     minimum_span_seconds=args.minimum_span_seconds,
                     fragment_gap_seconds=args.fragment_gap_seconds)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    result = report['analysis']
    print(f"{result['observation_count']} observations; {len(result['tracks'])} separate tracks; "
          f"{result['untracked_observation_count']} untracked detections")
    for track in result['tracks']:
        first, last = track['first_observation'], track['last_observation']
        print(f"  Track {track['local_track_id']}: {track['observation_count']} observations, "
              f"{first['timestamp_seconds']:.6f}-{last['timestamp_seconds']:.6f}s; "
              f"{len(track['gaps'])} internal gaps")
    for row in result['last_observed_candidates']:
        print(f"  Latest detector candidate: t={row['timestamp_seconds']:.6f}s, "
              f"frame={row['frame_index']}, ID={row['local_track_id']}, bbox={row['bbox_xyxy']}")
    print(f'Report: {args.output.resolve() / "analysis.json"}')
    print('Unreviewed image-space candidates only. Review human_review.csv against the video.')
    print('Database unchanged; no identity merge or physical surface inferred.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
