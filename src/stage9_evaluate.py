"""Evaluate saved backpack predictions against certain, latest human frame reviews.

No models, automatic ground-truth labels or database writes. This evaluates a
reviewed sample, not all footage or general model accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def _box(values):
    if len(values) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError('Invalid finite bounding box')
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError('Invalid bounding box bounds')
    return values


def _iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


def _match(predictions, truth, threshold):
    """Maximum-cardinality bipartite matching, deterministic IoU preferences.

    Unlike greedy assignment, augmenting paths can move a flexible match so a
    second valid detection is not needlessly counted as a miss. Does not claim
    maximum total IoU among all equally sized assignments.
    """
    edges = {p: sorted((t for t in range(len(truth))
                        if _iou(predictions[p]['bbox_xyxy'], truth[t]['bbox_xyxy']) >= threshold),
                       key=lambda t: (-_iou(predictions[p]['bbox_xyxy'], truth[t]['bbox_xyxy']), t))
             for p in range(len(predictions))}
    owners = {}

    def augment(p, seen):
        for t in edges[p]:
            if t in seen:
                continue
            seen.add(t)
            if t not in owners or augment(owners[t], seen):
                owners[t] = p
                return True
        return False

    for p in sorted(edges, key=lambda p: (
            -max((_iou(predictions[p]['bbox_xyxy'], truth[t]['bbox_xyxy']) for t in edges[p]), default=0),
            len(edges[p]), p)):
        augment(p, set())
    return owners


def evaluate_observations(predictions, annotations, frame_count, *, iou_threshold=.5):
    if (type(frame_count) is not int or frame_count < 1 or
            not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1):
        raise ValueError('Positive frame count and IoU threshold in (0,1] required')
    latest, revisions = {}, set()
    for row in annotations:
        f, revision = row['frame_index'], row['revision']
        if type(f) is not int or not 0 <= f < frame_count or type(revision) is not int or revision < 1:
            raise ValueError('Invalid annotation frame/revision')
        if (f, revision) in revisions:
            raise ValueError('Duplicate annotation revision')
        revisions.add((f, revision))
        if row['visibility'] not in ('present', 'absent', 'uncertain'):
            raise ValueError('Invalid annotation visibility')
        objects = row['objects']
        if ((row['visibility'] == 'present' and not objects) or
                (row['visibility'] == 'absent' and objects)):
            raise ValueError('Visibility and human boxes conflict')
        ids = [o['object_id'] for o in objects]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError('Manual object IDs must be unique and nonempty')
        for obj in objects:
            _box(obj['bbox_xyxy'])
        if f not in latest or revision > latest[f]['revision']:
            latest[f] = row
    grouped = defaultdict(list)
    for row in predictions:
        f = row['frame_index']
        if type(f) is not int or not 0 <= f < frame_count:
            raise ValueError('Prediction frame outside recording')
        _box(row['bbox_xyxy'])
        grouped[f].append(row)
    certain = {f: row for f, row in latest.items() if row['visibility'] != 'uncertain'}
    tp = fp = fn = switches = gap_starts = untracked = 0
    matched_ious, frame_results, continuity = [], [], {}
    for f, row in sorted(certain.items()):
        predicted, objects = grouped[f], row['objects']
        matches = _match(predicted, objects, iou_threshold)
        frame_tp = len(matches)
        tp += frame_tp
        fp += len(predicted)-frame_tp
        fn += len(objects)-frame_tp
        for t, obj in enumerate(objects):
            p = matches.get(t)
            track = predicted[p]['local_track_id'] if p is not None else None
            if p is not None:
                matched_ious.append(_iou(predicted[p]['bbox_xyxy'], obj['bbox_xyxy']))
                untracked += track is None
            old = continuity.get(obj['object_id'])
            if old and old[0] == f-1 and old[1] is not None:
                switches += track is not None and old[1] != track
                gap_starts += track is None
            continuity[obj['object_id']] = (f, track)
        frame_results.append(dict(frame_index=f, true_positives=frame_tp,
                                  false_positives=len(predicted)-frame_tp,
                                  false_negatives=len(objects)-frame_tp))
    return dict(
        status='evaluated_reviewed_sample' if certain else 'awaiting_review',
        reviewed_frame_count=len(latest), evaluated_frame_count=len(certain),
        uncertain_frame_count=len(latest)-len(certain), unreviewed_frame_count=frame_count-len(latest),
        iou_threshold=iou_threshold, true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        mean_matched_iou=sum(matched_ious)/len(matched_ious) if matched_ious else None,
        adjacent_reviewed_identity_switches=switches,
        adjacent_reviewed_track_gap_starts=gap_starts, matched_untracked_detections=untracked,
        motion_evaluation='not_evaluated_requires_interval_ground_truth', frames=frame_results,
        limitations=[
            'Only latest certain frame reviews are scored; unreviewed/uncertain frames are excluded.',
            'Absent means no visible target in the reviewed image, not physical absence.',
            'Precision/recall describe this reviewed sample, not overall model accuracy.',
            'Manual IDs must be visually justified; continuity counts use adjacent reviewed frames only.',
            'Continuity counts are not standard IDF1/MOTA or a validated trajectory reconstruction.',
            'Per-frame motion context cannot establish interval-level movement ground truth.',
        ])


def evaluate_report(report_path, review_db, *, iou_threshold=.5, include_assisted=False):
    report_path, review_db = Path(report_path).resolve(), Path(review_db).resolve()
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get('report_format') != 'group8-stage9-v1' or report.get('target_class_id') != 24:
        raise ValueError('Expected a Stage 9 backpack report')
    scope = report['provenance']['scope_id']
    observation_bytes = report_path.with_name('observations.jsonl').read_bytes()
    predictions = [json.loads(line) for line in observation_bytes.splitlines() if line.strip()]
    if (len(predictions) != report['analysis']['observation_count'] or
            any(p.get('scope_id') != scope or p.get('class_id') != 24 for p in predictions)):
        raise ValueError('Predictions do not match report')
    with closing(sqlite3.connect(review_db.as_uri()+'?mode=ro', uri=True)) as conn:
        metadata = conn.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
        if not metadata or metadata[0] != scope:
            raise ValueError('Review store belongs to another report')
        annotations = [json.loads(row[0]) for row in conn.execute('SELECT payload FROM reviews ORDER BY frame,revision')]
    if any(a.get('scope_id') != scope or a.get('annotation_source') not in
           ('human_review', 'ai_assisted_visual_review') for a in annotations):
        raise ValueError('Annotation provenance mismatch')
    # Select latest before filtering: an assisted revision must not resurrect an
    # older human label and silently present it as the latest reviewed state.
    latest = {}
    for a in annotations:
        if a['frame_index'] not in latest or a['revision'] > latest[a['frame_index']]['revision']:
            latest[a['frame_index']] = a
    assisted_count = sum(a['annotation_source'] == 'ai_assisted_visual_review' for a in latest.values())
    selected = [a for a in latest.values() if include_assisted or a['annotation_source'] == 'human_review']
    result = evaluate_observations(predictions, selected, report['video']['frame_count'], iou_threshold=iou_threshold)
    result['excluded_assisted_frame_count'] = 0 if include_assisted else assisted_count
    result['reference_type'] = 'includes_ai_assisted_reference' if include_assisted else 'human_review_only'
    if include_assisted:
        result['limitations'].append('AI-assisted box comparisons are diagnostic and are not independent human-ground-truth accuracy.')
    result['provenance'] = dict(scope_id=scope, report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        observations_sha256=hashlib.sha256(observation_bytes).hexdigest(),
        review_history_sha256=hashlib.sha256(json.dumps(annotations, sort_keys=True).encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--review-db', type=Path)
    parser.add_argument('--iou-threshold', type=float, default=.5)
    parser.add_argument('--include-assisted', action='store_true', help='Diagnostic comparison only; includes AI-assisted labels')
    parser.add_argument('--output', type=Path, help='Optional new JSON output; existing files rejected')
    args = parser.parse_args()
    try:
        result = evaluate_report(args.report, args.review_db or args.report.parent/'human_review.sqlite3',
                                 iou_threshold=args.iou_threshold, include_assisted=args.include_assisted)
        if args.output:
            with args.output.open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2, allow_nan=False)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
