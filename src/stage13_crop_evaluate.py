"""Exploratory object-crop comparison on human-verified query labels.

Relevance uses the Stage13 overlap coefficient; MRR is candidate bounded.
Small same-camera collections cannot establish generalization. Verified negatives
measure candidate return only: there is no calibrated no-match threshold or
negative accuracy claim. Proposed absence labels are never scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from src.stage13_evaluate import validate_manifest, score_query

MODES = {'full_frame': 'frame', 'crop_only': 'crop', 'combined': None}
SETTINGS = dict(visual_weight=1, text_weight=0, window_seconds=4, rrf_k=60,
                candidate_limit=100, limit=100)


def ranked_signature(results):
    """Compare ranked windows, independent of database row IDs and paths."""
    return [(h['video_id'], h['start_seconds'], h['end_seconds']) for h in results]


def evaluate(manifest, *, search_fn, baseline_report, annotations_sha256):
    clips = validate_manifest(manifest)
    if baseline_report.get('annotations_sha256') != annotations_sha256:
        raise ValueError('Baseline annotation SHA256 differs from loaded annotations')
    verified = [q for q in manifest['queries'] if q['review_status'] == 'verified']
    if not verified:
        raise ValueError('Crop comparison requires verified queries')
    prior = {q['query_id']: q for q in baseline_report['queries']}
    if set(prior) != {q['query_id'] for q in verified}:
        raise ValueError('Baseline verified query set differs')
    for q in verified:
        old = prior[q['query_id']]
        if old['query'] != q['query'] or old['present'] != q['present']:
            raise ValueError('Baseline query differs')
        if old['modes']['visual']['status'] != 'evaluated':
            raise ValueError('Baseline visual query was not evaluated')
    report = dict(report_format='group8-stage13-crop-evaluation-v1', status='evaluated',
                  annotations_sha256=annotations_sha256, queries=[], aggregates={}, warmup_seconds={},
                  skipped_queries=[dict(query_id=q['query_id'], reason='Proposed labels are not scored')
                                   for q in manifest['queries'] if q['review_status'] != 'verified'],
                  limitations=__doc__, metric_definition='src.stage13_evaluate.score_query: overlap coefficient >= 0.5; fixed P@5 denominator; deduplicated windows.',
                  latency_protocol='Each mode warmed once with first verified query; timed warm retrieval excludes indexing.',
                  mrr_protocol='At most 100 raw visual candidates before temporal window collapse; not exhaustive recall.',
                  indexing_time_seconds=None)
    for mode, variant in MODES.items():
        start = time.perf_counter()
        search_fn(verified[0]['query'], variant=variant, **SETTINGS)
        report['warmup_seconds'][mode] = time.perf_counter() - start
    for q in verified:
        entry = dict(query_id=q['query_id'], query=q['query'], present=q['present'], modes={})
        for mode, variant in MODES.items():
            start = time.perf_counter()
            found = search_fn(q['query'], variant=variant, **SETTINGS)
            elapsed = time.perf_counter() - start
            for hit in found['results']:
                clip = clips.get(hit['video_id'])
                if clip is None or hit['end_seconds'] > clip['duration_seconds']:
                    raise ValueError('Prediction outside evaluation clip')
                if hit.get('source_sha256', clip['source_sha256']) != clip['source_sha256']:
                    raise ValueError('Prediction source differs from annotation clip')
            metrics = score_query(q, found['results'])
            entry['modes'][mode] = dict(status='evaluated', metrics=metrics,
                first_relevant_rank=round(1 / metrics['mrr']) if metrics.get('mrr') else None,
                latency_seconds=elapsed, latency_label='warm', candidate_counts=found['candidate_counts'],
                results=found['results'], ranked_window_signature=ranked_signature(found['results']))
        actual = entry['modes']['full_frame']
        expected = prior[q['query_id']]['modes']['visual']
        if actual['metrics'] != expected['metrics']:
            raise ValueError('Full-frame baseline metrics drift for ' + q['query_id'])
        if ranked_signature(actual['results']) != ranked_signature(expected['results']):
            raise ValueError('Full-frame baseline ranked windows drift for ' + q['query_id'])
        report['queries'].append(entry)
    for mode in MODES:
        values = [q['modes'][mode] for q in report['queries']]
        positives=[q['modes'][mode]['metrics'] for q in report['queries'] if q['present']]
        negatives=[q['modes'][mode]['metrics'] for q in report['queries'] if not q['present']]
        aggregate = dict(positive_queries=len(positives), negative_queries=len(negatives))
        for key in ('hit_at_1', 'hit_at_5', 'mrr', 'precision_at_5'):
            if positives:aggregate[key] = sum(v[key] for v in positives) / len(positives)
        if negatives:
            aggregate['negative_return_rate']=sum(v['return_rate'] for v in negatives)/len(negatives)
            aggregate['negative_returned_candidates']=[v['returned_candidates'] for v in negatives]
        report['aggregates'][mode] = aggregate
        report.setdefault('latency_aggregates', {})[mode] = dict(
            mean_seconds=sum(v['latency_seconds'] for v in values) / len(values),
            min_seconds=min(v['latency_seconds'] for v in values),
            max_seconds=max(v['latency_seconds'] for v in values))
    if report['aggregates']['full_frame'] != baseline_report['aggregates']['visual']:
        raise ValueError('Full-frame baseline aggregate metrics drift')
    report['baseline_reproduction'] = dict(passed=True, exact_metrics=True,
                                          exact_ranked_windows=True, annotations_sha256_match=True)
    return report


def make_runner(manifest, schema):
    from src.stage13_index import require_local_tables
    from src.stage13_evaluate import make_runner as base_runner
    require_local_tables(schema)
    return base_runner(manifest, schema)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path,
                        default=Path('outputs/stage13/human_confirmed_positive_evaluation.json'))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.annotations.read_bytes()
    manifest = json.loads(raw)
    baseline = json.loads(args.baseline_report.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    validate_manifest(manifest)
    if baseline.get('annotations_sha256') != digest:
        raise ValueError('Baseline annotation SHA256 differs from loaded annotations')
    runner = make_runner(manifest, args.schema)
    report = evaluate(manifest, search_fn=runner, baseline_report=baseline, annotations_sha256=digest)
    report['settings'] = dict(SETTINGS, schema=args.schema, modes=MODES, cached_models_only=True)
    report['baseline_report'] = str(args.baseline_report.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()))))
    return report


if __name__ == '__main__':
    main()
