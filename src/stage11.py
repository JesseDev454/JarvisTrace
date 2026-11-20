"""Stage 11: read-only visual/text window retrieval with explained rank fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from src.hybrid_search import filter_windows_by_similarity, fuse_windows
from src.reviewed_text import ReviewedTextIndex, embed_text
from src.stage10 import _encoder, _identity
from src.visual_index import VisualIndex


def search(query, *, camera_id=None, video_id=None, start_seconds=None,
           end_seconds=None, window_seconds=4, rrf_k=60, visual_weight=1,
           text_weight=1, limit=10, candidate_limit=100, variant=None,camera_ids=None,
           min_visual_similarity=None, min_text_similarity=None,
           encoder=None, visual_index=None, text_index=None, text_embedder=None):
    if not isinstance(query,str) or not query.strip() or len(query)>1000:
        raise ValueError('Query must contain 1..1000 characters')
    if type(candidate_limit) is not int or not 1<=candidate_limit<=100:
        raise ValueError('Candidate limit must be 1..100')
    settings=dict(window_seconds=window_seconds,rrf_k=rrf_k,visual_weight=visual_weight,
                  text_weight=text_weight,start_seconds=start_seconds,end_seconds=end_seconds,limit=limit)
    fuse_windows([],[],**settings)  # Validate before loading encoders.
    filter_windows_by_similarity([],mode='hybrid',min_visual_similarity=min_visual_similarity,
                                 min_text_similarity=min_text_similarity)
    filters=dict(camera_id=camera_id,video_id=video_id,start_seconds=start_seconds,end_seconds=end_seconds)
    if camera_ids is not None:
        camera_ids=sorted(set(camera_ids))
        if camera_id is not None and camera_id not in camera_ids:
            raise ValueError('Requested camera is outside the authorized camera scope')
        filters['camera_ids']=camera_ids
    images=[]
    descriptions=dict(results=[],disabled=True)
    if visual_weight>0:
        encoder=encoder or _encoder()
        images=(visual_index or VisualIndex()).search(encoder.encode_text(query.strip()),
                    _identity(encoder),limit=candidate_limit,variant=variant,**filters)
    if text_weight>0:
        descriptions=(text_index or ReviewedTextIndex()).search(
                    (text_embedder or embed_text)(query.strip()),limit=candidate_limit,**filters)
    texts=descriptions['results']
    candidate_settings=dict(settings,limit=100)
    raw_comparisons=dict(visual=fuse_windows(images,[],**candidate_settings),
                         text=fuse_windows([],texts,**candidate_settings),
                         hybrid=fuse_windows(images,texts,**candidate_settings))
    raw_window_counts={mode:len(rows) for mode,rows in raw_comparisons.items()}
    accepted_settings=dict(candidate_settings,min_visual_similarity=min_visual_similarity,
                           min_text_similarity=min_text_similarity)
    accepted_comparisons=dict(
        visual=fuse_windows(images,[],**accepted_settings),
        text=fuse_windows([],texts,**accepted_settings),
        hybrid=fuse_windows(images,texts,**accepted_settings))
    window_counts={mode:len(rows) for mode,rows in accepted_comparisons.items()}
    comparisons={mode:rows[:limit] for mode,rows in accepted_comparisons.items()}
    return dict(report_format='group8-stage11-search-v1',query=query.strip(),
                settings=dict(settings,candidate_limit=candidate_limit,variant=variant,
                    min_visual_similarity=min_visual_similarity,min_text_similarity=min_text_similarity,**{
                    k:v for k,v in filters.items() if k not in settings}),
                results=comparisons['hybrid'],comparison=comparisons,
                raw_window_counts=raw_window_counts,window_counts=window_counts,
                candidate_counts=dict(visual=len(images),text=len(texts)),
                text_diagnostics={k:v for k,v in descriptions.items() if k!='results'},
                limitations='RRF and channel similarity are ranking evidence, not confidence. Configured thresholds are heuristics and do not prove presence or absence. Legacy labels may be inaccurate; no action or absence guarantee.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['search','compare'])
    queries=parser.add_mutually_exclusive_group(required=True)
    queries.add_argument('--query')
    queries.add_argument('--queries-file',type=Path)
    parser.add_argument('--camera',dest='camera_id')
    parser.add_argument('--video-id')
    parser.add_argument('--start-seconds',type=float)
    parser.add_argument('--end-seconds',type=float)
    parser.add_argument('--window-seconds',type=float,default=4)
    parser.add_argument('--rrf-k',type=float,default=60)
    parser.add_argument('--visual-weight',type=float,default=1)
    parser.add_argument('--text-weight',type=float,default=1)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--candidate-limit',type=int,default=100)
    parser.add_argument('--variant',choices=['frame','crop'])
    parser.add_argument('--output-root',type=Path,default=Path('outputs/stage11'))
    args=vars(parser.parse_args())
    command=args.pop('command'); query=args.pop('query'); suite=args.pop('queries_file')
    root=args.pop('output_root')
    try:
        entries=json.loads(suite.read_text(encoding='utf-8'))['queries'] if suite else [{'query':query}]
        reports=[dict(case=entry,report=search(entry['query'],**args)) for entry in entries]
        if command=='compare':
            folder=root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
            folder.mkdir(parents=True,exist_ok=False)
            path=folder/'comparison.json'
            path.write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf-8')
            result=dict(report_path=str(path.resolve()),queries=len(reports),status='comparison_saved')
        else:
            result=reports[0]['report'] if len(reports)==1 else reports
    except Exception as exc:
        parser.exit(1,f'ERROR: {type(exc).__name__}; check inputs, cached models and local database configuration.\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
