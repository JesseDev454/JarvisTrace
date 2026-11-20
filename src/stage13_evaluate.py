"""Verified-only temporal retrieval evaluation; draft labels never score predictions.

Relevance is intersection / min(predicted duration, labelled duration) >= .5
for the same video (overlap coefficient, NOT temporal IoU). Precision@5 has
fixed denominator five. Negative return rate measures candidate return only;
there is no absence threshold or accuracy claim. MRR is bounded candidate MRR:
100 raw candidates per channel before window collapse, never exhaustive recall.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

MODES={'visual':(1,0),'text':(0,1),'hybrid':(1,1)}


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_manifest(manifest):
    if manifest.get('format')!='group8-stage13-annotations-v1':
        raise ValueError('Unknown annotation format')
    clips=manifest.get('clips',[])
    if not clips:raise ValueError('At least one clip required')
    collection={}
    for clip in clips:
        key=clip.get('video_id')
        if not isinstance(key,str) or not key or key in collection:raise ValueError('Unique video IDs required')
        if not _number(clip.get('duration_seconds')) or clip['duration_seconds']<=0:raise ValueError('Invalid clip duration')
        if not re.fullmatch('[a-f0-9]{64}',clip.get('source_sha256','')):raise ValueError('Source SHA256 required')
        for field in ('clip_path','source_filename','camera_id'):
            if not isinstance(clip.get(field),str) or not clip[field].strip():raise ValueError('Missing clip '+field)
        if not _number(clip.get('source_offset_seconds')) or clip['source_offset_seconds']<0:raise ValueError('Invalid source offset')
        collection[key]=clip
    seen=set()
    for q in manifest.get('queries',[]):
        key=q.get('query_id')
        if not isinstance(key,str) or not key or key in seen:raise ValueError('Unique query IDs required')
        seen.add(key)
        if not isinstance(q.get('query'),str) or not 1<=len(q['query'].strip())<=1000:raise ValueError('Invalid query')
        if not isinstance(q.get('target_category'),str) or not q['target_category'].strip():raise ValueError('Target category required')
        if type(q.get('present')) is not bool:raise ValueError('Present must be boolean')
        if q.get('review_status') not in ('proposed','verified'):raise ValueError('Invalid review status')
        if q['review_status']!='verified':continue
        if not isinstance(q.get('reviewer'),str) or not q['reviewer'].strip():raise ValueError('Verified labels require reviewer')
        if q.get('independent_of_predictions') is not True:raise ValueError('Independent human review required')
        if q.get('scope_review_complete') is not True:raise ValueError('Complete scope review required')
        scope=q.get('scope_video_ids')
        if not isinstance(scope,list) or len(scope)!=len(set(scope)) or set(scope)!=set(collection):raise ValueError('Verified scope must equal complete manifest collection')
        intervals=q.get('relevant_intervals')
        if not isinstance(intervals,list) or bool(intervals)!=q['present']:raise ValueError('Positive labels need intervals; negatives must have none')
        for interval in intervals:
            video=interval.get('video_id');a=interval.get('start_seconds');b=interval.get('end_seconds')
            if video not in scope or not _number(a) or not _number(b) or not 0<=a<b<=collection[video]['duration_seconds']:raise ValueError('Invalid relevant interval')
    return collection


def score_query(query, ranked_hits):
    if query.get('review_status')!='verified':raise ValueError('Only verified labels may be scored')
    unique=[];seen=set()
    for h in ranked_hits:
        a,b=h['start_seconds'],h['end_seconds']
        if h['video_id'] not in query['scope_video_ids']:raise ValueError('Out-of-scope retrieval result')
        if not _number(a) or not _number(b) or not 0<=a<b:raise ValueError('Invalid predicted interval')
        key=(h['video_id'],a,b)
        if key not in seen:unique.append(h);seen.add(key)
    result={'returned_candidates':len(unique)}
    if not query['present']:return dict(result,return_rate=int(bool(unique)))
    relevant=[]
    for h in unique:
        relevant.append(any(h['video_id']==g['video_id'] and max(0,min(h['end_seconds'],g['end_seconds'])-max(h['start_seconds'],g['start_seconds']))/min(h['end_seconds']-h['start_seconds'],g['end_seconds']-g['start_seconds'])>=.5 for g in query['relevant_intervals']))
    return dict(result,hit_at_1=int(any(relevant[:1])),hit_at_5=int(any(relevant[:5])),mrr=next((1/i for i,r in enumerate(relevant,1) if r),0),precision_at_5=sum(relevant[:5])/5)


def evaluate(manifest, *, search_fn=None):
    clips=validate_manifest(manifest)
    verified=[q for q in manifest.get('queries',[]) if q['review_status']=='verified']
    report={'report_format':'group8-stage13-evaluation-v1','status':'evaluated' if verified else 'awaiting_human_verification','skipped_queries':[{'query_id':q['query_id'],'reason':'proposed labels require independent complete human verification'} for q in manifest.get('queries',[]) if q['review_status']!='verified'],'queries':[],'aggregates':{},'metric_definition':__doc__,'indexing_time_seconds':None,'indexing_time_note':'Not measured by this read-only evaluator'}
    if not verified:return report
    if search_fn is None:raise ValueError('Explicit isolated-schema retrieval runner required')
    coverage=getattr(search_fn,'text_coverage',None)
    report['text_coverage']=coverage
    report['latency_protocol']='Warm latency: each mode warmed once with first verified query before timed evaluation; excludes indexing.'
    report['mrr_protocol']='Bounded candidate MRR: at most 100 raw candidates per channel before window collapse; no exhaustive recall claim.'
    report['warmup_seconds']={}
    for mode,(vw,tw) in MODES.items():
        start=time.perf_counter()
        search_fn(verified[0]['query'],visual_weight=vw,text_weight=tw,limit=100,candidate_limit=100)
        report['warmup_seconds'][mode]=time.perf_counter()-start
    for q in verified:
        entry={'query_id':q['query_id'],'query':q['query'],'present':q['present'],'modes':{}}
        for mode,(vw,tw) in MODES.items():
            start=time.perf_counter()
            found=search_fn(q['query'],visual_weight=vw,text_weight=tw,limit=100,candidate_limit=100)
            elapsed=time.perf_counter()-start
            for hit in found['results']:
                clip=clips.get(hit['video_id'])
                if clip is None or hit['end_seconds']>clip['duration_seconds']:
                    raise ValueError('Prediction outside evaluation clip')
                if hit.get('source_sha256',clip['source_sha256'])!=clip['source_sha256']:
                    raise ValueError('Prediction source differs from annotation clip')
            metrics=score_query(q,found['results'])
            incomplete=coverage is not None and bool(coverage['missing_video_ids'])
            unavailable=tw>0 and (found['candidate_counts']['text']==0 or incomplete)
            reason='Incomplete source-validated reviewed text coverage' if incomplete else 'No reviewed text candidates for this query'
            entry['modes'][mode]={'status':('unavailable' if mode=='text' else 'not_comparable') if unavailable else 'evaluated','reason':reason if unavailable else None,'latency_seconds':elapsed,'latency_label':'warm','candidate_counts':found['candidate_counts'],'metrics':None if unavailable else metrics,'results':found['results']}
        report['queries'].append(entry)
    for mode in MODES:
        usable=[(e['present'],e['modes'][mode]['metrics']) for e in report['queries'] if e['modes'][mode]['status']=='evaluated']
        if not usable:continue
        positives=[v for p,v in usable if p];negatives=[v for p,v in usable if not p]
        aggregate={'positive_queries':len(positives),'negative_queries':len(negatives)}
        for key in ('hit_at_1','hit_at_5','mrr','precision_at_5'):
            if positives:aggregate[key]=sum(v[key] for v in positives)/len(positives)
        if negatives:
            aggregate['negative_return_rate']=sum(v['return_rate'] for v in negatives)/len(negatives)
            aggregate['negative_returned_candidates']=[v['returned_candidates'] for v in negatives]
        report['aggregates'][mode]=aggregate
    return report


def make_runner(manifest,schema):
    """Validate exact isolated collection before importing/loading any encoder."""
    clips=validate_manifest(manifest)
    if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*',schema) or schema=='public':raise ValueError('Explicit non-public isolated schema required')
    from src.pg_config import open_database
    with open_database(schema=schema,readonly=True) as connection:
        # Qualify explicitly: search_path falls back to public for missing tables.
        from psycopg import sql
        rows=connection.execute(sql.SQL('SELECT DISTINCT video_id, source_sha256 FROM {}.visual_embeddings').format(sql.Identifier(schema))).fetchall()
        actual={r['video_id'] for r in rows}
        if actual!=set(clips):raise ValueError('Indexed video IDs must exactly equal evaluation collection')
        if any(r['source_sha256']!=clips[r['video_id']]['source_sha256'] for r in rows):raise ValueError('Indexed source hash differs from annotation clip')
        for table in ('video_recordings','scenes','frame_candidates','temporal_window_index'):
            connection.execute(sql.SQL('SELECT 1 FROM {}.{} LIMIT 0').format(sql.Identifier(schema),sql.Identifier(table)))
    # HF cache-only mode also covers OpenCLIP hub weight resolution; MiniLM
    # already explicitly sets local_files_only=True in reviewed_text.py.
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE=True
    from src.stage11 import search
    from src.visual_index import VisualIndex
    from src.reviewed_text import ReviewedTextIndex
    visual=VisualIndex(schema=schema);text=ReviewedTextIndex(schema=schema)
    # A nonzero unit probe measures provenance-qualified source coverage only;
    # no annotation query, label, or newly generated embedding is indexed.
    probe=[1.0]+[0.0]*383
    available=[];missing=[]
    for video_id in sorted(clips):
        rows=text.search(probe,video_id=video_id,limit=1)['results']
        (available if rows else missing).append(video_id)
    def run(query,**settings):
        return search(query,visual_index=visual,text_index=text,**settings)
    run.text_coverage={'available_video_ids':available,'missing_video_ids':missing,'complete':not missing,'method':'source-validated ReviewedTextIndex unit-vector probe per clip; no label ingestion'}
    return run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--schema',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    raw=args.annotations.read_bytes();manifest=json.loads(raw)
    validate_manifest(manifest)
    if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*',args.schema) or args.schema=='public':raise ValueError('Explicit isolated non-public schema required')
    if args.output.exists():raise FileExistsError(args.output)
    if args.dry_run:
        report={'report_format':'group8-stage13-evaluation-v1','status':'validation_only','verified_queries':sum(q['review_status']=='verified' for q in manifest.get('queries',[])),'aggregates':{}}
    else:
        runner=make_runner(manifest,args.schema) if any(q['review_status']=='verified' for q in manifest.get('queries',[])) else None
        report=evaluate(manifest,search_fn=runner)
    report['annotations_sha256']=hashlib.sha256(raw).hexdigest()
    report['settings']={'schema':args.schema,'dry_run':args.dry_run,'modes':MODES,'candidate_limit':100,'limit':100,'window_seconds':4,'rrf_k':60,'cached_models_only':True}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'output':str(args.output.resolve())}))
    return report


if __name__=='__main__':main()
