"""Bounded offline indexing in an explicitly owned Stage13 PostgreSQL schema."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

MARKER='group8-stage13-isolated-v1'
BASE_TABLES=('video_recordings','scenes','frame_candidates','temporal_window_index')


def require_local_tables(schema,*,base_only=False):
    """Reject search-path fallback, views, and tables owned by another role."""
    validate_schema(schema)
    from psycopg import sql
    from src.pg_config import open_database
    required=set(BASE_TABLES)|{'stage13_control'}
    if not base_only: required.update(('visual_embeddings','alembic_version'))
    with open_database(readonly=True) as c:
        rows=c.execute('''SELECT t.relname,t.relkind,pg_get_userbyid(t.relowner)=current_user AS owned
            FROM pg_class t JOIN pg_namespace n ON n.oid=t.relnamespace
            WHERE n.nspname=%s AND t.relname=ANY(%s)''',(schema,sorted(required))).fetchall()
        if ({r['relname'] for r in rows}!=required
            or any(r['relkind']!='r' or not r['owned'] for r in rows)):
            raise ValueError('Stage13 requires all locally owned ordinary tables; public fallback is forbidden')
        markers=c.execute(sql.SQL('SELECT marker FROM {}.stage13_control').format(sql.Identifier(schema))).fetchall()
        if markers != [dict(marker=MARKER)]: raise ValueError('Stage13 ownership marker mismatch')


def validate_schema(schema):
    if not isinstance(schema,str) or not re.fullmatch(r'stage13_[a-z0-9][a-z0-9_]{0,54}',schema):
        raise ValueError('Require a dedicated lowercase stage13_ schema (maximum 63 characters)')
    return schema


def create_isolated_schema(schema):
    """Copy column types only; never inherit public defaults, sequences or rows."""
    validate_schema(schema)
    from psycopg import sql
    from src.pg_config import open_database, migrate
    with open_database() as c:
        exists=c.execute('SELECT 1 FROM pg_namespace WHERE nspname=%s',(schema,)).fetchone()
        if exists:
            found=c.execute('SELECT to_regclass(%s) AS name',(schema+'.stage13_control',)).fetchone()
            if found['name'] is None:
                raise ValueError('Existing schema is not owned by Stage13')
            rows=c.execute(sql.SQL('SELECT marker FROM {}.stage13_control').format(sql.Identifier(schema))).fetchall()
            if rows != [dict(marker=MARKER)]:
                raise ValueError('Existing schema ownership marker mismatch')
        else:
            c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            c.execute(sql.SQL('CREATE TABLE {}.stage13_control (marker TEXT PRIMARY KEY)').format(sql.Identifier(schema)))
            c.execute(sql.SQL('INSERT INTO {}.stage13_control VALUES (%s)').format(sql.Identifier(schema)),(MARKER,))
            for table in BASE_TABLES:
                c.execute(sql.SQL('CREATE TABLE {}.{} AS SELECT * FROM public.{} WITH NO DATA').format(
                    sql.Identifier(schema),sql.Identifier(table),sql.Identifier(table)))
            for table,key in [('video_recordings','video_id'),('scenes','id')]:
                c.execute(sql.SQL('ALTER TABLE {}.{} ADD PRIMARY KEY ({})').format(
                    sql.Identifier(schema),sql.Identifier(table),sql.Identifier(key)))
    require_local_tables(schema,base_only=True)
    migrate(schema=schema)
    require_local_tables(schema)


def compare_profile(actual,expected):
    for key in ('video_id','local_video_path','camera_id','fps','duration_seconds'):
        if actual[key]!=expected[key]:
            raise ValueError('Existing recording profile is immutable; use a new video ID')


def register_recording(index,path,camera,video_id,expected_hash):
    require_local_tables(index.schema)
    path=Path(path).resolve()
    if not path.is_file() or not re.fullmatch('[a-f0-9]{64}',expected_hash):
        raise ValueError('Source path and SHA256 required')
    def digest():
        with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
    if digest()!=expected_hash:
        raise ValueError('Source hash differs from manifest')
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps<=0 or count<=0:
            raise ValueError('Source requires usable CFR metadata')
    finally:
        cap.release()
    if digest()!=expected_hash: raise ValueError('Source changed during registration')
    profile=dict(video_id=video_id,local_video_path=str(path),camera_id=camera,fps=fps,duration_seconds=count/fps)
    from src.pg_config import open_database
    from psycopg import sql
    with open_database(schema=index.schema) as c:
        c.execute(sql.SQL('''INSERT INTO {}.video_recordings(video_id,local_video_path,camera_id,fps,duration_seconds)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT(video_id) DO NOTHING''').format(sql.Identifier(index.schema)),tuple(profile.values()))
        row=c.execute(sql.SQL('SELECT * FROM {}.video_recordings WHERE video_id=%s').format(sql.Identifier(index.schema)),(video_id,)).fetchone()
        compare_profile(row,profile)
    return profile


def run(manifest,schema,output_root,*,every_seconds=4,max_frames=10,batch_size=2,
        max_recordings=2,execute=False,verify_repeat=True):
    validate_schema(schema)
    if (type(max_frames) is not int or not 1<=max_frames<=60 or type(batch_size) is not int
        or not 1<=batch_size<=4 or type(max_recordings) is not int or not 1<=max_recordings<=10
        or not math.isfinite(every_seconds) or every_seconds<=0):
        raise ValueError('Workload exceeds Stage13 bounds')
    manifest=Path(manifest).resolve()
    clips=json.loads(manifest.read_text(encoding='utf-8'))['clips']
    if not isinstance(clips,list) or not clips: raise ValueError('Manifest clips must be nonempty')
    selected=[]; ids=set()
    for item in clips[:max_recordings]:
        path=Path(item['clip_path'])
        if not path.is_absolute(): path=manifest.parent/path
        path=path.resolve()
        if (not path.is_file() or not re.fullmatch('[a-f0-9]{64}',item['source_sha256'])
            or any(not isinstance(item.get(k),str) or not 1<=len(item[k])<=120 for k in ('video_id','camera_id'))
            or item['video_id'] in ids):
            raise ValueError('Invalid/duplicate clip profile or missing source')
        ids.add(item['video_id']); selected.append(dict(item,clip_path=str(path)))
    predicted=(len(selected)+int(verify_repeat))*max_frames
    if predicted>300: raise ValueError('Predicted workload exceeds 300 images including repeat verification')
    result=dict(format='group8-stage13-index-v1',status='planned',schema=schema,
        selected_recordings=len(selected),predicted_images=predicted,every_seconds=every_seconds,
        max_frames=max_frames,batch_size=batch_size,verify_repeat=verify_repeat,
        clips=selected,limitations='Caller must confirm CFR. Full frames only; sample caps do not establish complete coverage.')
    if not execute: return result
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    create_isolated_schema(schema)
    from src.visual_index import VisualIndex
    from src.stage10 import index_recording
    import torch
    index=VisualIndex(schema=schema)
    folder=Path(output_root).resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    result.update(status='indexing',reports=[],counts_before=index.stats())
    started=time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    try:
        for item in selected:
            register_recording(index,item['clip_path'],item['camera_id'],item['video_id'],item['source_sha256'])
            result['reports'].append(index_recording(item['video_id'],every_seconds=every_seconds,
                max_frames=max_frames,batch_size=batch_size,output_root=folder,index=index))
        if verify_repeat:
            repeat=index_recording(selected[0]['video_id'],every_seconds=every_seconds,max_frames=max_frames,
                batch_size=batch_size,output_root=folder,index=index)
            result['repeat_report']=repeat
            if repeat['inserted_embeddings']!=0: raise ValueError('Idempotency verification failed')
        result['status']='complete'
    except Exception as exc:
        result.update(status='failed',failure_type=type(exc).__name__)
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        result['counts_after']=index.stats()
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        try:
            import psutil
            result['rss_bytes_at_end']=psutil.Process().memory_info().rss
        except ImportError: result['rss_bytes_at_end']=None
        result['report_path']=str(folder/'run.json')
        (folder/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--schema',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--every-seconds',type=float,default=4)
    parser.add_argument('--max-frames',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--max-recordings',type=int,default=2)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--verify-repeat',action=argparse.BooleanOptionalAction,default=True)
    try: result=run(**vars(parser.parse_args()))
    except Exception as exc: parser.exit(1,f'ERROR: {type(exc).__name__}; check inputs, owned schema and cached models.\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
