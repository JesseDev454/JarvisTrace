"""Read-only retrieval from provenance-qualified legacy text annotations.

Legacy approval records do not contain signed source hashes. Bind them at query
time to the pinned visual source and its current bytes; report this limitation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.pg_config import open_database, ROOT

TEXT_MODEL='sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_POLICY=ROOT/'docs/evaluation/stage11_text_exclusions.json'


def normalize_text_vector(vector):
    vector=np.asarray(vector,dtype=np.float32)
    if vector.shape!=(384,) or not np.isfinite(vector).all():
        raise ValueError('Expected a finite 384-dimensional MiniLM vector')
    norm=float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm<=1e-12:
        raise ValueError('Invalid text vector norm')
    return vector/norm


@lru_cache(maxsize=1)
def text_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL,device='cpu',local_files_only=True)


def embed_text(query):
    if not isinstance(query,str) or not query.strip() or len(query)>1000:
        raise ValueError('Query must contain 1..1000 characters')
    return normalize_text_vector(text_encoder().encode(query.strip(),normalize_embeddings=True))


def _hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


class ReviewedTextIndex:
    def __init__(self,*,schema='public',exclusions=None):
        self.schema=schema
        if exclusions is None:
            policy=json.loads(DEFAULT_POLICY.read_text(encoding='utf-8'))
            if policy.get('format')!='group8-reviewed-text-exclusions-v1':
                raise ValueError('Unknown text review policy')
            exclusions=policy['exclusions']
        self.exclusions=exclusions

    def _sources(self,c,video_id,camera_id,camera_ids=None):
        rows=c.execute('''SELECT DISTINCT v.video_id,v.source_sha256,
            v.source_camera_id AS camera_id,v.source_video_path,v.source_fps,
            v.source_duration_seconds AS duration_seconds,
            r.camera_id AS registry_camera,r.local_video_path AS registry_path,
            r.fps AS registry_fps,r.duration_seconds AS registry_duration
            FROM visual_embeddings v JOIN video_recordings r USING(video_id)
            WHERE v.source_camera_id IS NOT NULL''').fetchall()
        grouped={}
        for row in rows:
            if ((video_id is not None and row['video_id']!=video_id)
                    or (camera_id is not None and row['camera_id']!=camera_id)
                    or (camera_ids is not None and row['camera_id'] not in camera_ids)):
                continue
            grouped.setdefault(row['video_id'],[]).append(row)
        accepted,excluded={},[]
        for key,profiles in grouped.items():
            if len(profiles)!=1:
                excluded.append(dict(video_id=key,reason='ambiguous pinned source metadata'))
                continue
            r=profiles[0]
            try:
                if (r['registry_camera']!=r['camera_id'] or r['registry_path']!=r['source_video_path']
                        or abs(r['registry_fps']-r['source_fps'])>1e-4
                        or abs(r['registry_duration']-r['duration_seconds'])>max(.01,1/r['source_fps'])
                        or _hash(r['source_video_path'])!=r['source_sha256']):
                    raise ValueError('Source/registry mismatch')
            except (OSError,ValueError):
                excluded.append(dict(video_id=key,reason='legacy text could not bind to unchanged indexed source'))
                continue
            accepted[key]=r
        return accepted,excluded

    def search(self,vector,*,limit=100,video_id=None,camera_id=None,start_seconds=None,end_seconds=None,camera_ids=None):
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Text candidate limit must be 1..100')
        for value in (start_seconds,end_seconds):
            if value is not None and (not math.isfinite(value) or value<0):
                raise ValueError('Time bounds must be finite and nonnegative')
        if start_seconds is not None and end_seconds is not None and end_seconds<=start_seconds:
            raise ValueError('End must follow start')
        vector=normalize_text_vector(vector)
        excluded_counts=dict(policy=0,source=0,invalid_temporal_provenance=0,invalid_range=0)
        result=[]
        with open_database(schema=self.schema,readonly=True) as c:
            sources,excluded_sources=self._sources(c,video_id,camera_id,camera_ids)
            # A general scenes row is never sufficient: require the actual manual
            # annotation or explicit temporal index record and exact text/range match.
            rows=c.execute('''WITH q AS(SELECT %s::vector AS embedding), eligible AS(
                SELECT s.id AS scene_id,s.video_id,s.camera_id AS scene_camera,
                    s.start_seconds,s.end_seconds,s.description,s.embedding_model,
                    1-(s.embedding <=> q.embedding) AS similarity,
                    'legacy_manual_frame_label' AS provenance_type,f.id AS frame_candidate_id,
                    f.timestamp_seconds AS frame_timestamp_seconds,
                    NULL::jsonb AS frame_candidate_ids,NULL::jsonb AS frame_times_seconds,
                    NULL::text AS caption_model,NULL::text AS prompt_version
                FROM scenes s JOIN frame_candidates f ON f.video_id=s.video_id
                    AND f.clip_start_seconds=s.start_seconds AND f.clip_end_seconds=s.end_seconds
                    AND f.description=s.description
                CROSS JOIN q WHERE f.description IS NOT NULL AND s.embedding_model=%s
                UNION ALL
                SELECT s.id,s.video_id,s.camera_id,s.start_seconds,s.end_seconds,
                    s.description,s.embedding_model,1-(s.embedding <=> q.embedding),
                    'reviewed_temporal_caption',NULL::bigint,NULL::double precision,tw.frame_candidate_ids,
                    tw.frame_times_seconds,tw.model_id,tw.prompt_version
                FROM scenes s JOIN temporal_window_index tw ON tw.scene_id=s.id
                    AND tw.video_id=s.video_id AND tw.start_seconds=s.start_seconds
                    AND tw.end_seconds=s.end_seconds AND tw.caption=s.description
                CROSS JOIN q WHERE s.embedding_model=%s)
                SELECT * FROM eligible ORDER BY similarity DESC,scene_id,provenance_type''',
                (vector,TEXT_MODEL,TEXT_MODEL)).fetchall()
            for row in rows:
                source=sources.get(row['video_id'])
                if source is None or row['scene_camera']!=source['camera_id']:
                    excluded_counts['source']+=1
                    continue
                a,b=float(row['start_seconds']),float(row['end_seconds'])
                if not all(math.isfinite(x) for x in (a,b)) or not 0<=a<b<=source['duration_seconds']:
                    excluded_counts['invalid_range']+=1
                    continue
                if row['provenance_type']=='legacy_manual_frame_label':
                    t=row['frame_timestamp_seconds']
                    if t is None or not math.isfinite(t) or not a<=t<b:
                        excluded_counts['invalid_range']+=1
                        continue
                    row['frame_timestamp_seconds']=float(t)
                if ((start_seconds is not None and b<=start_seconds) or
                        (end_seconds is not None and a>=end_seconds)):
                    continue
                digest=hashlib.sha256(row['description'].encode()).hexdigest()
                if any(p['video_id']==row['video_id'] and p['scene_id']==row['scene_id']
                       and p['description_sha256']==digest for p in self.exclusions):
                    excluded_counts['policy']+=1
                    continue
                if row['provenance_type']=='reviewed_temporal_caption':
                    ids,times=row['frame_candidate_ids'],row['frame_times_seconds']
                    valid=(isinstance(ids,list) and isinstance(times,list) and len(ids)>=2
                           and all(type(i) is int and i>0 for i in ids)
                           and len(set(ids))==len(ids) and len(times)==len(ids))
                    if valid:
                        frames=c.execute('''SELECT id,video_id,timestamp_seconds FROM frame_candidates
                            WHERE id=ANY(%s)''',(ids,)).fetchall()
                        by_id={f['id']:f for f in frames}
                        valid=len(frames)==len(ids)
                        for i,t in zip(ids,times):
                            f=by_id.get(i)
                            if (f is None or f['video_id']!=row['video_id']
                                    or not isinstance(t,(int,float)) or not math.isfinite(t)
                                    or not a<=t<=b or abs(float(f['timestamp_seconds'])-t)>1e-5):
                                valid=False
                    if not valid:
                        excluded_counts['invalid_temporal_provenance']+=1
                        continue
                row.pop('scene_camera')
                row.update(start_seconds=a,end_seconds=b,similarity=float(row['similarity']),
                    camera_id=source['camera_id'],source_sha256=source['source_sha256'],
                    source_video_path=source['source_video_path'],duration_seconds=source['duration_seconds'],
                    description_sha256=digest,review_status='legacy_provenance_qualified_annotation')
                result.append(row)
        return dict(results=result[:limit],eligible_sources=len(sources),excluded_sources=excluded_sources,
                    excluded_counts=excluded_counts,
                    provenance_limit='Legacy review status comes from manual/approval records; source binding is verified now, not cryptographically recorded at original annotation time.')
