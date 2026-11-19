"""Typed pgvector image retrieval, independent of legacy text/Chroma stores."""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
from psycopg.types.json import Jsonb

from src.pg_config import open_database


def normalize_visual_vector(vector):
    a = np.asarray(vector,dtype=np.float32)
    if a.shape != (512,) or not np.isfinite(a).all():
        raise ValueError('Expected a finite 512-dimensional CLIP vector')
    norm = float(np.linalg.norm(a))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Cannot normalize zero/invalid visual vector')
    return a/norm


def encoder_identity(model_id, preprocessing, version, dimension):
    return hashlib.sha256(json.dumps([model_id,preprocessing,version,dimension],ensure_ascii=True).encode()).hexdigest()


class VisualIndex:
    def __init__(self, *, schema='public'):
        self.schema = schema

    def recording(self, video_id):
        with open_database(schema=self.schema,readonly=True) as c:
            row = c.execute('SELECT * FROM video_recordings WHERE video_id=%s',(video_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown recording ID; register/ingest its metadata first')
            return row

    def store_samples(self,samples,*,encoder_key,embedding_model,preprocess_version):
        if not samples:
            return 0
        if not re.fullmatch('[a-f0-9]{64}',encoder_key) or not embedding_model or not preprocess_version:
            raise ValueError('Encoder identity and preprocessing metadata required')
        inserted = 0
        with open_database(schema=self.schema) as c:
            for video_id in sorted({s['video_id'] for s in samples}):
                video = c.execute('SELECT * FROM video_recordings WHERE video_id=%s FOR UPDATE',(video_id,)).fetchone()
                if video is None:
                    raise ValueError('Unknown recording ID')
                prior = {r['source_sha256'] for r in c.execute('SELECT DISTINCT source_sha256 FROM visual_embeddings WHERE video_id=%s',(video_id,)).fetchall()}
                incoming = {s['source_sha256'] for s in samples if s['video_id']==video_id}
                if len(incoming) != 1 or (prior and incoming != prior):
                    raise ValueError('Recording content changed; register a new recording ID instead of replacing indexed source')
                incoming_profiles={tuple(s[k] for k in ('source_camera_id','source_video_path','source_fps','source_duration_seconds'))
                    for s in samples if s['video_id']==video_id}
                if len(incoming_profiles)!=1:
                    raise ValueError('Conflicting source metadata in batch')
                profile=incoming_profiles.pop()
                fps,duration=profile[2:]
                registered_fps,registered_duration=float(video['fps']),float(video['duration_seconds'])
                if (not all(math.isfinite(v) and v>0 for v in (fps,duration,registered_fps,registered_duration))
                        or profile[:2]!=(video['camera_id'],video['local_video_path'])
                        or abs(fps-registered_fps)>1e-4
                        or abs(duration-registered_duration)>max(.01,1/fps)):
                    raise ValueError('Recording metadata changed during sampling')
                previous_profiles={tuple(r[k] for k in ('source_camera_id','source_video_path','source_fps','source_duration_seconds'))
                    for r in c.execute('''SELECT DISTINCT source_camera_id,source_video_path,source_fps,source_duration_seconds
                        FROM visual_embeddings WHERE video_id=%s AND source_camera_id IS NOT NULL''',(video_id,)).fetchall()}
                if previous_profiles and previous_profiles!={profile}:
                    raise ValueError('Indexed recording metadata changed; register a new ID to preserve old evidence')
                for s in (s for s in samples if s['video_id']==video_id):
                    if tuple(s[k] for k in ('source_camera_id','source_video_path','source_fps','source_duration_seconds'))!=profile:
                        raise ValueError('Recording metadata changed during sampling')
                    frame,t = s['frame_index'],s['timestamp_seconds']
                    if (type(frame) is not int or frame<0 or not math.isfinite(t) or t<0
                            or t>=duration or abs(t-frame/fps)>1e-5):
                        raise ValueError('Invalid frame/time for this recording')
                    if any(not re.fullmatch('[a-f0-9]{64}',s[k]) for k in ('source_sha256','image_sha256')):
                        raise ValueError('Content hashes required')
                    bbox = s['bbox_xyxy']
                    if s['variant']=='frame' and bbox is None:
                        box_key = 'full'
                    elif s['variant']=='crop' and bbox is not None and len(bbox)==4:
                        if not all(math.isfinite(x) for x in bbox) or not (0<=bbox[0]<bbox[2] and 0<=bbox[1]<bbox[3]):
                            raise ValueError('Invalid crop box')
                        box_key = json.dumps(bbox,separators=(',',':'))
                    else:
                        raise ValueError('Invalid visual variant/box')
                    vector = normalize_visual_vector(s['embedding'])
                    # Only explicit reindexing with a verified source hash may fill
                    # pre-snapshot rows. Count inserts separately from this repair.
                    c.execute('''UPDATE visual_embeddings SET source_camera_id=%s,source_video_path=%s,
                        source_fps=%s,source_duration_seconds=%s WHERE video_id=%s AND source_sha256=%s
                        AND source_camera_id IS NULL''',(*profile,video_id,s['source_sha256']))
                    inserted += c.execute('''INSERT INTO visual_embeddings
                        (video_id,source_sha256,frame_index,timestamp_seconds,variant,bbox_key,bbox_xyxy,
                         image_path,image_sha256,encoder_key,embedding_model,preprocess_version,crop_provenance,embedding,
                         source_camera_id,source_video_path,source_fps,source_duration_seconds)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT ON CONSTRAINT visual_sample_encoder_key DO NOTHING''',
                        (video_id,s['source_sha256'],frame,t,s['variant'],box_key,Jsonb(bbox) if bbox else None,
                         s['image_path'],s['image_sha256'],encoder_key,embedding_model,preprocess_version,
                         Jsonb(s.get('crop_provenance')) if s.get('crop_provenance') else None,vector,*profile)).rowcount
        return inserted

    def stats(self):
        with open_database(schema=self.schema,readonly=True) as c:
            return c.execute('''SELECT count(*) AS total_embeddings,
                count(DISTINCT video_id) AS recordings,
                count(*) FILTER(WHERE variant='frame') AS full_frames,
                count(*) FILTER(WHERE variant='crop') AS crops FROM visual_embeddings''').fetchone()

    def search(self,vector,encoder_key,*,limit=10,camera_id=None,video_id=None,
               start_seconds=None,end_seconds=None,variant=None,camera_ids=None):
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Result limit must be 1..100')
        for value in (start_seconds,end_seconds):
            if value is not None and (not math.isfinite(value) or value<0):
                raise ValueError('Time bounds must be finite and nonnegative')
        if start_seconds is not None and end_seconds is not None and end_seconds<=start_seconds:
            raise ValueError('End must follow start')
        if variant not in (None,'frame','crop'):
            raise ValueError('Unknown visual variant')
        conditions,params = ['v.encoder_key=%s','v.source_camera_id IS NOT NULL'],[normalize_visual_vector(vector),encoder_key]
        if camera_ids is not None:
            if not isinstance(camera_ids,(list,tuple)) or any(not isinstance(v,str) for v in camera_ids):
                raise ValueError('Camera scope must be a list of camera IDs')
            if not camera_ids:
                return []
            conditions.append('v.source_camera_id=ANY(%s)');params.append(list(camera_ids))
        for expression,value in [('v.source_camera_id=%s',camera_id),('v.video_id=%s',video_id),
                                 ('v.timestamp_seconds >= %s',start_seconds),('v.timestamp_seconds < %s',end_seconds),('v.variant=%s',variant)]:
            if value is not None:
                conditions.append(expression);params.append(value)
        params.append(limit)
        query = '''WITH q AS (SELECT %s::vector AS embedding), ranked AS (
            SELECT v.id,v.video_id,v.source_camera_id AS camera_id,v.source_video_path,
                v.source_sha256,v.source_duration_seconds AS duration_seconds,v.frame_index,v.timestamp_seconds,v.variant,v.bbox_xyxy,
                v.image_path,v.image_sha256,v.embedding_model,v.encoder_key,v.crop_provenance,
                1-(v.embedding <=> q.embedding) AS similarity,
                row_number() OVER(PARTITION BY v.video_id,v.source_sha256,v.frame_index
                    ORDER BY v.embedding <=> q.embedding,v.variant,v.id) AS per_frame_rank
            FROM visual_embeddings v CROSS JOIN q
            WHERE '''+' AND '.join(conditions)+''')
            SELECT * FROM ranked WHERE per_frame_rank=1
            ORDER BY similarity DESC,video_id,frame_index,id LIMIT %s'''
        with open_database(schema=self.schema,readonly=True) as c:
            rows = c.execute(query,params).fetchall()
        for row in rows:
            row.pop('per_frame_rank')
            row['timestamp_seconds'] = float(row['timestamp_seconds'])
            row['similarity'] = float(row['similarity'])
        return rows
