"""Offline, bounded portable-object crop experiment on frozen Stage13 frames.

Detector outputs are unreviewed proposals, never identities or event labels.
Two explicit phases expose actual detection workload before CLIP encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.stage10 import _file_hash, _save_image
from src.stage13_index import require_local_tables, validate_schema

SETTINGS=dict(imgsz=1280,conf=.25,iou=.7,classes=[24,26,28,39,63,67],
              padding_fraction=.1,max_per_frame=8,max_total=500,batch_size=2)
COPY_COLUMNS=('video_id,source_sha256,frame_index,timestamp_seconds,variant,bbox_key,bbox_xyxy,'
    'image_path,image_sha256,encoder_key,embedding_model,preprocess_version,crop_provenance,embedding,'
    'source_camera_id,source_video_path,source_fps,source_duration_seconds')


def check_schemas(baseline,target):
    validate_schema(baseline);validate_schema(target)
    if baseline==target:raise ValueError('Baseline and experiment schemas must differ')


def manifest_signature(document):
    selected={k:document[k] for k in ('settings','weights_sha256','baseline_digest','frames','crops')}
    return hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def select_crops(detections,width,height):
    """Sort by confidence then geometry; pad once; deduplicate actual pixel boxes."""
    selected=[]
    for d in detections:
        box=d['bbox'];confidence=d['confidence']
        if (len(box)!=4 or not all(math.isfinite(x) for x in box)
            or not 0<=confidence<=1 or not box[0]<box[2] or not box[1]<box[3]):
            raise ValueError('Invalid detector geometry/confidence')
        if d['class_id'] not in SETTINGS['classes'] or confidence<SETTINGS['conf']:continue
        x1,y1,x2,y2=box;px=(x2-x1)*.1;py=(y2-y1)*.1
        actual=[max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),
                min(width,math.ceil(x2+px)),min(height,math.ceil(y2+py))]
        if actual[2]-actual[0]<2 or actual[3]-actual[1]<2:continue
        selected.append(dict(d,bbox_xyxy=actual))
    selected.sort(key=lambda d:(-d['confidence'],d['class_id'],d['bbox_xyxy']))
    unique=[];seen=set()
    for row in selected:
        key=tuple(row['bbox_xyxy'])
        if key not in seen:unique.append(row);seen.add(key)
    return unique[:SETTINGS['max_per_frame']]


def snapshot(schema):
    """Content hashes include IDs, creation times and exact vector text."""
    from src.pg_config import open_database
    from psycopg import sql
    tables=('video_recordings','scenes','frame_candidates','temporal_window_index','visual_embeddings','alembic_version')
    result={}
    with open_database(readonly=True) as c:
        for table in tables:
            rows=c.execute(sql.SQL('SELECT row_to_json(t)::text AS data FROM {}.{} t').format(sql.Identifier(schema),sql.Identifier(table))).fetchall()
            payload='\n'.join(sorted(r['data'] for r in rows)).encode()
            result[table]=dict(rows=len(rows),sha256=hashlib.sha256(payload).hexdigest())
    return result


def baseline_rows(schema):
    require_local_tables(schema)
    from src.pg_config import open_database
    from psycopg import sql
    with open_database(readonly=True) as c:
        rows=c.execute(sql.SQL('SELECT * FROM {}.visual_embeddings ORDER BY video_id,frame_index,id').format(sql.Identifier(schema))).fetchall()
    if not rows or len(rows)>300 or any(r['variant']!='frame' for r in rows):
        raise ValueError('Baseline must contain 1..300 full frames only')
    if len({(r['encoder_key'],r['embedding_model'],r['preprocess_version']) for r in rows})!=1:
        raise ValueError('Baseline requires a single encoder identity')
    return rows


def verify_files(rows):
    checked={}
    for r in rows:
        for path,digest in [(r['source_video_path'],r['source_sha256']),(r['image_path'],r['image_sha256'])]:
            if path in checked:
                if checked[path]!=digest:raise ValueError('Conflicting content bindings')
            else:
                if _file_hash(path)!=digest:raise ValueError('Evidence/source hash mismatch')
                checked[path]=digest
    return checked


def validate_document(document,rows):
    """Reconstruct each crop from frozen frame pixels before trusting a manifest."""
    import cv2
    if (document.get('format')!='group8-stage13-object-crops-v1' or document['status']!='detected'
        or document['settings']!=SETTINGS or len(document['crops'])>500
        or document['manifest_signature']!=manifest_signature(document)):
        raise ValueError('Invalid/modified bounded detector manifest')
    expected={(r['video_id'],r['frame_index']):r for r in rows}
    if len(document['frames'])!=len(rows):raise ValueError('All baseline frames required')
    seen=set();expected_crops=[]
    for number,frame in enumerate(document['frames']):
        key=(frame['video_id'],frame['frame_index']);row=expected.get(key)
        if row is None or key in seen:raise ValueError('Invalid/duplicate frame binding')
        seen.add(key)
        for field in ('video_id','source_sha256','frame_index','image_path','image_sha256','encoder_key',
                      'embedding_model','preprocess_version','source_camera_id','source_video_path',
                      'source_fps','source_duration_seconds'):
            if frame[field]!=row[field]:raise ValueError('Frame provenance differs from baseline')
        if frame['timestamp_seconds']!=float(row['timestamp_seconds']):raise ValueError('Frame time differs from baseline')
        pixels=cv2.imread(frame['image_path'])
        if pixels is None or pixels.shape[:2]!=(frame['height'],frame['width']):raise ValueError('Frame dimensions differ')
        for proposal in select_crops(frame['raw_detections'],frame['width'],frame['height']):
            if len(expected_crops)>=500:continue
            x1,y1,x2,y2=proposal['bbox_xyxy']
            ok,payload=cv2.imencode('.png',pixels[y1:y2,x1:x2])
            if not ok:raise ValueError('Cannot reconstruct crop')
            expected_crops.append(dict(proposal,frame_number=number,image_sha256=hashlib.sha256(payload.tobytes()).hexdigest()))
    if len(expected_crops)!=len(document['crops']):raise ValueError('Crop proposals omitted/added')
    for expected_crop,crop in zip(expected_crops,document['crops']):
        if {k:v for k,v in crop.items() if k!='image_path'}!=expected_crop:
            raise ValueError('Crop differs from frozen frame proposal/pixels')
        if _file_hash(crop['image_path'])!=crop['image_sha256']:raise ValueError('Crop hash mismatch')


def detect(baseline,weights,output,*,max_frames=None):
    """Run inference only on immutable baseline raw PNGs, never label intervals."""
    import cv2
    import torch
    from ultralytics import YOLO
    from importlib.metadata import version
    output=Path(output).resolve();weights=Path(weights).resolve()
    if output.exists():raise FileExistsError(output)
    if not weights.is_file():raise ValueError('Existing local detector weights required')
    rows=baseline_rows(baseline)
    if max_frames is not None:
        if type(max_frames) is not int or not 1<=max_frames<=len(rows):raise ValueError('Invalid smoke frame cap')
        rows=rows[:max_frames]
    bindings=verify_files(rows);before=snapshot(baseline)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(format='group8-stage13-object-crops-v1',baseline_schema=baseline,
        baseline_snapshot=before,baseline_digest=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        settings=SETTINGS,weights_path=str(weights),weights_sha256=_file_hash(weights),
        ultralytics_version=version('ultralytics'),torch_version=torch.__version__,
        frames=[],crops=[],raw_detection_count=0,per_frame_cap_reached=0,total_cap_dropped=0,
        status='detecting',interpretation='Unreviewed detector proposals; no track identity or event ground truth.')
    model=YOLO(str(weights));started=time.perf_counter()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for row in rows:
        pixels=cv2.imread(row['image_path'])
        if pixels is None:raise ValueError('Could not read verified frame')
        h,w=pixels.shape[:2]
        prediction=model.predict(pixels,imgsz=1280,conf=.25,iou=.7,classes=SETTINGS['classes'],
                                 device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
        raw=[dict(class_id=int(cls),class_name=model.names[int(cls)],confidence=float(conf),bbox=box.tolist())
             for box,conf,cls in zip(prediction.boxes.xyxy.cpu().numpy(),prediction.boxes.conf.cpu().numpy(),prediction.boxes.cls.cpu().numpy())]
        selected=select_crops(raw,w,h)
        report['raw_detection_count']+=len(raw)
        report['per_frame_cap_reached']+=int(len(selected)==8)
        frame={k:row[k] for k in ('video_id','source_sha256','frame_index','image_path','image_sha256','encoder_key',
            'embedding_model','preprocess_version','source_camera_id','source_video_path','source_fps','source_duration_seconds')}
        frame.update(timestamp_seconds=float(row['timestamp_seconds']),width=w,height=h,raw_detections=raw)
        frame_number=len(report['frames']);report['frames'].append(frame)
        for proposal in selected:
            if len(report['crops'])>=500:
                report['total_cap_dropped']+=1;continue
            x1,y1,x2,y2=proposal['bbox_xyxy']
            path,digest=_save_image(pixels[y1:y2,x1:x2],output/'images')
            report['crops'].append(dict(proposal,frame_number=frame_number,image_path=path,image_sha256=digest))
    report.update(status='detected',elapsed_seconds=time.perf_counter()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        source_and_frame_bindings=bindings)
    verify_files(rows)
    if _file_hash(weights)!=report['weights_sha256'] or snapshot(baseline)!=before:raise ValueError('Baseline/weights changed during detection')
    report['manifest_signature']=manifest_signature(report)
    (output/'detections.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def clone_baseline(baseline,target,document):
    """Exact SQL vector copy; preserve numeric precision without re-normalization."""
    check_schemas(baseline,target)
    from src.stage13_index import create_isolated_schema
    from src.pg_config import open_database
    from psycopg import sql
    from psycopg.types.json import Jsonb
    require_local_tables(baseline)
    if snapshot(baseline)!=document['baseline_snapshot']:raise ValueError('Frozen baseline changed')
    create_isolated_schema(target);require_local_tables(target)
    signature=manifest_signature(document)
    with open_database() as c:
        c.execute(sql.SQL('CREATE TABLE IF NOT EXISTS {}.crop_experiment (signature TEXT PRIMARY KEY, settings JSONB NOT NULL)').format(sql.Identifier(target)))
        prior=c.execute(sql.SQL('SELECT signature FROM {}.crop_experiment').format(sql.Identifier(target))).fetchall()
        if prior and prior!=[dict(signature=signature)]:raise ValueError('Experiment configuration changed; use a new schema')
        if not prior:
            if c.execute(sql.SQL('SELECT count(*) AS n FROM {}.visual_embeddings').format(sql.Identifier(target))).fetchone()['n']:
                raise ValueError('Refuse adopting a populated unbound experiment schema')
            c.execute(sql.SQL('INSERT INTO {}.crop_experiment VALUES(%s,%s)').format(sql.Identifier(target)),(signature,Jsonb(document['settings'])))
        c.execute(sql.SQL('INSERT INTO {}.video_recordings SELECT * FROM {}.video_recordings ON CONFLICT(video_id) DO NOTHING').format(sql.Identifier(target),sql.Identifier(baseline)))
        profiles=c.execute(sql.SQL('SELECT * FROM {}.video_recordings EXCEPT SELECT * FROM {}.video_recordings').format(sql.Identifier(target),sql.Identifier(baseline))).fetchall()
        if profiles:raise ValueError('Experiment recording profiles differ')
        c.execute(sql.SQL('INSERT INTO {}.visual_embeddings ('+COPY_COLUMNS+') SELECT '+COPY_COLUMNS+' FROM {}.visual_embeddings ON CONFLICT ON CONSTRAINT visual_sample_encoder_key DO NOTHING').format(sql.Identifier(target),sql.Identifier(baseline)))
        differs=c.execute(sql.SQL('SELECT '+COPY_COLUMNS+' FROM {}.visual_embeddings WHERE variant=\'frame\' EXCEPT SELECT '+COPY_COLUMNS+' FROM {}.visual_embeddings').format(sql.Identifier(target),sql.Identifier(baseline))).fetchall()
        if differs:raise ValueError('Cloned full-frame evidence differs')


def crop_sample(document,crop):
    frame=document['frames'][crop['frame_number']]
    sample={k:frame[k] for k in ('video_id','source_sha256','frame_index','timestamp_seconds',
        'source_camera_id','source_video_path','source_fps','source_duration_seconds','embedding_model','preprocess_version')}
    sample.update(variant='crop',bbox_xyxy=crop['bbox_xyxy'],image_path=crop['image_path'],
        image_sha256=crop['image_sha256'],crop_provenance=dict(
            kind='unreviewed_detector_output',detector='YOLO11n',weights_sha256=document['weights_sha256'],
            manifest_signature=document['manifest_signature'],settings=SETTINGS,class_id=crop['class_id'],
            class_name=crop['class_name'],confidence=crop['confidence'],raw_bbox_xyxy=crop['bbox'],
            parent_image_sha256=frame['image_sha256'],local_track_id=None))
    return sample


def sample_key(sample):
    return (sample['video_id'],sample['frame_index'],tuple(sample['bbox_xyxy']))


def verify_existing_crops(schema,samples,encoder_key):
    """Reject extra or conflicting rows instead of hiding them behind ON CONFLICT."""
    from src.pg_config import open_database
    from psycopg import sql
    expected={sample_key(s):s for s in samples}
    if len(expected)!=len(samples):raise ValueError('Duplicate configured crop samples')
    with open_database(readonly=True) as c:
        rows=c.execute(sql.SQL("SELECT * FROM {}.visual_embeddings WHERE variant='crop'").format(sql.Identifier(schema))).fetchall()
    actual={}
    for row in rows:
        key=sample_key(row);sample=expected.get(key)
        if sample is None or key in actual or row['encoder_key']!=encoder_key:raise ValueError('Unexpected existing crop row')
        for field,value in sample.items():
            if field=='embedding':continue
            actual_value=float(row[field]) if field=='timestamp_seconds' else row[field]
            if actual_value!=value:raise ValueError('Existing crop evidence/provenance differs')
        actual[key]=row
    return actual


def index_crops(manifest,target):
    from PIL import Image
    import torch
    from src.stage10 import _encoder,_identity
    from src.visual_index import VisualIndex
    manifest=Path(manifest)
    if manifest.with_name('index.json').exists():raise FileExistsError('Index report already exists; preserve completed experiment')
    document=json.loads(manifest.read_text(encoding='utf-8'))
    check_schemas(document['baseline_schema'],target)
    rows=baseline_rows(document['baseline_schema'])
    verify_files(rows)
    validate_document(document,rows)
    if _file_hash(document['weights_path'])!=document['weights_sha256']:raise ValueError('Detector weights changed')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    encoder=_encoder();key=_identity(encoder)
    if any(r['encoder_key']!=key for r in rows):raise ValueError('Cached encoder differs from frozen baseline')
    clone_baseline(document['baseline_schema'],target,document)
    index=VisualIndex(schema=target);before=index.stats();started=time.perf_counter();inserted=repeat=0
    all_samples=[crop_sample(document,crop) for crop in document['crops']]
    existing=verify_existing_crops(target,all_samples,key)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for offset in range(0,len(document['crops']),2):
        crops=document['crops'][offset:offset+2];images=[]
        for crop in crops:
            with Image.open(crop['image_path']) as image:images.append(image.convert('RGB'))
        vectors=encoder.encode_images(images);samples=[]
        for crop,vector in zip(crops,vectors):
            sample=crop_sample(document,crop);sample['embedding']=vector
            if sample_key(sample) in existing:
                import numpy as np
                from src.visual_index import normalize_visual_vector
                if not np.allclose(existing[sample_key(sample)]['embedding'],normalize_visual_vector(vector),rtol=1e-5,atol=1e-6):
                    raise ValueError('Existing crop vector differs from recomputed embedding')
            samples.append(sample)
        identity=dict(encoder_key=key,embedding_model=encoder.model_id,preprocess_version=encoder.preprocess_description)
        inserted+=index.store_samples(samples,**identity)
        repeat+=index.store_samples(samples,**identity)
    if repeat:raise ValueError('Duplicate crop insertion changed index')
    if len(verify_existing_crops(target,all_samples,key))!=len(all_samples):raise ValueError('Final crop set incomplete')
    verify_files(rows)
    if snapshot(document['baseline_schema'])!=document['baseline_snapshot']:raise ValueError('Baseline changed')
    result=dict(status='indexed',schema=target,manifest_signature=document['manifest_signature'],counts_before=before,
        counts_after=index.stats(),inserted_crops=inserted,repeat_inserted=repeat,elapsed_seconds=time.perf_counter()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    with Path(manifest).with_name('index.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('detect');p.add_argument('--baseline',required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int)
    p=sub.add_parser('index');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--target',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result=detect(**args) if command=='detect' else index_crops(**args)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
