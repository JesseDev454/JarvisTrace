"""Stage 10: explicit migration, raw-frame/crop CLIP indexing and pgvector search.

Indexing expects a registered local constant-frame-rate video. No captions or
tracker identities are treated as ground truth. Use --help for Windows CLI use.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import uuid

from src.visual_index import VisualIndex, encoder_identity


def _file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def _encoder():
    from src.clip_encoder import OpenCLIPEmbedder
    return OpenCLIPEmbedder.get_instance()


def _identity(encoder):
    if encoder.embedding_dim != 512:
        raise ValueError('This index requires a 512-dimensional CLIP model')
    return encoder_identity(encoder.model_id,encoder.preprocess_description,
                            version('open_clip_torch'),encoder.embedding_dim)


def _save_image(pixels,root):
    import cv2
    ok,encoded = cv2.imencode('.png',pixels)
    if not ok:
        raise ValueError('Could not encode evidence image')
    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    root.mkdir(parents=True,exist_ok=True)
    path = root/(digest+'.png')
    try:
        with path.open('xb') as f:
            f.write(payload)
    except FileExistsError:
        if _file_hash(path) != digest:
            raise ValueError('Existing evidence image content differs from its hash')
    return str(path.resolve()),digest


def _tracking_rows(path,video_id,camera_id,source_hash,metadata):
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report.get('report_format') != 'group8-stage9-v1' or report.get('video_id') != video_id
            or report.get('camera_id') != camera_id
            or report.get('provenance',{}).get('video_sha256') != source_hash
            or report.get('video') != metadata):
        raise ValueError('Tracking report does not match this video/camera/content/metadata')
    rows = [json.loads(line) for line in path.with_name('observations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    grouped = {}
    if len(rows) != report['analysis']['observation_count']:
        raise ValueError('Tracking observation count mismatch')
    for row in rows:
        f = row['frame_index']
        if (type(f) is not int or not 0<=f<metadata['frame_count']
                or row.get('scope_id') != report['provenance']['scope_id']
                or row.get('class_id') != report['target_class_id']
                or not math.isfinite(row['timestamp_seconds'])
                or abs(row['timestamp_seconds']-f/metadata['fps'])>1e-5):
            raise ValueError('Tracking row provenance/time mismatch')
        grouped.setdefault(f,[]).append(row)
    return grouped,hashlib.sha256(raw).hexdigest(),_file_hash(path.with_name('observations.jsonl'))


def index_recording(video_id,*,every_seconds=1.0,max_frames=300,batch_size=8,
                    tracking_report=None,output_root=Path('outputs/stage10'),encoder=None,index=None,
                    expected_source_sha256=None):
    import cv2
    from PIL import Image
    if (not math.isfinite(every_seconds) or every_seconds<=0 or type(max_frames) is not int
            or max_frames<1 or type(batch_size) is not int or not 1<=batch_size<=64):
        raise ValueError('Sampling interval/max frames must be positive; batch size 1..64')
    index = index or VisualIndex()
    recording = index.recording(video_id)
    source = Path(recording['local_video_path'])
    if not source.is_file():
        raise FileNotFoundError('Registered source video is missing')
    digest = _file_hash(source)
    if expected_source_sha256 is not None and digest != expected_source_sha256:
        raise ValueError('Source bytes differ from the queued source hash')
    cap = cv2.VideoCapture(str(source.resolve()))
    samples,frame_count,observed_frames = [],0,0
    started = time.perf_counter()
    root = Path(output_root)
    run_dir = root/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    report_hash = observation_hash = None
    try:
        if not cap.isOpened():
            raise ValueError('Source could not be decoded')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width,height,frame_count = (int(cap.get(prop)) for prop in
            (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_COUNT))
        if (not math.isfinite(fps) or fps<=0 or min(width,height,frame_count)<=0
                or abs(fps-float(recording['fps']))>1e-4
                or abs(frame_count/fps-float(recording['duration_seconds']))>max(.01,1/fps)):
            raise ValueError('Usable video metadata must match the registered recording; no FPS is invented')
        metadata = dict(fps=fps,width=width,height=height,frame_count=frame_count)
        stride = max(1,round(every_seconds*fps))
        tracking = {}
        if tracking_report is not None:
            tracking,report_hash,observation_hash = _tracking_rows(tracking_report,video_id,
                        recording['camera_id'],digest,metadata)
        i=0
        while i<frame_count and observed_frames<max_frames:
            if not cap.grab():
                raise ValueError(f'Source ended before declared frame count at frame {i}')
            if i%stride==0:
                ok,pixels=cap.retrieve()
                if not ok:
                    raise ValueError(f'Failed to decode frame {i}')
                variants=[('frame',pixels,None,None)]
                for row in tracking.get(i,[]):
                    bbox = row['bbox_xyxy']
                    if (len(bbox)!=4 or not all(math.isfinite(v) for v in bbox)
                            or not (0<=bbox[0]<bbox[2]<=width and 0<=bbox[1]<bbox[3]<=height)):
                        raise ValueError('Tracking crop outside source frame')
                    x1,y1,x2,y2=math.floor(bbox[0]),math.floor(bbox[1]),math.ceil(bbox[2]),math.ceil(bbox[3])
                    provenance=dict(tracking_report_sha256=report_hash,observations_sha256=observation_hash,
                        class_id=row['class_id'],local_track_id=row['local_track_id'],
                        detector_confidence=row['confidence'],review_status='unreviewed_detector_output')
                    variants.append(('crop',pixels[y1:y2,x1:x2],[x1,y1,x2,y2],provenance))
                for variant,picture,bbox,provenance in variants:
                    path,image_hash=_save_image(picture,root/'images'/digest)
                    samples.append(dict(video_id=video_id,source_sha256=digest,frame_index=i,
                        source_camera_id=recording['camera_id'],source_video_path=recording['local_video_path'],
                        source_fps=fps,source_duration_seconds=frame_count/fps,
                        timestamp_seconds=i/fps,variant=variant,bbox_xyxy=bbox,image_path=path,
                        image_sha256=image_hash,crop_provenance=provenance))
                observed_frames+=1
            i+=1
    finally:
        cap.release()
    if _file_hash(source)!=digest:
        raise ValueError('Source changed during decoding; no embeddings were inserted')
    if not samples:
        raise ValueError('No frames sampled')
    decode_seconds=time.perf_counter()-started
    load_started=time.perf_counter()
    encoder=encoder or _encoder()
    key=_identity(encoder)
    load_seconds=time.perf_counter()-load_started
    run_dir.mkdir(parents=True,exist_ok=False)
    report_path=run_dir/'index_report.json'
    report=dict(report_format='group8-stage10-index-v1',video_id=video_id,camera_id=recording['camera_id'],
        source_sha256=digest,video=metadata,encoder_key=key,embedding_model=encoder.model_id,
        preprocessing=encoder.preprocess_description,open_clip_version=version('open_clip_torch'),
        device=encoder.device_str,sampled_frames=observed_frames,
        full_frame_samples=sum(s['variant']=='frame' for s in samples),crop_samples=sum(s['variant']=='crop' for s in samples),
        requested_interval_seconds=every_seconds,actual_interval_seconds=stride/fps,
        sampling_complete=observed_frames==len(range(0,frame_count,stride)),
        last_sample_seconds=samples[-1]['timestamp_seconds'],max_frames=max_frames,
        decoded_sample_sha256=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
        decode_seconds=decode_seconds,encoder_load_seconds=load_seconds,inserted_embeddings=0,
        completed_samples=0,status='indexing',report_path=str(report_path.resolve()))
    inference_started=time.perf_counter()
    try:
        for start in range(0,len(samples),batch_size):
            batch=samples[start:start+batch_size]
            images=[]
            for sample in batch:
                with Image.open(sample['image_path']) as image:
                    images.append(image.convert('RGB'))
            vectors=encoder.encode_images(images)
            if len(vectors)!=len(batch):
                raise ValueError('Encoder returned an incorrect batch length')
            rows=[dict(sample,embedding=vector) for sample,vector in zip(batch,vectors)]
            report['inserted_embeddings']+=index.store_samples(rows,encoder_key=key,
                            embedding_model=encoder.model_id,preprocess_version=encoder.preprocess_description)
            report['completed_samples']+=len(batch)
            report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        report['status']='complete'
    except Exception as exc:
        report['status']='failed_partial_index'
        report['failure_type']=type(exc).__name__
        raise
    finally:
        report['embedding_and_storage_seconds']=time.perf_counter()-inference_started
        report['elapsed_seconds']=time.perf_counter()-started
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('migrate',help='Inspect/adopt existing schema and add visual index explicitly')
    commands.add_parser('stats',help='Show visual index counts')
    indexing=commands.add_parser('index',help='Generate image embeddings outside the API')
    indexing.add_argument('--video-id',required=True)
    indexing.add_argument('--every-seconds',type=float,default=1)
    indexing.add_argument('--max-frames',type=int,default=300)
    indexing.add_argument('--batch-size',type=int,default=8)
    indexing.add_argument('--tracking-report',type=Path)
    indexing.add_argument('--confirm-cfr',action='store_true',required=True,help='Confirm a constant-frame-rate source')
    searching=commands.add_parser('search',help='Encode text with matching CLIP and rank indexed images')
    searching.add_argument('--query',required=True)
    searching.add_argument('--limit',type=int,default=5)
    searching.add_argument('--camera')
    searching.add_argument('--video-id')
    searching.add_argument('--start-seconds',type=float)
    searching.add_argument('--end-seconds',type=float)
    searching.add_argument('--variant',choices=['frame','crop'])
    args=parser.parse_args()
    try:
        if args.command=='migrate':
            from src.pg_config import migrate
            migrate()
            result={'status':'migrated','legacy_tables':'preserved','visual_embedding_dimension':512}
        elif args.command=='stats':
            result=VisualIndex().stats()
        elif args.command=='index':
            result=index_recording(args.video_id,every_seconds=args.every_seconds,max_frames=args.max_frames,
                                   batch_size=args.batch_size,tracking_report=args.tracking_report)
        else:
            encoder=_encoder()
            begin=time.perf_counter()
            rows=VisualIndex().search(encoder.encode_text(args.query),_identity(encoder),limit=args.limit,
                    camera_id=args.camera,video_id=args.video_id,start_seconds=args.start_seconds,
                    end_seconds=args.end_seconds,variant=args.variant)
            result=dict(query=args.query,results=rows,query_seconds=time.perf_counter()-begin,
                        score_meaning='CLIP cosine similarity, not calibrated confidence or proof of absence',
                        temporal_limit='Frame/crop appearance retrieval; no automatic action understanding')
    except Exception as exc:
        # Do not expose connection strings or credentials in command failures.
        parser.exit(1,f'ERROR: {type(exc).__name__}; operation failed. Check local configuration, inputs and schema.\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
