"""Extract derived clips and independent-review drafts without modifying originals."""
from __future__ import annotations
import argparse
from datetime import timedelta
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from src.stage13_dataset import contact_sheet,filename_times,probe,sha256


def validate_segment(segment,source_duration):
    if not re.fullmatch('[a-zA-Z0-9_-]+',segment['clip_id']):
        raise ValueError('Unsafe clip ID')
    if Path(segment['source_filename']).name!=segment['source_filename'] or '/' in segment['source_filename'] or '\\' in segment['source_filename']:
        raise ValueError('Source must be a filename')
    start,duration=segment['start_seconds'],segment['duration_seconds']
    if (not all(type(v) in (int,float) and math.isfinite(v) for v in (start,duration))
            or start<0 or not 20<=duration<=60 or start+duration>source_duration):
        raise ValueError('Clip must be 20..60 seconds within source')
    return start,duration


def frame_extraction(source,target,start,duration,fps):
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Usable CFR required')
    first=round(start*fps);count=round(duration*fps)
    if first<0 or count<=0:
        raise ValueError('Invalid frame range')
    rate=format(fps,'.12g')
    spec=dict(timeline='decoded_frame_index_cfr_v2',source_frame_start=first,frame_count=count,
              source_offset_seconds=first/fps,duration_seconds=count/fps,fps=fps,
              codec='libx264',crf=20,preset='veryfast',audio='omitted')
    command=[shutil.which('ffmpeg') or 'ffmpeg','-v','error','-nostdin','-n','-threads','2',
        '-i',str(source),'-map','0:v:0','-an','-vf',
        f'trim=start_frame={first}:end_frame={first+count},setpts=N/({rate}*TB)',
        '-frames:v',str(count),'-c:v','libx264','-preset','veryfast','-crf','20','-threads','2',
        '-r',rate,'-fps_mode','cfr','-movflags','+faststart',str(target)]
    return spec,command


def prepare(root,output,selections):
    root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(selections).read_text(encoding='utf-8'))
    if config.get('format')!='group8-stage13-candidates-v1':
        raise ValueError('Unknown selections format')
    clips=[];cards=[]
    for segment in config['segments']:
        name=segment['source_filename']
        if Path(name).name!=name or '/' in name or '\\' in name:
            raise ValueError('Source path is not a filename')
        source=(root/name).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Source outside dataset')
        metadata=probe(source)
        start,duration=validate_segment(segment,metadata['duration_seconds'])
        video_id=segment['clip_id']+'_'+metadata['sha256'][:12]+'_f2'
        clips_root=output/'clips';clips_root.mkdir(exist_ok=True)
        bundle=clips_root/video_id
        target=bundle/'video.mp4'
        sidecar=bundle/'provenance.json'
        extraction,_=frame_extraction(source,target,start,duration,metadata['fps'])
        extraction['source_sha256']=metadata['sha256']
        start=extraction['source_offset_seconds'];duration=extraction['duration_seconds']
        if bundle.exists():
            if not target.exists() or not sidecar.exists():
                raise ValueError('Existing clip lacks provenance sidecar')
            existing=json.loads(sidecar.read_text())
            if existing['extraction']!=extraction or existing['clip_sha256']!=sha256(target):
                raise ValueError('Existing derivative differs; use another output directory')
        else:
            # Publish video and provenance together by renaming one completed
            # directory. A timeout leaves no deterministic partial filename.
            with tempfile.TemporaryDirectory(prefix='.prepare_',dir=clips_root) as staging:
                staging=Path(staging);temporary=staging/'video.mp4'
                legacy=clips_root/(video_id+'.mp4');legacy_sidecar=legacy.with_suffix('.json')
                if legacy.exists() and legacy_sidecar.exists():
                    old=json.loads(legacy_sidecar.read_text())
                    if old['extraction']!=extraction or old['clip_sha256']!=sha256(legacy):
                        raise ValueError('Legacy derivative provenance differs')
                    shutil.copyfile(legacy,temporary)
                    provenance=old
                else:
                    _,command=frame_extraction(source,temporary,start,duration,metadata['fps'])
                    subprocess.run(command,check=True,capture_output=True,text=True,timeout=180)
                    provenance=dict(extraction=extraction,clip_sha256=sha256(temporary),
                        command=command,ffmpeg_version=subprocess.run([shutil.which('ffmpeg') or 'ffmpeg','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0])
                (staging/'provenance.json').write_text(json.dumps(provenance,indent=2))
                staging.rename(bundle)
        if sha256(source)!=metadata['sha256']:
            raise ValueError('Original source changed during extraction')
        import cv2
        cap=cv2.VideoCapture(str(target));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if abs(frames/fps-duration)>1/fps:
            raise ValueError('Derived timeline differs from requested interval')
        sheet=output/'previews'/(video_id+'.jpg')
        contact_sheet(target,sheet,times=[i*duration/12 for i in range(12)],columns=3)
        recording_start,_,camera=filename_times(source.name)
        clip=dict(video_id=video_id,clip_id=segment['clip_id'],clip_path=str(target),
            source_sha256=sha256(target),original_source_sha256=metadata['sha256'],
            source_filename=source.name,source_path=str(source),camera_id=camera,
            recording_date=recording_start.date().isoformat(),filename_recording_start=recording_start.isoformat(),
            filename_clip_start=(recording_start+timedelta(seconds=start)).isoformat(),
            timezone_status='unknown; do not treat as UTC',source_offset_seconds=start,
            source_frame_start=extraction['source_frame_start'],timeline=extraction['timeline'],
            duration_seconds=frames/fps,fps=fps,frame_count=frames,preview_path=str(sheet),
            proposed_observation=segment['observed'],scenario=segment['scenario'],
            discovery_source_times_seconds=segment['evidence_seconds'],review_status='proposed',
            expected_visibility_start_seconds=None,expected_visibility_end_seconds=None,
            occlusion_status='not_verified',event_status='requires_full_motion_review')
        clips.append(clip)
        cards.append('<article><h2>'+html.escape(segment['clip_id'])+'</h2><p>'+html.escape(segment['observed'])+'</p><p>'+html.escape(source.name)+'</p><p>Source offset '+str(start)+'s; clip times start at 0. '+html.escape(segment['scenario'])+'</p><video aria-label="'+html.escape(segment['clip_id'])+'" controls preload="none" src="'+target.relative_to(output).as_posix()+'"></video><details><summary>Sampled preview (not exhaustive)</summary><img alt="Sampled preview for '+html.escape(segment['clip_id'])+'" src="previews/'+sheet.name+'"></details></article>')
        print(f'Prepared {video_id}: {start}..{start+duration}s',flush=True)
    ids=[c['video_id'] for c in clips]
    proposed=[('q01','A person carrying a dark backpack.','backpack',True),
              ('q02','A dark bag resting on a bench.','bag',True),
              ('q03','A person carrying a shoulder bag.','shoulder bag',True),
              ('q04','A person carrying a blue handled case.','luggage/case',True),
              ('q05','A bag being put down beside a bench.','bag',True),
              ('q06','A person picking up a handled case.','luggage/case',True),
              ('q07','A red wheeled suitcase in the waiting room.','suitcase',False),
              ('q08','A laptop placed on a bench.','laptop',False)]
    queries=[dict(query_id=q,query=text,target_category=category,present=present,
                  review_status='proposed',reviewer=None,independent_of_predictions=True,
                  scope_video_ids=ids,scope_review_complete=False,relevant_intervals=[],
                  notes='Hypothesis for human review, not an established event/absence. Review all seven clips; correct or reject unsupported queries.')
             for q,text,category,present in proposed]
    manifest=dict(format='group8-stage13-annotations-v1',clips=clips,queries=queries,
                  annotation_basis=config['selection_basis'],source_predictions_used=False)
    annotation_path=output/'annotations.proposed.json'
    # Never replace human edits; a rerun can validate/reuse clips and regenerate previews.
    if not annotation_path.exists():
        annotation_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'clips_manifest.json').write_text(json.dumps(dict(format='group8-stage13-clips-v1',clips=clips),indent=2))
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>G507 independent annotation review</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:20px}video,img{width:100%}article{border-top:1px solid #aaa;margin-top:30px;padding-top:20px}</style><h1>G507 annotation review</h1><p>All observations and query labels are PROPOSED. Watch continuous clips before marking verified. Do not consult retrieval predictions. Record every relevant interval across all seven clips; confirm negatives over the entire collection. Review annotations.proposed.json, save a separate annotations.verified.json and preserve the draft.</p>'+''.join(cards)+'</html>',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selections',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.root,args.output,args.selections)


if __name__=='__main__':
    main()
