"""Reproducible local G507 inventory and discovery contact sheets; no inference."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def filename_times(name):
    match=re.fullmatch(r'(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.[^.]+\.([^.]+)\.[^.]+\.avi',name,re.I)
    if not match:
        raise ValueError('Unrecognized dataset filename')
    date,start,end,camera=match.groups()
    a=datetime.strptime(date+'.'+start,'%Y-%m-%d.%H-%M-%S')
    b=datetime.strptime(date+'.'+end,'%Y-%m-%d.%H-%M-%S')
    if b<a:
        b+=timedelta(days=1)
    if b==a:
        raise ValueError('Empty filename time range')
    return a,b,camera


def timeline(rows):
    result=[];previous=None
    for row in sorted(rows,key=lambda r:r['filename_start']):
        current=dict(row)
        a=datetime.fromisoformat(row['filename_start'])
        current['filename_gap_seconds']=None if previous is None else (a-datetime.fromisoformat(previous['filename_end'])).total_seconds()
        current['media_gap_seconds']=None if previous is None else (a-(datetime.fromisoformat(previous['filename_start'])+timedelta(seconds=previous['duration_seconds']))).total_seconds()
        result.append(current);previous=row
    return result


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def probe(path,*,full_decode=False):
    import cv2
    a,b,camera=filename_times(path.name)
    digest=sha256(path)
    raw=subprocess.run([shutil.which('ffprobe') or 'ffprobe','-v','error','-show_streams','-show_format',
                        '-of','json',str(path)],capture_output=True,text=True,check=True,timeout=30)
    metadata=json.loads(raw.stdout)
    stream=next(s for s in metadata['streams'] if s['codec_type']=='video')
    fps=float(Fraction(stream['avg_frame_rate']))
    duration=float(stream.get('duration') or metadata['format']['duration'])
    cap=cv2.VideoCapture(str(path))
    ok,frame=cap.read();cap.release()
    result=dict(filename=path.name,path=str(path.resolve()),sha256=digest,file_size_bytes=path.stat().st_size,
        filename_start=a.isoformat(),filename_end=b.isoformat(),camera_id=camera,
        timezone_status='not established; filename wall clock only',duration_seconds=duration,
        nominal_duration_seconds=(b-a).total_seconds(),width=stream['width'],height=stream['height'],
        codec=stream['codec_name'],fps=fps,r_frame_rate=stream['r_frame_rate'],avg_frame_rate=stream['avg_frame_rate'],
        declared_frames=int(stream['nb_frames']) if stream.get('nb_frames','').isdigit() else None,
        first_frame_decoded=bool(ok and frame is not None),full_decode_status='not_run',
        filename_duration_difference_seconds=duration-(b-a).total_seconds())
    if full_decode:
        started=time.perf_counter()
        try:
            run=subprocess.run([shutil.which('ffmpeg') or 'ffmpeg','-v','error','-xerror','-threads','2',
                '-i',str(path),'-map','0:v:0','-an','-progress','pipe:1','-nostats','-f','null','-'],
                capture_output=True,text=True,timeout=300)
            frames=re.findall(r'^frame=(\d+)',run.stdout,re.M)
            result.update(full_decode_status='passed' if run.returncode==0 else 'failed',
                          decoded_frames=int(frames[-1]) if frames else None,
                          decode_error=run.stderr[-2000:] or None)
            if result['declared_frames'] is not None and result['decoded_frames']!=result['declared_frames']:
                result['full_decode_status']='frame_count_mismatch'
        except subprocess.TimeoutExpired:
            result['full_decode_status']='timeout'
        result['full_decode_seconds']=time.perf_counter()-started
    if sha256(path)!=digest:
        raise ValueError('Original source changed during inventory')
    return result


def contact_sheet(path,destination,*,times=None,columns=3,width=640):
    import cv2
    from PIL import Image,ImageDraw
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps<=0 or count<=0:
            raise ValueError('No decodable video')
        duration=count/fps
        times=times if times is not None else [i*duration/9 for i in range(9)]
        tiles=[];evidence=[]
        for timestamp in times:
            if not 0<=timestamp<duration:
                raise ValueError('Frame timestamp outside recording')
            index=min(count-1,round(timestamp*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,pixels=cap.read()
            if not ok:
                raise ValueError('Contact sheet frame decode failed')
            actual=max(0,round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1)
            image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
            height=round(width*image.height/image.width)
            image=image.resize((width,height))
            tile=Image.new('RGB',(width,height+28),'#14212a');tile.paste(image,(0,28))
            ImageDraw.Draw(tile).text((8,7),f'{path.name[:19]} | frame {actual} | t={actual/fps:.3f}s',fill='white')
            tiles.append(tile);evidence.append(dict(frame_index=actual,timestamp_seconds=actual/fps))
        sheet=Image.new('RGB',(columns*tiles[0].width,((len(tiles)+columns-1)//columns)*tiles[0].height),'#14212a')
        for i,tile in enumerate(tiles):
            sheet.paste(tile,((i%columns)*tile.width,(i//columns)*tile.height))
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        sheet.save(destination,quality=92)
        destination.with_suffix('.json').write_text(json.dumps(dict(source_path=str(path.resolve()),
            source_sha256=sha256(path),frames=evidence,note='Discovery samples only, not exhaustive presence/absence labels'),indent=2))
        return evidence
    finally:
        cap.release()


def inventory(root,output,*,full_decode=False):
    root=Path(root).resolve();output=Path(output).resolve()
    if output==root:
        raise ValueError('Use a separate derived artifact directory')
    output.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob('*.avi'))
    if not paths:
        raise ValueError('No AVI recordings found')
    rows=[]
    for path in paths:
        row=probe(path,full_decode=full_decode)
        sheet=output/'contact_sheets'/(path.stem+'.jpg')
        contact_sheet(path,sheet)
        row['contact_sheet']=str(sheet)
        rows.append(row)
        result=dict(format='group8-stage13-inventory-v1',root=str(root),recordings=timeline(rows),
                    originals_modified=False,observed_files=len(rows),total_duration_seconds=sum(r['duration_seconds'] for r in rows))
        (output/'inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(f"{len(rows)}/{len(paths)} {path.name}: {row['full_decode_status']}",flush=True)
    fields=[k for k in result['recordings'][0] if k not in {'path','contact_sheet'}]
    with (output/'inventory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['recordings'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-decode',action='store_true')
    args=parser.parse_args()
    inventory(args.root,args.output,full_decode=args.full_decode)


if __name__=='__main__':
    main()
