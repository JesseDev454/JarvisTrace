"""Local read-only search and verified source playback. Run python -m src.stage12."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timedelta, timezone
import hmac
import math
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles
from dotenv import dotenv_values

from src.pg_config import ROOT, open_database
from src.stage11 import search
from src.stage12_ingest import (MAX_UPLOAD_BYTES, parse_browser_datetime, segment_ids,
    segment_start_from_record_path, sha256_file, store_upload)
from src.stage12_workspace import (PostgresWorkspaceStore, new_session_token,
    normalize_email, session_token_hash)

APP_ENV=dotenv_values(ROOT/'scene-search-prototype/.env')


def app_setting(name,default=None):
    return os.environ.get(name,APP_ENV.get(name,default))


def configured_similarity_threshold(name):
    value=app_setting(name)
    if value is None or not value.strip():
        return None
    try:
        threshold=float(value)
    except (TypeError,ValueError):
        raise ValueError(f'{name} must be a number from -1 to 1') from None
    if not math.isfinite(threshold) or not -1<=threshold<=1:
        raise ValueError(f'{name} must be a number from -1 to 1')
    return threshold


def configured_path(value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def source_key(row):
    return (row['video_id'],row['camera_id'],row['source_sha256'])


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verified_snapshot(item,roots):
    """Private temporary snapshot: hashed bytes are the bytes sent to the browser."""
    snapshot=tempfile.SpooledTemporaryFile(max_size=8*1024*1024,mode='w+b')
    try:
        path=item['path'].resolve()
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError('Source outside storage roots')
        digest=hashlib.sha256()
        with path.open('rb') as source:
            while block:=source.read(1024*1024):
                digest.update(block)
                snapshot.write(block)
        if digest.hexdigest()!=item['source_sha256']:
            raise ValueError('Source changed')
        snapshot.seek(0)
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def byte_range(header,size):
    """Single byte ranges, including open-ended and suffix forms."""
    if header is None:
        return 0,size-1,200
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if match and size>0:
        a,b=match.groups()
        if a:
            start=int(a); end=min(int(b),size-1) if b else size-1
        elif b and int(b)>0:
            start=max(0,size-int(b)); end=size-1
        else:
            start,end=1,0
        if 0<=start<=end<size:
            return start,end,206
    raise HTTPException(416,'Requested byte range is unavailable',headers={'Content-Range':f'bytes */{size}'})


class UploadBodyTooLarge(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Bound upload request bytes even when the client omits Content-Length."""
    def __init__(self, app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path')!='/api/recordings/upload' or scope.get('method')!='POST':
            await self.app(scope,receive,send)
            return
        limit=MAX_UPLOAD_BYTES+2*1024*1024
        headers=dict(scope.get('headers',()))
        try:
            declared=int(headers.get(b'content-length',b'0'))
        except ValueError:
            declared=limit+1
        if declared>limit:
            await self._too_large(send)
            return
        consumed=0
        response_started=False

        async def limited_receive():
            nonlocal consumed
            message=await receive()
            if message['type']=='http.request':
                consumed+=len(message.get('body',b''))
                if consumed>limit:
                    raise UploadBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message['type']=='http.response.start':
                response_started=True
            await send(message)

        try:
            await self.app(scope,limited_receive,tracked_send)
        except UploadBodyTooLarge:
            if not response_started:
                await self._too_large(send)

    @staticmethod
    async def _too_large(send):
        body=b'{"detail":"The file is larger than the 1 GB upload limit"}'
        await send({'type':'http.response.start','status':413,'headers':[
            (b'content-type',b'application/json'),(b'content-length',str(len(body)).encode()),
            (b'x-content-type-options',b'nosniff'),(b'cache-control',b'no-store')]})
        await send({'type':'http.response.body','body':body})


class PlaybackCatalog:
    """Startup snapshot, restricted to verified indexed files inside storage roots."""
    def __init__(self,rows,roots):
        self.items={}
        roots=[Path(p).resolve() for p in roots]
        self.roots=roots
        groups={}
        for row in rows:
            groups.setdefault(source_key(row),[]).append(row)
        self.unavailable=0
        self._last_refresh=time.monotonic()
        for key,profiles in groups.items():
            try:
                paths={Path(r['source_video_path']).resolve() for r in profiles}
                durations={float(r['duration_seconds']) for r in profiles}
                if len(paths)!=1 or len(durations)!=1:
                    raise ValueError('Ambiguous pinned source')
                path=paths.pop(); duration=durations.pop()
                if (not math.isfinite(duration) or duration<=0 or not path.is_file()
                        or not any(path.is_relative_to(root) for root in roots)
                        or file_hash(path)!=key[2]):
                    raise ValueError('Source unavailable')
                media_id=hashlib.sha256('\0'.join(key).encode()).hexdigest()
                self.items[media_id]=dict(video_id=key[0],camera_id=key[1],source_sha256=key[2],
                                         path=path,duration_seconds=duration)
            except (OSError,ValueError,TypeError):
                self.unavailable+=1

    @classmethod
    def from_database(cls,roots):
        rows=cls.database_rows()
        return cls(rows,roots)

    @staticmethod
    def database_rows():
        with open_database(readonly=True) as c:
            return c.execute('''SELECT DISTINCT video_id,source_camera_id AS camera_id,
                source_sha256,source_video_path,source_duration_seconds AS duration_seconds
                FROM visual_embeddings WHERE source_camera_id IS NOT NULL''').fetchall()

    def refresh_if_due(self, *, force=False, interval_seconds=3):
        if not force and time.monotonic()-self._last_refresh < interval_seconds:
            return
        rows=self.database_rows()
        grouped={}
        for row in rows:
            grouped.setdefault(source_key(row),[]).append(row)
        for key,profiles in grouped.items():
            paths={Path(r['source_video_path']).resolve() for r in profiles}
            durations={float(r['duration_seconds']) for r in profiles}
            if len(paths)!=1 or len(durations)!=1:
                self.items.pop(hashlib.sha256('\0'.join(key).encode()).hexdigest(),None)
                self.unavailable+=1
                continue
            path=paths.pop(); duration=durations.pop()
            media_id=hashlib.sha256('\0'.join(key).encode()).hexdigest()
            prior=self.items.get(media_id)
            if prior is not None:
                if prior['path']!=path or prior['duration_seconds']!=duration:
                    self.items.pop(media_id,None)
                    self.unavailable+=1
                continue
            added=PlaybackCatalog(profiles,self.roots)
            self.items.update(added.items)
            self.unavailable+=added.unavailable
        self._last_refresh=time.monotonic()

    def recordings(self):
        return [dict(media_id=k,video_id=v['video_id'],camera_id=v['camera_id'],
                     duration_seconds=v['duration_seconds'],playback_url='/media/'+k)
                for k,v in sorted(self.items.items(),key=lambda item:(item[1]['camera_id'],item[1]['video_id']))]

    def media_id_for(self,row):
        for key,value in self.items.items():
            if source_key(value)==source_key(row):
                return key
        return None


class SearchRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    query:str=Field(min_length=1,max_length=1000)
    mode:Literal['hybrid','visual','text']='hybrid'
    camera_id:str|None=Field(default=None,max_length=120)
    video_id:str|None=Field(default=None,max_length=120)
    start_seconds:float|None=Field(default=None,ge=0)
    end_seconds:float|None=Field(default=None,gt=0)
    limit:int=Field(default=10,ge=1,le=20)

    @field_validator('query')
    @classmethod
    def clean_query(cls,value):
        value=value.strip()
        if not value:
            raise ValueError('Enter a search description')
        return value

    @model_validator(mode='after')
    def check_times(self):
        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds<=self.start_seconds:
            raise ValueError('End time must follow start time')
        return self


class SetupAdminRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    display_name:str=Field(min_length=1,max_length=120)
    email:str=Field(min_length=3,max_length=320)
    password:str=Field(min_length=12,max_length=256)

    @field_validator('display_name')
    @classmethod
    def clean_name(cls,value):
        if not value.strip(): raise ValueError('Name cannot be blank')
        return value.strip()


class LoginRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email:str=Field(min_length=3,max_length=320)
    password:str=Field(min_length=1,max_length=256)


class UserCreateRequest(SetupAdminRequest):
    role:Literal['administrator','investigator']='investigator'
    cameras:list[str]=Field(default_factory=list,max_length=100)


class CameraCreateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    camera_id:str=Field(min_length=1,max_length=80)
    name:str=Field(min_length=1,max_length=120)
    source_type:Literal['upload','mediamtx']='mediamtx'
    mtx_path:str|None=Field(default=None,max_length=120)


class MediaMTXSegmentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    path:str=Field(min_length=1,max_length=120)
    segment_path:str=Field(min_length=1,max_length=2048)


def public_evidence(value):
    if isinstance(value,dict):
        return {k:public_evidence(v) for k,v in value.items()
                if k not in {'source_video_path','image_path','report_path'}}
    if isinstance(value,list):
        return [public_evidence(v) for v in value]
    return value


class EvidenceCatalog:
    """Short-lived opaque references to immutable, hash-verified image evidence."""
    def __init__(self, roots):
        self.roots=[Path(p).resolve() for p in roots]
        self.items={}
        self.lock=threading.Lock()

    def register(self, evidence, camera_id):
        path=evidence.get('image_path'); digest=evidence.get('image_sha256')
        if not path or not isinstance(digest,str) or not re.fullmatch(r'[a-f0-9]{64}',digest):
            return
        candidate=Path(path).resolve()
        if not any(candidate.is_relative_to(root) for root in self.roots):
            return
        asset_id=uuid4().hex
        with self.lock:
            if len(self.items)>=5000:
                self.items.clear()
            self.items[asset_id]=dict(path=candidate,sha256=digest,camera_id=camera_id)
        evidence['evidence_url']='/evidence/'+asset_id

    def get(self, asset_id):
        with self.lock:
            return self.items.get(asset_id)


def create_app(*,catalog=None,search_fn=None,roots=None,store=None,storage_root=None,video_probe=None):
    supplied_catalog=catalog is not None
    managed_root=configured_path(storage_root or app_setting('JARVISTRACE_STORAGE_ROOT','scene-search-prototype/videos/managed'))
    allowed_roots=[Path(p).resolve() for p in (roots or [ROOT/'scene-search-prototype/videos'])]
    if managed_root not in allowed_roots:
        allowed_roots.append(managed_root)
    catalog=catalog if supplied_catalog else PlaybackCatalog.from_database(allowed_roots)
    search_fn=search_fn or search
    store=store or PostgresWorkspaceStore()
    evidence_catalog=EvidenceCatalog([ROOT/'outputs/stage10/images',*allowed_roots])
    lock=threading.Lock()
    app=FastAPI(title='JarvisTrace Search',docs_url=None,redoc_url=None,openapi_url=None)
    app.state.store=store
    app.state.catalog=catalog
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','[::1]'])
    app.add_middleware(UploadBodyLimitMiddleware)

    @app.middleware('http')
    async def local_only(request:Request,call_next):
        if request.client is None or request.client.host not in {'127.0.0.1','::1'}:
            return JSONResponse({'detail':'Local access only'},status_code=403)
        origin=request.headers.get('origin')
        if (origin and origin!=str(request.base_url).rstrip('/')) or request.headers.get('sec-fetch-site')=='cross-site':
            return JSONResponse({'detail':'Same-origin access required'},status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    def current_user(request):
        raw=request.cookies.get('jarvistrace_session')
        if not raw:
            raise HTTPException(401,'Sign in to use JarvisTrace')
        try:
            user=store.session_user(session_token_hash(raw))
        except Exception:
            raise HTTPException(503,'Account service unavailable. Check the local database configuration.') from None
        if not user:
            raise HTTPException(401,'Your session has expired. Sign in again.')
        return user

    def admin_user(request):
        user=current_user(request)
        if user['role']!='administrator':
            raise HTTPException(403,'Administrator access is required')
        return user

    def issue_session(user,request,response):
        token=new_session_token()
        try:
            store.create_session(user['id'],session_token_hash(token),datetime.now(timezone.utc)+timedelta(hours=12))
        except Exception:
            raise HTTPException(503,'Could not start a session. Check the local database configuration.') from None
        response.set_cookie('jarvistrace_session',token,max_age=12*60*60,httponly=True,
            secure=request.url.scheme=='https',samesite='strict',path='/')

    def media_scope(user,item):
        if user['role']!='administrator' and item['camera_id'] not in user.get('cameras',[]):
            raise HTTPException(403,'This recording is outside your camera access')

    def refresh_catalog():
        if not supplied_catalog:
            try:
                catalog.refresh_if_due()
            except Exception:
                raise HTTPException(503,'Recording catalog unavailable. Check the local database and index.') from None

    @app.get('/',response_class=HTMLResponse)
    def page():
        built=ROOT/'frontend/dist/index.html'
        if built.is_file():
            return HTMLResponse(built.read_text(encoding='utf-8'))
        return HTMLResponse(Path(__file__).with_name('stage12.html').read_text(encoding='utf-8'))

    @app.get('/stage12',response_class=HTMLResponse)
    def legacy_page():
        return Path(__file__).with_name('stage12.html').read_text(encoding='utf-8')

    assets=ROOT/'frontend/dist/assets'
    if assets.is_dir():
        app.mount('/assets',StaticFiles(directory=assets),name='frontend-assets')

    @app.get('/research-evaluation-report',response_class=HTMLResponse)
    def research_report():
        report=ROOT/'docs/evaluation/STAGE13_VALIDATION_ROUND2_RESULTS.md'
        if not report.is_file(): raise HTTPException(404,'Evaluation report unavailable')
        return Response(report.read_text(encoding='utf-8'),media_type='text/plain; charset=utf-8')

    @app.get('/api/session')
    def get_session(request:Request):
        try:
            raw=request.cookies.get('jarvistrace_session')
            user=store.session_user(session_token_hash(raw)) if raw else None
            return dict(user=user,setup_required=store.user_count()==0)
        except Exception:
            raise HTTPException(503,'Account service unavailable. Check the local database configuration.') from None

    @app.post('/api/setup/admin',status_code=201)
    def setup_admin(body:SetupAdminRequest,request:Request,response:Response):
        try:
            email=normalize_email(body.email)
            user=store.setup_admin(body.display_name,email,body.password)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(409,str(exc)) from None
        except Exception:
            raise HTTPException(503,'Administrator setup could not complete. Check the local database configuration.') from None
        user['cameras']=store.allowed_cameras(user)
        issue_session(user,request,response)
        return dict(user=user,setup_required=False)

    @app.post('/api/login')
    def login(body:LoginRequest,request:Request,response:Response):
        try:
            user=store.authenticate(body.email,body.password)
        except ValueError as exc:
            raise HTTPException(401,'Email or password is incorrect') from exc
        except Exception:
            raise HTTPException(503,'Sign in is unavailable. Check the local database configuration.') from None
        if not user:
            raise HTTPException(401,'Email or password is incorrect')
        user['cameras']=store.allowed_cameras(user)
        issue_session(user,request,response)
        return dict(user=user)

    @app.post('/api/logout',status_code=204)
    def logout(request:Request,response:Response):
        raw=request.cookies.get('jarvistrace_session')
        if raw:
            try: store.delete_session(session_token_hash(raw))
            except Exception: pass
        response.delete_cookie('jarvistrace_session',path='/',httponly=True,samesite='strict',secure=request.url.scheme=='https')
        response.status_code=204
        return response

    @app.get('/api/recordings')
    def recordings(request:Request):
        user=current_user(request); refresh_catalog()
        try: metadata=store.list_recording_metadata(user)
        except Exception: raise HTTPException(503,'Recording list unavailable. Check the local database.') from None
        by_video={row['video_id']:row for row in metadata}
        ready={row['video_id']:row for row in catalog.recordings()}
        output=[]
        for video_id,row in by_video.items():
            item=ready.get(video_id)
            duration=float(row['duration_seconds'])
            status=row.get('status') or ('ready' if item else 'queued')
            capture=row.get('capture_start_utc')
            output.append(dict(media_id=item['media_id'] if item else None,video_id=video_id,
                camera_id=row['camera_id'],filename=row.get('filename') or video_id,duration_seconds=duration,
                playback_url=item['playback_url'] if item else None,playback_status='available' if item else 'unavailable',
                source_type=row.get('source_type') or 'indexed',status=status,
                capture_start=capture.isoformat() if capture else None,capture_timezone=row.get('capture_timezone'),
                coverage=(dict(start_seconds=float(row.get('coverage_start_seconds') or 0),
                    end_seconds=float(duration if row.get('coverage_end_seconds') is None
                                      else row['coverage_end_seconds'])) if status=='ready' else None)))
        return dict(recordings=output,unavailable_sources=catalog.unavailable)

    @app.post('/api/search')
    def run_search(body:SearchRequest,request:Request):
        user=current_user(request)
        if body.camera_id and user['role']!='administrator' and body.camera_id not in user.get('cameras',[]):
            raise HTTPException(403,'This camera is outside your access')
        if not lock.acquire(blocking=False):
            raise HTTPException(503,'A search is already running. Try again shortly.')
        try:
            refresh_catalog()
            args=body.model_dump(exclude={'query','mode'})
            if user['role']!='administrator': args['camera_ids']=user.get('cameras',[])
            args.update(visual_weight=0 if body.mode=='text' else 1,
                        text_weight=0 if body.mode=='visual' else 1,
                        min_visual_similarity=configured_similarity_threshold(
                            'JARVISTRACE_MIN_VISUAL_SIMILARITY'),
                        min_text_similarity=configured_similarity_threshold(
                            'JARVISTRACE_MIN_TEXT_SIMILARITY'))
            report=search_fn(body.query,**args)
            results=[]
            for row in report['comparison'][body.mode]:
                if user['role']!='administrator' and row['camera_id'] not in user.get('cameras',[]):
                    continue
                for channel in row.get('channels',{}).values():
                    for evidence in channel.get('evidence',[]):
                        evidence_catalog.register(evidence,row['camera_id'])
                media_id=catalog.media_id_for(row)
                media_item=catalog.items.get(media_id) if media_id else None
                if media_item: media_scope(user,media_item)
                item=public_evidence(row)
                item['playback_url']='/media/'+media_id if media_id else None
                item['playback_status']='available' if media_id else 'source unavailable; restart after checking indexed storage'
                results.append(item)
            raw_window_counts=report.get('raw_window_counts') or {
                mode:len(rows) for mode,rows in report['comparison'].items()}
            accepted_window_counts=report.get('window_counts') or {
                mode:len(rows) for mode,rows in report['comparison'].items()}
            result_status=('matches' if results else
                'no_strong_match' if raw_window_counts.get(body.mode,0)>0 else 'no_candidates')
            return dict(query=body.query,mode=body.mode,results=results,result_status=result_status,
                        raw_window_counts=raw_window_counts,accepted_window_counts=accepted_window_counts,
                        candidate_counts=report['candidate_counts'],limitations=report['limitations'],
                        text_diagnostics=public_evidence(report['text_diagnostics']))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503,'Search unavailable. Check the local database, indexed recordings and cached models.') from None
        finally:
            lock.release()

    @app.get('/api/cameras')
    def cameras(request:Request):
        user=current_user(request)
        try: return dict(cameras=store.list_cameras(user))
        except Exception: raise HTTPException(503,'Camera list unavailable. Check the local database.') from None

    @app.post('/api/cameras',status_code=201)
    def create_camera(body:CameraCreateRequest,request:Request):
        admin_user(request)
        try: camera=store.create_camera(body.camera_id,body.name,body.source_type,body.mtx_path)
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        except Exception: raise HTTPException(409,'Camera ID already exists or could not be saved') from None
        return dict(camera=camera)

    @app.get('/api/users')
    def users(request:Request):
        admin_user(request)
        try: return dict(users=store.list_users())
        except Exception: raise HTTPException(503,'User list unavailable. Check the local database.') from None

    @app.post('/api/users',status_code=201)
    def create_user(body:UserCreateRequest,request:Request):
        admin_user(request)
        try: user=store.create_user(body.display_name,body.email,body.password,body.role,body.cameras)
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        except Exception: raise HTTPException(409,'Email already exists or the user could not be saved') from None
        return dict(user=user)

    @app.get('/api/jobs')
    def jobs(request:Request):
        admin_user(request)
        try: return dict(jobs=store.list_jobs())
        except Exception: raise HTTPException(503,'Processing queue unavailable. Check the local database.') from None

    @app.post('/api/jobs/{job_id}/retry')
    def retry_job(job_id:str,request:Request):
        admin_user(request)
        try:
            if not store.retry_job(job_id): raise HTTPException(409,'Only failed jobs can be retried')
            return dict(status='queued')
        except HTTPException: raise
        except Exception: raise HTTPException(503,'Job could not be retried. Check the local database.') from None

    @app.post('/api/recordings/upload',status_code=202)
    async def upload_recording(request:Request):
        await run_in_threadpool(admin_user,request)
        form=None
        try:
            form=await request.form(max_files=1,max_fields=3,max_part_size=1024*1024)
            allowed_fields={'file','camera_id','capture_start_local','capture_timezone'}
            if set(form.keys())-allowed_fields or any(len(form.getlist(key))!=1 for key in form.keys()):
                raise HTTPException(422,'Provide one video file and one value for each upload field')
            file=form.get('file')
            camera_id=form.get('camera_id')
            capture_start_local=form.get('capture_start_local')
            capture_timezone=form.get('capture_timezone')
            if not isinstance(file,StarletteUploadFile) or not isinstance(camera_id,str):
                raise HTTPException(422,'Choose an MP4 recording and a configured camera')
            if capture_start_local is not None and not isinstance(capture_start_local,str):
                raise HTTPException(422,'Capture start must be text')
            if capture_timezone is not None and not isinstance(capture_timezone,str):
                raise HTTPException(422,'Time zone must be text')
            camera_rows=await run_in_threadpool(store.list_cameras,{'role':'administrator','id':'','cameras':[]})
            camera_ids={camera['camera_id'] for camera in camera_rows}
            if camera_id not in camera_ids: raise HTTPException(422,'Select a configured camera')
            captured,zone=parse_browser_datetime(capture_start_local,capture_timezone)
            saved=await run_in_threadpool(lambda:store_upload(file,filename=file.filename or '',
                storage_root=managed_root,probe=video_probe))
            try:
                job_id=await run_in_threadpool(lambda:store.enqueue_recording(video_id=saved['video_id'],
                    camera_id=camera_id,path=saved['path'],filename=saved['filename'],fps=saved['fps'],
                    duration=saved['duration_seconds'],source_sha256=saved['source_sha256'],source_type='upload',
                    capture_start=captured,capture_timezone=zone))
            except Exception:
                saved['path'].unlink(missing_ok=True)
                raise
            return dict(job_id=job_id,video_id=saved['video_id'],status='queued')
        except UploadBodyTooLarge: raise
        except StarletteHTTPException: raise
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        except Exception: raise HTTPException(503,'Upload could not be added to the processing queue. Check the local database.') from None
        finally:
            if form is not None:
                await form.close()

    @app.post('/api/mediamtx/segment',status_code=202)
    def mediamtx_segment(body:MediaMTXSegmentRequest,request:Request):
        configured=app_setting('JARVISTRACE_MEDIAMTX_TOKEN','')
        supplied=request.headers.get('authorization','')
        if not configured or not supplied.startswith('Bearer ') or not hmac.compare_digest(supplied[7:],configured):
            raise HTTPException(401,'MediaMTX notification token is invalid')
        try:
            camera=store.camera_for_mtx_path(body.path)
        except Exception:
            raise HTTPException(503,'Camera configuration unavailable') from None
        if not camera: raise HTTPException(404,'MediaMTX path is not registered to a camera')
        source_root=configured_path(app_setting('JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT','mediamtx/recordings'))
        source=Path(body.segment_path).resolve()
        if (source_root==source or not source.is_relative_to(source_root) or source.suffix.lower()!='.mp4'
                or not source.is_file()):
            raise HTTPException(422,'Completed segment is outside the configured MediaMTX recording folder')
        relative=source.relative_to(source_root)
        expected_parts=tuple(body.path.split('/'))
        if (not all(part and part not in {'.','..'} for part in expected_parts)
                or relative.parent.parts!=expected_parts):
            raise HTTPException(422,'Completed segment does not belong to the registered MediaMTX path')
        try:
            size=source.stat().st_size
            if size<=0 or size>MAX_UPLOAD_BYTES: raise HTTPException(413,'Completed segment exceeds the 1 GB processing limit')
            source_hash=sha256_file(source)
            relative_name=relative.as_posix()
            event_id,video_id=segment_ids(camera['camera_id'],relative_name,source_hash)
            existing=store.segment_event(event_id)
            if existing:
                return dict(job_id=existing['job_id'],video_id=existing['video_id'],status=existing['status'],duplicate=True)
            captured=segment_start_from_record_path(source.name)
            with source.open('rb') as stream:
                saved=store_upload(stream,filename=source.name,storage_root=managed_root,video_id=video_id,
                                   probe=video_probe,expected_sha256=source_hash)
            result=store.enqueue_recording(video_id=saved['video_id'],camera_id=camera['camera_id'],path=saved['path'],
                filename=saved['filename'],fps=saved['fps'],duration=saved['duration_seconds'],
                source_sha256=saved['source_sha256'],source_type='mediamtx',capture_start=captured,
                capture_timezone='UTC' if captured else None,segment_event=(event_id,))
            return dict(job_id=result['job_id'],video_id=result['video_id'],status=result['status'],duplicate=result['duplicate'])
        except HTTPException: raise
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        except Exception: raise HTTPException(503,'Completed segment could not be queued. Check the database and recording permissions.') from None

    @app.get('/api/research-evaluation')
    def research_evaluation(request:Request):
        current_user(request)
        return dict(title='G507 validation round 2 · preliminary',clips=6,cameras=1,positives=2,negatives=3,unresolved=1,
            full_frames=dict(hit_at_1='1 of 2',precision_at_5='20%'),
            frames_and_crops=dict(hit_at_1='2 of 2',precision_at_5='70%'),
            absent_item_queries_returning_candidates='3 of 3',
            warning='Small single-camera sample; labels were human-confirmed after AI assistance. Candidate returns do not prove presence or absence.')

    @app.api_route('/media/{media_id}',methods=['GET','HEAD'])
    def media(media_id:str,request:Request):
        user=current_user(request); refresh_catalog()
        item=catalog.items.get(media_id)
        if item is None:
            raise HTTPException(404,'Recording not available')
        media_scope(user,item)
        try:
            snapshot=verified_snapshot(item,catalog.roots)
        except (OSError,ValueError):
            raise HTTPException(409,'Recording changed or is missing. Restore the indexed source or index it under a new recording ID.') from None
        try:
            snapshot.seek(0,2);size=snapshot.tell()
            etag='"'+item['source_sha256']+'"'
            requested_range=request.headers.get('range')
            if request.headers.get('if-range') not in (None,etag): requested_range=None
            start,end,code=byte_range(requested_range,size)
            headers={'Accept-Ranges':'bytes','Content-Length':str(end-start+1),'ETag':etag}
            if code==206: headers['Content-Range']=f'bytes {start}-{end}/{size}'
            media_type=mimetypes.guess_type(item['path'].name)[0] or 'application/octet-stream'
            if request.method=='HEAD':
                snapshot.close(); return Response(status_code=code,headers=headers,media_type=media_type)
            snapshot.seek(start)
            def chunks():
                remaining=end-start+1
                try:
                    while remaining>0:
                        block=snapshot.read(min(65536,remaining))
                        if not block: break
                        remaining-=len(block); yield block
                finally: snapshot.close()
            return StreamingResponse(chunks(),status_code=code,headers=headers,media_type=media_type,
                                     background=BackgroundTask(snapshot.close))
        except BaseException:
            snapshot.close(); raise

    @app.get('/evidence/{asset_id}')
    def evidence(asset_id:str,request:Request):
        user=current_user(request)
        item=evidence_catalog.get(asset_id)
        if item is None: raise HTTPException(404,'Evidence image unavailable; run the search again')
        if user['role']!='administrator' and item['camera_id'] not in user.get('cameras',[]):
            raise HTTPException(403,'This evidence is outside your camera access')
        try: snapshot=verified_snapshot(dict(path=item['path'],source_sha256=item['sha256']),evidence_catalog.roots)
        except (OSError,ValueError): raise HTTPException(409,'Evidence image changed or is missing') from None
        return StreamingResponse(snapshot,media_type='image/png',headers={'Cache-Control':'no-store'},background=BackgroundTask(snapshot.close))

    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--storage-root',type=Path,action='append',help='Allowed recording folder; repeat for multiple roots')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('Port must be 1024..65535')
    import uvicorn
    app=create_app(roots=args.storage_root)
    print(f'JarvisTrace: http://127.0.0.1:{args.port}')
    uvicorn.run(app,host='127.0.0.1',port=args.port,proxy_headers=False)


if __name__=='__main__':
    main()
