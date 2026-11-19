import hashlib
from pathlib import Path
import tempfile
import unittest
import os
import base64
import re
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.stage12 import PlaybackCatalog, create_app
from src.stage12_workspace import session_token_hash


class MemoryStore:
    def __init__(self):
        self.users={}; self.sessions={}; self.cameras={}; self.jobs=[]; self.uploads=[]; self.segment_events={}

    def user_count(self): return len(self.users)
    def setup_admin(self,name,email,password):
        if self.users: raise RuntimeError('Administrator setup has already been completed')
        row={'id':'admin-1','display_name':name,'email':email,'role':'administrator','cameras':[],'password':password}
        self.users[row['id']]=row
        return {k:v for k,v in row.items() if k!='password'}
    def authenticate(self,email,password):
        return next(({k:v for k,v in row.items() if k not in {'password'}} for row in self.users.values()
                     if row['email']==email.lower() and row['password']==password),None)
    def create_session(self,user_id,token_hash,expires): self.sessions[token_hash]=user_id
    def session_user(self,token_hash):
        user=self.users.get(self.sessions.get(token_hash))
        if user is None: return None
        result={k:v for k,v in user.items() if k!='password'}
        result['cameras']=self.allowed_cameras(result)
        return result
    def delete_session(self,token_hash): self.sessions.pop(token_hash,None)
    def allowed_cameras(self,user):
        if user['role']=='administrator': return list(self.cameras)
        return user.get('cameras',[])
    def create_camera(self,camera_id,name,source_type,mtx_path):
        if camera_id in self.cameras: raise ValueError('duplicate')
        row={'camera_id':camera_id,'name':name,'source_type':source_type,'mtx_path':mtx_path or (camera_id if source_type=='mediamtx' else None),
             'connection_status':'configured','searchable_until':None}
        self.cameras[camera_id]=row; return row
    def list_cameras(self,user):
        allowed=set(self.allowed_cameras(user))
        return [row for key,row in self.cameras.items() if user['role']=='administrator' or key in allowed]
    def create_user(self,name,email,password,role,cameras):
        row={'id':f'user-{len(self.users)}','display_name':name,'email':email.lower(),'role':role,
             'cameras':list(cameras),'password':password}
        self.users[row['id']]=row
        return {k:v for k,v in row.items() if k!='password'}
    def list_users(self): return [{k:v for k,v in row.items() if k!='password'} for row in self.users.values()]
    def list_recording_metadata(self,user): return []
    def list_jobs(self): return self.jobs
    def enqueue_recording(self,**recording):
        if recording.get('segment_event'):
            event_id=recording['segment_event'][0]
            if event_id in self.segment_events:
                return {**self.segment_events[event_id],'duplicate':True}
            result={'job_id':f"job-{len(self.uploads)+1}",'video_id':recording['video_id'],'status':'queued'}
            self.segment_events[event_id]=result
            self.uploads.append(recording)
            return {**result,'duplicate':False}
        self.uploads.append(recording)
        return f"job-{len(self.uploads)}"
    def camera_for_mtx_path(self,path):
        return next(({'camera_id':camera['camera_id']} for camera in self.cameras.values() if camera['mtx_path']==path),None)
    def segment_event(self,event_id): return self.segment_events.get(event_id)


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        rows=[]
        for video_id,camera_id in [('a','CAM-A'),('b','CAM-B')]:
            path=root/f'{video_id}.mp4'; payload=video_id.encode()*100; path.write_bytes(payload)
            rows.append(dict(video_id=video_id,camera_id=camera_id,source_sha256=hashlib.sha256(payload).hexdigest(),
                source_video_path=str(path),duration_seconds=10))
        self.catalog=PlaybackCatalog(rows,[root]); self.store=MemoryStore()
        self.search_calls=[]
        def search(query,**kwargs):
            self.search_calls.append(kwargs)
            candidates=[dict(row,start_seconds=1,end_seconds=3,rrf_score=.02,channels={}) for row in rows]
            return dict(comparison={mode:candidates for mode in ('visual','text','hybrid')},
                candidate_counts={'visual':2,'text':0},limitations='ranked candidates',text_diagnostics={})
        self.client=TestClient(create_app(catalog=self.catalog,roots=[root],store=self.store,search_fn=search,
            storage_root=root/'managed',video_probe=lambda _path:{'fps':30.0,'duration_seconds':10.0,
                'width':640,'height':360,'format_name':'mov,mp4'}),
            base_url='http://127.0.0.1',client=('127.0.0.1',3210))
        self.addCleanup(self.client.close)

    def test_first_run_admin_setup_session_and_logout(self):
        self.assertEqual(self.client.get('/api/session').json()['setup_required'],True)
        self.assertEqual(self.client.get('/api/jobs').status_code,401)

    def test_fastapi_serves_the_built_frontend_bundle(self):
        page=self.client.get('/')
        self.assertEqual(page.status_code,200)
        script=re.search(r'src="(/assets/[^\"]+\.js)"',page.text)
        self.assertIsNotNone(script,page.text)
        asset=self.client.get(script.group(1))
        self.assertEqual(asset.status_code,200)
        self.assertIn('Search footage',asset.text)
        response=self.client.post('/api/setup/admin',json={'display_name':'Goodluck Kassa','email':'GOODLUCK@example.test',
            'password':'very secure test password'})
        self.assertEqual(response.status_code,201,response.text)
        self.assertEqual(response.json()['user']['role'],'administrator')
        cookie=response.headers['set-cookie'].lower()
        self.assertIn('httponly',cookie); self.assertIn('samesite=strict',cookie)
        self.assertEqual(self.client.get('/api/session').json()['setup_required'],False)
        self.assertEqual(self.client.post('/api/setup/admin',json={'display_name':'Second','email':'second@example.test',
            'password':'very secure test password'}).status_code,409)
        self.assertEqual(self.client.post('/api/logout').status_code,204)
        self.assertEqual(self.client.get('/api/jobs').status_code,401)

    def test_investigator_scope_applies_to_search_playback_and_admin_routes(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        for camera in ('CAM-A','CAM-B'):
            response=self.client.post('/api/cameras',json={'camera_id':camera,'name':camera,'source_type':'mediamtx'})
            self.assertEqual(response.status_code,201,response.text)
        response=self.client.post('/api/users',json={'display_name':'Investigator','email':'user@example.test',
            'password':'very secure test password','role':'investigator','cameras':['CAM-A']})
        self.assertEqual(response.status_code,201,response.text)
        self.client.post('/api/logout')
        response=self.client.post('/api/login',json={'email':'user@example.test','password':'very secure test password'})
        self.assertEqual(response.status_code,200,response.text)
        search=self.client.post('/api/search',json={'query':'a bag','camera_id':'CAM-A'})
        self.assertEqual(search.status_code,200,search.text)
        self.assertEqual([row['camera_id'] for row in search.json()['results']],['CAM-A'])
        self.assertEqual(self.search_calls[-1]['camera_ids'],['CAM-A'])
        self.assertEqual(self.client.post('/api/search',json={'query':'a bag','camera_id':'CAM-B'}).status_code,403)
        media_b=next(v for v in self.catalog.recordings() if v['camera_id']=='CAM-B')
        self.assertEqual(self.client.get(media_b['playback_url']).status_code,403)
        self.assertEqual(self.client.get('/api/jobs').status_code,403)

    def test_verified_upload_is_saved_under_managed_root_and_queued(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        self.client.post('/api/cameras',json={'camera_id':'CAM-UP','name':'Upload test','source_type':'upload'})
        response=self.client.post('/api/recordings/upload',data={'camera_id':'CAM-UP',
            'capture_start_local':'2026-11-19T12:30','capture_timezone':'Africa/Lagos'},
            files={'file':('source.mp4',b'validated test bytes','video/mp4')})
        self.assertEqual(response.status_code,202,response.text)
        self.assertEqual(response.json()['status'],'queued')
        saved=self.store.uploads[0]
        self.assertEqual(saved['source_type'],'upload')
        self.assertTrue(Path(saved['path']).is_relative_to(Path(self.temp.name)/'managed'))
        self.assertEqual(saved['capture_start'].isoformat(),'2026-11-19T11:30:00+00:00')
        self.assertEqual(saved['capture_timezone'],'Africa/Lagos')

    def test_queued_recording_does_not_claim_searchable_coverage(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        self.store.list_recording_metadata=lambda _user:[dict(video_id='queued-video',camera_id='CAM-A',
            duration_seconds=90,coverage_start_seconds=0,coverage_end_seconds=0,status='queued',
            capture_start_utc=None,capture_timezone=None,filename='queued.mp4',source_type='upload')]

        response=self.client.get('/api/recordings')

        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['recordings'][0]['status'],'queued')
        self.assertIsNone(response.json()['recordings'][0]['coverage'])

    def test_upload_requires_an_administrator_and_a_registered_camera(self):
        response=self.client.post('/api/recordings/upload',data={'camera_id':'NOPE'},
            files={'file':('source.mp4',b'bytes','video/mp4')})
        self.assertEqual(response.status_code,401)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        response=self.client.post('/api/recordings/upload',data={'camera_id':'NOPE'},
            files={'file':('source.mp4',b'bytes','video/mp4')})
        self.assertEqual(response.status_code,422)

    def test_upload_authentication_runs_before_multipart_parsing(self):
        response=self.client.post('/api/recordings/upload',content=b'not a multipart body',
            headers={'Content-Type':'multipart/form-data; boundary=missing-boundary'})
        self.assertEqual(response.status_code,401,response.text)

    def test_authenticated_malformed_multipart_gets_a_client_error(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        response=self.client.post('/api/recordings/upload',content=b'not a multipart body',
            headers={'Content-Type':'multipart/form-data; boundary=missing-boundary'})
        self.assertEqual(response.status_code,400,response.text)

    def test_upload_database_and_file_work_runs_off_the_async_event_loop(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        self.client.post('/api/cameras',json={'camera_id':'CAM-UP','name':'Upload test','source_type':'upload'})
        thread_names=[]
        for method_name in ('session_user','list_cameras','enqueue_recording'):
            original=getattr(self.store,method_name)
            def record_thread(*args,_original=original,**kwargs):
                thread_names.append(threading.current_thread().name)
                return _original(*args,**kwargs)
            setattr(self.store,method_name,record_thread)
        from src import stage12
        original_upload=stage12.store_upload
        def record_upload_thread(*args,**kwargs):
            thread_names.append(threading.current_thread().name)
            return original_upload(*args,**kwargs)
        with patch('src.stage12.store_upload',side_effect=record_upload_thread):
            response=self.client.post('/api/recordings/upload',data={'camera_id':'CAM-UP'},
                files={'file':('clip.mp4',b'fake validated upload','video/mp4')})
        self.assertEqual(response.status_code,202,response.text)
        self.assertTrue(thread_names)
        self.assertTrue(all('AnyIO worker thread' in name for name in thread_names),thread_names)

    def test_upload_without_content_length_is_stopped_at_the_body_limit(self):
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        self.client.post('/api/cameras',json={'camera_id':'CAM-UP','name':'Upload test','source_type':'upload'})
        prefix=(b'--x\r\nContent-Disposition: form-data; name="file"; filename="large.mp4"\r\n'
                b'Content-Type: video/mp4\r\n\r\n')
        chunks=iter((prefix,b'x'*(2*1024*1024+128)))
        with patch('src.stage12.MAX_UPLOAD_BYTES',64):
            response=self.client.post('/api/recordings/upload',content=chunks,
                headers={'Content-Type':'multipart/form-data; boundary=x'})
        self.assertEqual(response.status_code,413,response.text)

    def test_mediamtx_segment_hook_is_secret_gated_path_bounded_and_idempotent(self):
        root=Path(self.temp.name)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        source_root=root/'mediamtx-recordings'; segment=source_root/'CAM-MTX'/'1795000000-123456.mp4'
        segment.parent.mkdir(parents=True)
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        segment.write_bytes(base64.b64decode(encoded))
        self.client.post('/api/cameras',json={'camera_id':'CAM-MTX','name':'MediaMTX camera','source_type':'mediamtx'})
        payload={'path':'CAM-MTX','segment_path':str(segment)}
        with patch.dict(os.environ,{'JARVISTRACE_MEDIAMTX_TOKEN':'test-hook-secret',
                                    'JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT':str(source_root)}):
            self.assertEqual(self.client.post('/api/mediamtx/segment',json=payload).status_code,401)
            headers={'Authorization':'Bearer test-hook-secret'}
            first=self.client.post('/api/mediamtx/segment',json=payload,headers=headers)
            self.assertEqual(first.status_code,202,first.text)
            second=self.client.post('/api/mediamtx/segment',json=payload,headers=headers)
        self.assertFalse(first.json()['duplicate'])
        self.assertTrue(second.json()['duplicate'])
        self.assertEqual(len(self.store.uploads),1)
        self.assertEqual(self.store.uploads[0]['source_type'],'mediamtx')
        self.assertEqual(self.store.uploads[0]['capture_timezone'],'UTC')

    def test_mediamtx_hook_rejects_a_segment_from_another_registered_path(self):
        root=Path(self.temp.name)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        source_root=root/'mediamtx-recordings'; segment=source_root/'CAM-OTHER'/'1795000000-123456.mp4'
        segment.parent.mkdir(parents=True)
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        segment.write_bytes(base64.b64decode(encoded))
        self.client.post('/api/cameras',json={'camera_id':'CAM-MTX','name':'MediaMTX camera','source_type':'mediamtx'})
        self.client.post('/api/cameras',json={'camera_id':'CAM-OTHER','name':'Other camera','source_type':'mediamtx'})
        payload={'path':'CAM-MTX','segment_path':str(segment)}
        with patch.dict(os.environ,{'JARVISTRACE_MEDIAMTX_TOKEN':'test-hook-secret',
                                    'JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT':str(source_root)}):
            response=self.client.post('/api/mediamtx/segment',json=payload,
                headers={'Authorization':'Bearer test-hook-secret'})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.store.uploads,[])

    def test_mediamtx_parent_path_cannot_claim_a_nested_camera_segment(self):
        root=Path(self.temp.name)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        source_root=root/'mediamtx-recordings'; segment=source_root/'CAM'/'SUB'/'1795000000-123456.mp4'
        segment.parent.mkdir(parents=True)
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        segment.write_bytes(base64.b64decode(encoded))
        self.client.post('/api/cameras',json={'camera_id':'CAM-PARENT','name':'Parent camera',
            'source_type':'mediamtx','mtx_path':'CAM'})
        self.client.post('/api/cameras',json={'camera_id':'CAM-SUB','name':'Nested camera',
            'source_type':'mediamtx','mtx_path':'CAM/SUB'})
        payload={'path':'CAM','segment_path':str(segment)}
        with patch.dict(os.environ,{'JARVISTRACE_MEDIAMTX_TOKEN':'test-hook-secret',
                                    'JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT':str(source_root)}):
            response=self.client.post('/api/mediamtx/segment',json=payload,
                headers={'Authorization':'Bearer test-hook-secret'})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.store.uploads,[])

    def test_mediamtx_hook_rejects_segment_outside_recording_root(self):
        root=Path(self.temp.name)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        source_root=root/'mediamtx-recordings'; outside=root/'outside.mp4'
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        outside.write_bytes(base64.b64decode(encoded))
        self.client.post('/api/cameras',json={'camera_id':'CAM-MTX','name':'MediaMTX camera','source_type':'mediamtx'})
        payload={'path':'CAM-MTX','segment_path':str(outside)}
        with patch.dict(os.environ,{'JARVISTRACE_MEDIAMTX_TOKEN':'test-hook-secret',
                                    'JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT':str(source_root)}):
            response=self.client.post('/api/mediamtx/segment',json=payload,
                headers={'Authorization':'Bearer test-hook-secret'})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.store.uploads,[])

    def test_mediamtx_hook_rejects_source_changes_between_hash_and_copy(self):
        root=Path(self.temp.name)
        self.client.post('/api/setup/admin',json={'display_name':'Admin','email':'admin@example.test','password':'very secure test password'})
        source_root=root/'mediamtx-recordings'; segment=source_root/'CAM-MTX'/'1795000000-123456.mp4'
        segment.parent.mkdir(parents=True)
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        segment.write_bytes(base64.b64decode(encoded))
        self.client.post('/api/cameras',json={'camera_id':'CAM-MTX','name':'MediaMTX camera','source_type':'mediamtx'})
        payload={'path':'CAM-MTX','segment_path':str(segment)}
        with (patch.dict(os.environ,{'JARVISTRACE_MEDIAMTX_TOKEN':'test-hook-secret',
                                     'JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT':str(source_root)}),
              patch('src.stage12.sha256_file',return_value='a'*64)):
            response=self.client.post('/api/mediamtx/segment',json=payload,
                headers={'Authorization':'Bearer test-hook-secret'})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.store.uploads,[])


if __name__=='__main__':
    unittest.main()
