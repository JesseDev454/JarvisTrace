import hashlib
from pathlib import Path
from unittest.mock import patch
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient
from src.stage12 import PlaybackCatalog, SearchRequest, create_app
from src.stage12_workspace import session_token_hash
from tests.test_hybrid_search import visual


class MemoryWorkspaceStore:
    def __init__(self, catalog):
        self.catalog=catalog
        self.user={'id':'test-admin','display_name':'Test Admin','email':'admin@example.test',
                   'role':'administrator','cameras':['CAM']}
        self.token_hash=session_token_hash('test-session')

    def session_user(self, token_hash):
        return self.user if token_hash==self.token_hash else None

    def user_count(self): return 1
    def allowed_cameras(self,user): return ['CAM']

    def list_recording_metadata(self,user):
        return [dict(video_id=row['video_id'],camera_id=row['camera_id'],duration_seconds=row['duration_seconds'],
                     filename=row['video_id']+'.mp4',source_type='upload',status='ready',
                     capture_start_utc=None,capture_timezone=None,coverage_start_seconds=0,
                     coverage_end_seconds=row['duration_seconds']) for row in self.catalog.items.values()]


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.video=self.root/'test.mp4'
        self.payload=b'0123456789'*100
        self.video.write_bytes(self.payload)
        self.sha=hashlib.sha256(self.payload).hexdigest()
        self.row=dict(video_id='a',camera_id='CAM',source_sha256=self.sha,
                      source_video_path=str(self.video),duration_seconds=10)
        self.catalog=PlaybackCatalog([self.row],[self.root])
        self.calls=[]
        def search(query,**kwargs):
            self.calls.append((query,kwargs))
            hit=dict(self.row,start_seconds=4,end_seconds=8,rrf_score=.02,
                     channels={'visual':{'rank':1,'rrf_contribution':.02,'evidence':[
                         dict(visual(5),image_path='secret.png',source_video_path=str(self.video))]}})
            return dict(comparison={m:[hit] for m in ('visual','text','hybrid')},
                        limitations='not confidence',candidate_counts={'visual':1,'text':0},
                        text_diagnostics={'excluded_sources':[]})
        self.store=MemoryWorkspaceStore(self.catalog)
        self.client=self.make_client(search)
        self.client.cookies.set('jarvistrace_session','test-session')
        self.addCleanup(self.client.close)

    def make_client(self,search_fn):
        client=TestClient(create_app(catalog=self.catalog,search_fn=search_fn,store=self.store),
                          base_url='http://127.0.0.1',client=('127.0.0.1',1234))
        client.cookies.set('jarvistrace_session','test-session')
        return client

    def test_search_filters_mode_and_private_paths(self):
        r=self.client.post('/api/search',json={'query':' bag ','mode':'visual',
              'camera_id':'CAM','start_seconds':4,'end_seconds':8})
        self.assertEqual(r.status_code,200)
        data=r.json(); hit=data['results'][0]
        self.assertTrue(hit['playback_url'].startswith('/media/'))
        self.assertNotIn(str(self.video),r.text)
        self.assertNotIn('secret.png',r.text)
        self.assertEqual(self.calls[0][0],'bag')
        self.assertEqual(self.calls[0][1]['text_weight'],0)
        self.assertEqual(self.calls[0][1]['start_seconds'],4)
        self.assertEqual(r.json()['result_status'],'matches')

    def test_search_reports_no_strong_match_separately_from_no_candidates(self):
        from unittest.mock import patch
        received={}
        def weak_search(query,**kwargs):
            received.update(kwargs)
            return dict(comparison={'hybrid':[]},raw_window_counts={'hybrid':3},
                        limitations='ranked',candidate_counts={'visual':20,'text':0},
                        text_diagnostics={})
        with patch.dict('os.environ',{
            'JARVISTRACE_MIN_VISUAL_SIMILARITY':'0.28',
            'JARVISTRACE_MIN_TEXT_SIMILARITY':'0.47',
        }), self.make_client(weak_search) as client:
            response=client.post('/api/search',json={'query':'a blue case'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['result_status'],'no_strong_match')
        self.assertEqual(response.json()['results'],[])
        self.assertEqual(response.json()['raw_window_counts']['hybrid'],3)
        self.assertEqual(received['min_visual_similarity'],.28)
        self.assertEqual(received['min_text_similarity'],.47)

    def test_ranges_head_invalid_ranges_and_unknown_ids(self):
        item=self.client.get('/api/recordings').json()['recordings'][0]
        url=item['playback_url']
        r=self.client.get(url,headers={'Range':'bytes=10-19'})
        self.assertEqual(r.status_code,206)
        self.assertEqual(r.content,self.payload[10:20])
        self.assertEqual(r.headers['content-range'],'bytes 10-19/1000')
        self.assertEqual(self.client.head(url).headers['content-length'],'1000')
        self.assertEqual(self.client.get(url,headers={'Range':'bytes=1001-'}).status_code,416)
        self.assertEqual(self.client.get('/media/not-known').status_code,404)
        self.assertEqual(self.client.get(url,headers={'Range':'bytes=-7'}).content,self.payload[-7:])
        self.assertEqual(self.client.get(url,headers={'Range':'bytes=990-'}).content,self.payload[990:])
        self.assertEqual(self.client.get(url,headers={'Range':'bytes=0-1','If-Range':'"old"'}).status_code,200)
        for value in ('bytes=-0','bytes=3-2','bytes=0-1,3-4','garbage'):
            self.assertEqual(self.client.get(url,headers={'Range':value}).status_code,416)

    def test_serves_verified_snapshot_if_source_replaced_after_verification(self):
        import src.stage12 as module
        original=module.verified_snapshot
        def replace_after(*args):
            snapshot=original(*args)
            self.video.write_bytes(b'replacement during response')
            return snapshot
        url=self.catalog.recordings()[0]['playback_url']
        with patch('src.stage12.verified_snapshot',side_effect=replace_after):
            self.assertEqual(self.client.get(url).content,self.payload)

    def test_source_replacement_and_outside_root_are_rejected(self):
        url=self.client.get('/api/recordings').json()['recordings'][0]['playback_url']
        self.video.write_bytes(b'replaced source')
        self.assertEqual(self.client.get(url).status_code,409)
        self.video.unlink()
        self.assertEqual(self.client.get(url).status_code,409)
        denied=PlaybackCatalog([self.row],[self.root/'other'])
        self.assertEqual(denied.recordings(),[])

    def test_validation_and_cross_origin_requests(self):
        for body in ({'query':' '},{'query':'bag','end_seconds':2,'start_seconds':4},
                     {'query':'bag','mode':'unknown'}):
            self.assertEqual(self.client.post('/api/search',json=body).status_code,422)
        with self.assertRaises(ValueError):
            SearchRequest(query='bag',start_seconds=float('inf'))
        self.assertEqual(self.client.post('/api/search',json={'query':'bag'},
                          headers={'Origin':'https://untrusted.example'}).status_code,403)
        self.assertEqual(self.client.get('/',headers={'Host':'untrusted.example'}).status_code,400)

    def test_page_and_empty_search_error_state(self):
        self.assertEqual(self.client.get('/').status_code,200)
        def broken(*args,**kwargs): raise RuntimeError('secret password')
        with self.make_client(broken) as c:
            r=c.post('/api/search',json={'query':'bag'})
            self.assertEqual(r.status_code,503)
            self.assertNotIn('secret password',r.text)

    def test_empty_results_and_busy_search(self):
        entered=threading.Event(); release=threading.Event()
        def slow(*args,**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError('test timeout')
            return dict(comparison={'hybrid':[]},candidate_counts={'visual':0,'text':0},
                        limitations='test',text_diagnostics={})
        with self.make_client(slow) as c:
            responses=[]
            worker=threading.Thread(target=lambda:responses.append(c.post('/api/search',json={'query':'bag'})))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(c.post('/api/search',json={'query':'bag'}).status_code,503)
            finally:
                release.set();worker.join(5)
            self.assertEqual(responses[0].json()['results'],[])
            self.assertEqual(responses[0].json()['result_status'],'no_candidates')
