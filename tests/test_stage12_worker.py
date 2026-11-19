import hashlib
from pathlib import Path
import tempfile
import unittest
from src.stage12_worker import process_next


class FakeStore:
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory()
        self.source_path=Path(self.temp.name)/'clip.mp4'
        payload=b'verified source footage'
        self.source_path.write_bytes(payload)
        self.source_sha256=hashlib.sha256(payload).hexdigest()
        self.job={'job_id':'job-1','video_id':'video-1','local_video_path':str(self.source_path),
                  'source_sha256':self.source_sha256}
        self.finished=[]
        self.heartbeats=[]
        self.finish_result=True

    def claim_job(self,worker_id):
        job=self.job
        self.job=None
        return job

    def heartbeat_job(self,job_id,worker_id):
        self.heartbeats.append((job_id,worker_id))
        return True

    def finish_job(self,*args,**kwargs):
        self.finished.append((args,kwargs))
        return self.finish_result


class WorkerTests(unittest.TestCase):
    def test_empty_queue_returns_without_touching_indexer(self):
        store=FakeStore(); store.job=None
        self.assertIsNone(process_next(store,indexer=lambda *_:self.fail('should not run')))

    def test_complete_job_marks_searchable_coverage(self):
        store=FakeStore()
        result=process_next(store,indexer=lambda video_id,**kw:{'status':'complete','sampling_complete':True,
            'video':{'frame_count':1200,'fps':30},'last_sample_seconds':39,'actual_interval_seconds':1},worker_id='worker')
        self.assertEqual(result['status'],'ready')
        self.assertEqual(result['coverage_end_seconds'],40)
        self.assertEqual(store.finished[0][0],('job-1','ready'))
        self.assertEqual(store.finished[0][1]['worker_id'],'worker')

    def test_queued_source_hash_is_passed_to_the_indexer(self):
        store=FakeStore(); observed=[]
        def indexer(_video_id,**kwargs):
            observed.append(kwargs['expected_source_sha256'])
            return {'status':'complete','sampling_complete':True,'video':{'frame_count':1200,'fps':30}}
        result=process_next(store,indexer=indexer,worker_id='worker')
        self.assertEqual(result['status'],'ready')
        self.assertEqual(observed,[store.source_sha256])

    def test_changed_source_bytes_fail_before_a_job_can_become_ready(self):
        store=FakeStore(); store.source_path.write_bytes(b'replaced footage')
        def indexer(_video_id,**kwargs):
            actual=hashlib.sha256(store.source_path.read_bytes()).hexdigest()
            if actual!=kwargs['expected_source_sha256']:
                raise ValueError('source hash mismatch')
            return {'status':'complete','sampling_complete':True,'video':{'frame_count':1200,'fps':30}}
        result=process_next(store,indexer=indexer,worker_id='worker')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(store.finished[0][0][1],'failed')

    def test_lost_lease_does_not_report_stale_worker_completion(self):
        store=FakeStore(); store.finish_result=False
        result=process_next(store,indexer=lambda *_args,**kw:{'status':'complete','sampling_complete':True,
            'video':{'frame_count':1200,'fps':30},'last_sample_seconds':39,'actual_interval_seconds':1},worker_id='old-worker')
        self.assertEqual(result['status'],'lease_lost')
        self.assertEqual(store.finished[0][1]['worker_id'],'old-worker')

    def test_partial_sampling_keeps_the_real_coverage_watermark(self):
        store=FakeStore()
        result=process_next(store,indexer=lambda *_args,**kw:{'status':'complete','sampling_complete':False,
            'last_sample_seconds':299,'actual_interval_seconds':1},worker_id='worker')
        self.assertEqual(result['coverage_end_seconds'],300)

    def test_index_failure_is_a_sanitized_failed_job(self):
        store=FakeStore()
        result=process_next(store,indexer=lambda *_args,**kw:(_ for _ in ()).throw(RuntimeError('secret path')),
                            worker_id='worker')
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('secret path',result['failure_reason'])
        self.assertEqual(store.finished[0][0][1],'failed')


if __name__=='__main__':
    unittest.main()
