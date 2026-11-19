import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.stage10 import index_recording


class TinyEncoder:
    model_id = 'test:512'
    embedding_dim = 512
    preprocess_description = 'test RGB encoder'
    device_str = 'cpu'

    def encode_images(self, images):
        vectors = np.zeros((len(images),512),dtype=np.float32)
        vectors[:,0] = 1
        return vectors


class TemporaryIndex:
    def __init__(self, path):
        self.path,self.rows = path,[]

    def recording(self,video_id):
        return dict(video_id=video_id,local_video_path=str(self.path),camera_id='CAM',fps=10,duration_seconds=1)

    def store_samples(self,rows,**settings):
        self.rows.extend(rows)
        return len(rows)


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'clip.avi'
        writer = cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.full((60,80,3),80,np.uint8))
        writer.release()
        self.index = TemporaryIndex(self.video)

    def run_index(self,**kwargs):
        return index_recording('clip',output_root=self.root/'output',encoder=TinyEncoder(),index=self.index,**kwargs)

    def test_samples_raw_frames_and_hashes_without_detector(self):
        report = self.run_index(every_seconds=.2)
        self.assertEqual([r['frame_index'] for r in self.index.rows],[0,2,4,6,8])
        self.assertEqual([r['timestamp_seconds'] for r in self.index.rows],[0,.2,.4,.6,.8])
        self.assertTrue(report['sampling_complete'])
        self.assertEqual(report['inserted_embeddings'],5)
        for row in self.index.rows:
            self.assertEqual(hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest(),row['image_sha256'])
            self.assertEqual(row['variant'],'frame')
        self.assertTrue(Path(report['report_path']).is_file())

    def test_sample_limit_is_explicit_and_does_not_claim_full_coverage(self):
        report = self.run_index(every_seconds=.1,max_frames=3)
        self.assertFalse(report['sampling_complete'])
        self.assertEqual(report['sampled_frames'],3)
        self.assertEqual(report['last_sample_seconds'],.2)

    def test_reject_invalid_sampling_and_wrong_tracking_source(self):
        for value in [0,-1,float('nan')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.run_index(every_seconds=value)
        tracking = self.root/'analysis.json'
        tracking.write_text(json.dumps(dict(report_format='group8-stage9-v1',video_id='clip',
                            provenance={'video_sha256':'bad'},video={})))
        with self.assertRaises(ValueError):
            self.run_index(tracking_report=tracking)
        self.assertEqual(self.index.rows,[])

    def test_rejects_recording_bytes_that_differ_from_the_queued_source_hash(self):
        expected=hashlib.sha256(b'other footage').hexdigest()
        with self.assertRaisesRegex(ValueError,'queued source hash'):
            self.run_index(expected_source_sha256=expected)
        self.assertEqual(self.index.rows,[])

    def test_crops_come_from_original_pixels_with_detector_provenance(self):
        tracking = self.root/'analysis.json'
        source = hashlib.sha256(self.video.read_bytes()).hexdigest()
        tracking.write_text(json.dumps(dict(report_format='group8-stage9-v1',video_id='clip',camera_id='CAM',
            target_class_id=24,video=dict(fps=10,width=80,height=60,frame_count=10),
            provenance=dict(video_sha256=source,scope_id='scope'),analysis=dict(observation_count=1))))
        tracking.with_name('observations.jsonl').write_text(json.dumps(dict(frame_index=0,
            timestamp_seconds=0,scope_id='scope',class_id=24,local_track_id=17,
            confidence=.6,bbox_xyxy=[10,10,30,40]))+'\n')
        report = self.run_index(tracking_report=tracking,every_seconds=.5)
        self.assertEqual(report['crop_samples'],1)
        crop = [r for r in self.index.rows if r['variant']=='crop'][0]
        self.assertEqual(crop['bbox_xyxy'],[10,10,30,40])
        self.assertEqual(crop['crop_provenance']['review_status'],'unreviewed_detector_output')
        image = cv2.imread(crop['image_path'])
        self.assertEqual(image.shape[:2],(30,20))
