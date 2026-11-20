import unittest
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from src import stage13_crops as target


class CropGeometryTests(unittest.TestCase):
    def test_padding_clamping_and_deterministic_dedup(self):
        detections=[dict(class_id=24,confidence=.8,bbox=[10,20,30,40]),
                    dict(class_id=24,confidence=.9,bbox=[10,20,30,40]),
                    dict(class_id=0,confidence=.99,bbox=[0,0,50,50]),
                    dict(class_id=26,confidence=.3,bbox=[95,95,100,100])]
        rows=target.select_crops(detections,100,100)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['bbox_xyxy'],[8,18,32,42])
        self.assertEqual(rows[0]['confidence'],.9)
        self.assertEqual(rows[1]['bbox_xyxy'],[94,94,100,100])

    def test_invalid_boxes_and_confidence_are_rejected(self):
        for box,confidence in [([3,2,1,4],.8),([0,0,float('nan'),4],.8),([0,0,4,4],2)]:
            with self.subTest(box=box),self.assertRaises(ValueError):
                target.select_crops([dict(class_id=24,confidence=confidence,bbox=box)],100,100)

    def test_cap_and_threshold_are_fixed(self):
        rows=[dict(class_id=24,confidence=.9-i*.01,bbox=[i*10,0,i*10+5,5]) for i in range(10)]
        self.assertEqual(len(target.select_crops(rows,200,100)),8)
        self.assertEqual(target.select_crops([dict(class_id=24,confidence=.24,bbox=[0,0,5,5])],20,20),[])

    def test_public_or_equal_target_schema_is_refused_before_io(self):
        for source,destination in [('public','stage13_new'),('stage13_old','public'),('stage13_old','stage13_old')]:
            with self.subTest(source=source),self.assertRaises(ValueError):
                target.check_schemas(source,destination)

    def test_configuration_binding_detects_modified_manifest(self):
        document={'settings':target.SETTINGS,'weights_sha256':'a'*64,'baseline_digest':'b'*64,'frames':[],'crops':[]}
        signature=target.manifest_signature(document)
        document['crops'].append({'image_sha256':'c'*64})
        self.assertNotEqual(signature,target.manifest_signature(document))

    def test_completed_report_refused_before_database_or_models(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=Path(root)/'detections.json'
            manifest.with_name('index.json').write_text('{}')
            with patch.object(target,'baseline_rows',side_effect=AssertionError('DB touched')):
                with self.assertRaises(FileExistsError):target.index_crops(manifest,'stage13_new')

    def test_frame_and_crop_pixels_cannot_be_rebound_by_rehashing_manifest(self):
        import numpy as np
        from src.stage10 import _save_image
        with tempfile.TemporaryDirectory() as root:
            pixels=np.arange(20*20*3,dtype=np.uint8).reshape(20,20,3)
            image,digest=_save_image(pixels,Path(root))
            row=dict(video_id='a',source_sha256='a'*64,frame_index=0,image_path=image,image_sha256=digest,
                encoder_key='b'*64,embedding_model='CLIP',preprocess_version='v1',source_camera_id='C',
                source_video_path='source.avi',source_fps=10.,source_duration_seconds=1.,timestamp_seconds=0.)
            raw=dict(class_id=24,class_name='backpack',confidence=.9,bbox=[5,5,15,15])
            proposal=target.select_crops([raw],20,20)[0]
            x1,y1,x2,y2=proposal['bbox_xyxy']
            crop_path,crop_hash=_save_image(pixels[y1:y2,x1:x2],Path(root))
            document=dict(format='group8-stage13-object-crops-v1',status='detected',settings=target.SETTINGS,
                weights_sha256='c'*64,baseline_digest='d'*64,
                frames=[dict(row,width=20,height=20,raw_detections=[raw])],
                crops=[dict(proposal,frame_number=0,image_path=crop_path,image_sha256=crop_hash)])
            document['manifest_signature']=target.manifest_signature(document)
            target.validate_document(document,[row])
            for mutate in ('timestamp','crop'):
                broken=copy.deepcopy(document)
                if mutate=='timestamp':broken['frames'][0]['timestamp_seconds']=.1
                else:broken['crops'][0].update(image_path=image,image_sha256=digest)
                broken['manifest_signature']=target.manifest_signature(broken)
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):target.validate_document(broken,[row])


@unittest.skipUnless(os.getenv('GROUP8_TEST_POSTGRES')=='1','PostgreSQL opt-in required')
class CropDatabaseTests(unittest.TestCase):
    def test_exact_clone_and_existing_crop_conflict_rejection(self):
        import uuid
        import cv2
        import numpy as np
        from psycopg import sql
        from src.pg_config import open_database
        from src.stage13_index import create_isolated_schema,register_recording
        from src.visual_index import VisualIndex
        from src.stage10 import _file_hash
        source='stage13_test_'+uuid.uuid4().hex;destination='stage13_test_'+uuid.uuid4().hex
        before=target.snapshot('public')
        try:
            create_isolated_schema(source)
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'tiny.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
                self.assertTrue(writer.isOpened())
                for _ in range(10):writer.write(np.zeros((32,32,3),np.uint8))
                writer.release();index=VisualIndex(schema=source)
                register_recording(index,path,'CAM','tiny',_file_hash(path))
                vector=np.arange(512,dtype=np.float32)+1
                sample=dict(video_id='tiny',source_sha256=_file_hash(path),frame_index=0,timestamp_seconds=0.,
                    variant='frame',bbox_xyxy=None,image_path='frame.png',image_sha256='a'*64,embedding=vector,
                    source_camera_id='CAM',source_video_path=str(path.resolve()),source_fps=10.,source_duration_seconds=1.)
                identity=dict(encoder_key='b'*64,embedding_model='fixture',preprocess_version='v1')
                index.store_samples([sample],**identity)
                document=dict(settings=target.SETTINGS,weights_sha256='c'*64,baseline_digest='d'*64,
                    baseline_snapshot=target.snapshot(source),frames=[],crops=[])
                target.clone_baseline(source,destination,document)
                target.clone_baseline(source,destination,document)
                with open_database(readonly=True) as c:
                    original=c.execute(sql.SQL('SELECT embedding::text AS v FROM {}.visual_embeddings').format(sql.Identifier(source))).fetchone()['v']
                    cloned=c.execute(sql.SQL('SELECT embedding::text AS v FROM {}.visual_embeddings').format(sql.Identifier(destination))).fetchone()['v']
                self.assertEqual(original,cloned)
                crop=dict(sample,variant='crop',bbox_xyxy=[0,0,8,8],crop_provenance={'kind':'unreviewed_detector_output'})
                VisualIndex(schema=destination).store_samples([crop],**identity)
                self.assertEqual(len(target.verify_existing_crops(destination,[crop],'b'*64)),1)
                with open_database() as c:
                    c.execute(sql.SQL("UPDATE {}.visual_embeddings SET image_sha256=%s WHERE variant='crop'").format(sql.Identifier(destination)),('f'*64,))
                with self.assertRaises(ValueError):target.verify_existing_crops(destination,[crop],'b'*64)
                self.assertEqual(target.snapshot(source),document['baseline_snapshot'])
            self.assertEqual(target.snapshot('public'),before)
        finally:
            with open_database() as c:
                for schema in (source,destination):c.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
