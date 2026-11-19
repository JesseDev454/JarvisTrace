import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import stage13_index as target


class PlanningTests(unittest.TestCase):
    def test_schema_validation_rejects_public_and_injection(self):
        for value in ['public', 'stage13_', 'stage13_a;drop schema public', 'stage13_A']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                target.validate_schema(value)
        self.assertEqual(target.validate_schema('stage13_test_123'), 'stage13_test_123')

    def test_dry_run_has_no_database_or_encoder_side_effects(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'clip.avi'; source.write_bytes(b'fixture')
            manifest=Path(root)/'manifest.json'
            manifest.write_text(json.dumps({'clips':[dict(video_id='a', camera_id='CAM',
                clip_path=str(source), source_sha256=hashlib.sha256(b'fixture').hexdigest())]}))
            with patch.object(target,'create_isolated_schema',side_effect=AssertionError('DB accessed')):
                result=target.run(manifest,'stage13_test_plan',Path(root)/'out')
            self.assertEqual(result['status'],'planned')
            self.assertEqual(result['predicted_images'],20)
            self.assertFalse((Path(root)/'out').exists())

    def test_workload_and_manifest_rejections(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'manifest.json'; p.write_text('{"clips": []}')
            for kwargs in [dict(max_frames=61),dict(batch_size=5),dict(max_recordings=11),
                           dict(max_recordings=10,max_frames=60),dict(every_seconds=0)]:
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                    target.run(p,'stage13_test_plan',Path(root)/'out',**kwargs)
            p.write_text(json.dumps({'clips':[dict(video_id='a',camera_id='C',clip_path='missing',source_sha256='x')]}))
            with self.assertRaises(ValueError):
                target.run(p,'stage13_test_plan',Path(root)/'out')

    def test_profile_changes_are_rejected(self):
        profile=dict(video_id='a',local_video_path='C:/a.avi',camera_id='C',fps=10.,duration_seconds=1.)
        target.compare_profile(profile,profile)
        for key,value in [('camera_id','D'),('fps',11),('local_video_path','C:/b.avi')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                target.compare_profile(dict(profile,**{key:value}),profile)


@unittest.skipUnless(os.getenv('GROUP8_TEST_POSTGRES')=='1','PostgreSQL opt-in required')
class IsolatedDatabaseTests(unittest.TestCase):
    def test_owned_schema_registration_and_public_preservation(self):
        import uuid
        import cv2
        import numpy as np
        from psycopg import sql
        from src.pg_config import open_database
        from src.visual_index import VisualIndex
        schema='stage13_test_'+uuid.uuid4().hex
        with open_database(readonly=True) as c:
            before=c.execute('SELECT count(*) AS n FROM public.video_recordings').fetchone()['n']
        try:
            target.create_isolated_schema(schema)
            target.create_isolated_schema(schema)
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'tiny.avi'
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
                self.assertTrue(writer.isOpened())
                for _ in range(10): writer.write(np.zeros((32,32,3),np.uint8))
                writer.release()
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                index=VisualIndex(schema=schema)
                target.register_recording(index,path,'CAM','tiny',digest)
                target.register_recording(index,path,'CAM','tiny',digest)
                with self.assertRaises(ValueError): target.register_recording(index,path,'OTHER','tiny',digest)
                self.assertEqual(index.recording('tiny')['camera_id'],'CAM')
                with open_database(schema=schema) as c:
                    c.execute(sql.SQL('DROP TABLE {}.frame_candidates').format(sql.Identifier(schema)))
                with self.assertRaises(ValueError): target.create_isolated_schema(schema)
                with open_database(schema=schema) as c:
                    c.execute(sql.SQL('CREATE TABLE {}.frame_candidates AS SELECT * FROM public.frame_candidates WITH NO DATA').format(sql.Identifier(schema)))
                with open_database(schema=schema) as c:
                    c.execute(sql.SQL('DROP TABLE {}.video_recordings CASCADE').format(sql.Identifier(schema)))
                with self.assertRaises(ValueError):
                    target.register_recording(index,path,'CAM','must_not_enter_public',digest)
                with self.assertRaises(ValueError):
                    target.create_isolated_schema(schema)
            with open_database(readonly=True) as c:
                self.assertEqual(c.execute('SELECT count(*) AS n FROM public.video_recordings').fetchone()['n'],before)
        finally:
            with open_database() as c:
                c.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
