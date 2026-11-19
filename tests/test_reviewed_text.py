from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

import tests.test_visual_index as baseline
from src.pg_config import open_database
from src.reviewed_text import ReviewedTextIndex, normalize_text_vector, TEXT_MODEL


class TextVectorTests(unittest.TestCase):
    def test_separate_finite_384_dimensional_space(self):
        self.assertAlmostEqual(float(np.linalg.norm(normalize_text_vector(np.ones(384)))),1,places=6)
        for vector in [np.ones(512),np.zeros(384),np.full(384,np.nan)]:
            with self.subTest(shape=vector.shape),self.assertRaises(ValueError):
                normalize_text_vector(vector)


@unittest.skipUnless(os.getenv('GROUP8_TEST_POSTGRES')=='1','PostgreSQL opt-in required')
class ReviewedTextTests(unittest.TestCase):
    setUp_base=baseline.PostgresVisualIndexTests.setUp
    drop_schema=baseline.PostgresVisualIndexTests.drop_schema
    sample=baseline.PostgresVisualIndexTests.sample
    store=baseline.PostgresVisualIndexTests.store

    def setUp(self):
        self.setUp_base()
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video=Path(self.tmp.name)/'source.mp4'
        self.video.write_bytes(b'fixed source content for provenance fixture')
        digest=hashlib.sha256(self.video.read_bytes()).hexdigest()
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE video_recordings SET local_video_path=%s WHERE video_id='a'",(str(self.video),))
            c.execute('ALTER TABLE frame_candidates ADD COLUMN clip_start_seconds NUMERIC, ADD COLUMN clip_end_seconds NUMERIC, ADD COLUMN description TEXT')
            c.execute('''ALTER TABLE temporal_window_index ADD COLUMN start_seconds NUMERIC,
                ADD COLUMN end_seconds NUMERIC, ADD COLUMN caption TEXT,
                ADD COLUMN model_id TEXT, ADD COLUMN prompt_version TEXT,
                ADD COLUMN frame_candidate_ids JSONB, ADD COLUMN frame_times_seconds JSONB,
                ADD COLUMN report_path TEXT''')
        self.store([self.sample(source_sha256=digest,source_video_path=str(self.video))])
        self.text_index=ReviewedTextIndex(schema=self.schema,exclusions=[])
        self.vector=np.zeros(384,dtype=np.float32);self.vector[0]=1

    def scene(self,description,start,end,model=TEXT_MODEL,camera='CAM-A'):
        with open_database(schema=self.schema) as c:
            return c.execute('''INSERT INTO scenes(video_id,camera_id,start_seconds,end_seconds,
                description,embedding_model,embedding) VALUES('a',%s,%s,%s,%s,%s,%s) RETURNING id''',
                (camera,start,end,description,model,self.vector)).fetchone()['id']

    def manual(self,description='backpack on desk',start=4,end=5):
        scene=self.scene(description,start,end)
        with open_database(schema=self.schema) as c:
            f=c.execute('''INSERT INTO frame_candidates(video_id,timestamp_seconds,clip_start_seconds,
                clip_end_seconds,frame_path,description) VALUES('a',%s,%s,%s,'fixture.jpg',%s) RETURNING id''',
                ((start+end)/2,start,end,description)).fetchone()['id']
        return scene,f

    def temporal(self,scene,ids,times):
        from psycopg.types.json import Jsonb
        with open_database(schema=self.schema) as c:
            c.execute('''INSERT INTO temporal_window_index(video_id,scene_id,start_seconds,end_seconds,
                caption,model_id,prompt_version,frame_candidate_ids,frame_times_seconds,report_path)
                VALUES('a',%s,0,4,'approved movement','vlm','prompt',%s,%s,'report.json')''',
                (scene,Jsonb(ids),Jsonb(times)))

    def test_manual_and_approved_temporal_only_exclude_arbitrary_scenes(self):
        manual,f=self.manual(start=0,end=1)
        _,g=self.manual('second manual frame',start=2,end=3)
        temporal=self.scene('approved movement',0,4)
        self.temporal(temporal,[f,g],[.5,2.5])
        self.scene('unreviewed AI output',6,7)
        self.scene('fictional seeded row',8,9)
        r=self.text_index.search(self.vector)
        self.assertEqual({row['scene_id'] for row in r['results']},{manual,temporal,manual+1})
        self.assertEqual({row['provenance_type'] for row in r['results']},
                         {'legacy_manual_frame_label','reviewed_temporal_caption'})

    def test_filters_and_exact_exclusion_policy(self):
        scene,_=self.manual()
        self.manual('earlier label',start=0,end=1)
        r=self.text_index.search(self.vector,start_seconds=4,end_seconds=6,camera_id='CAM-A')
        self.assertEqual([row['scene_id'] for row in r['results']],[scene])
        self.assertEqual(self.text_index.search(self.vector,camera_id='WRONG')['results'],[])
        self.text_index.exclusions=[dict(video_id='a',scene_id=scene,
            description_sha256=hashlib.sha256(b'backpack on desk').hexdigest(),reason='fixture exclusion')]
        r=self.text_index.search(self.vector,start_seconds=4)
        self.assertEqual(r['results'],[])
        self.assertEqual(r['excluded_counts']['policy'],1)

    def test_invalid_temporal_frame_references_and_changed_caption_excluded(self):
        scene=self.scene('approved movement',0,4)
        self.temporal(scene,[999,1000],[0,2])
        self.assertEqual(self.text_index.search(self.vector)['results'],[])
        self.manual('valid manual')
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE scenes SET description='changed without frame review' WHERE description='valid manual'")
        self.assertEqual(self.text_index.search(self.vector)['results'],[])

    def test_changed_source_or_camera_never_binds_old_text(self):
        self.manual()
        self.video.write_bytes(b'replacement')
        self.assertEqual(self.text_index.search(self.vector)['results'],[])
        self.video.write_bytes(b'fixed source content for provenance fixture')
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE video_recordings SET camera_id='OTHER' WHERE video_id='a'")
        self.assertEqual(self.text_index.search(self.vector)['results'],[])

    def test_wrong_text_model_and_bad_bounds(self):
        scene,_=self.manual()
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE scenes SET embedding_model='incompatible:384' WHERE id=%s",(scene,))
        self.assertEqual(self.text_index.search(self.vector)['results'],[])
        for kwargs in [dict(start_seconds=-1),dict(limit=0),dict(start_seconds=3,end_seconds=2)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.text_index.search(self.vector,**kwargs)

    def test_manual_frame_outside_its_interval_is_excluded(self):
        _,frame=self.manual()
        with open_database(schema=self.schema) as c:
            c.execute('UPDATE frame_candidates SET timestamp_seconds=9 WHERE id=%s',(frame,))
        self.assertEqual(self.text_index.search(self.vector)['results'],[])
