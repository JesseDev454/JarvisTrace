"""Stage 10 validation plus opt-in PostgreSQL tests in disposable schemas."""
from contextlib import closing
import os
import unittest
import uuid

import numpy as np
import psycopg
from psycopg import sql

from src.pg_config import connection_parameters, open_database, migrate
from src.visual_index import VisualIndex, normalize_visual_vector, encoder_identity


class VectorValidationTests(unittest.TestCase):
    def test_compatible_dimension_and_finite_normalization(self):
        vector = np.zeros(512); vector[0] = 3
        self.assertEqual(normalize_visual_vector(vector)[0], 1)
        for bad in [np.zeros(512), np.ones(384), np.full(512, np.nan), np.full(512, np.inf)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                normalize_visual_vector(bad)

    def test_encoder_identity_includes_preprocessing_and_model(self):
        a = encoder_identity('clip-A', 'resize224', '3.3', 512)
        self.assertEqual(a, encoder_identity('clip-A', 'resize224', '3.3', 512))
        self.assertNotEqual(a, encoder_identity('clip-B', 'resize224', '3.3', 512))
        self.assertNotEqual(a, encoder_identity('clip-A', 'resize336', '3.3', 512))


@unittest.skipUnless(os.getenv('GROUP8_TEST_POSTGRES') == '1', 'Set GROUP8_TEST_POSTGRES=1 for isolated-schema integration tests')
class PostgresVisualIndexTests(unittest.TestCase):
    def setUp(self):
        self.schema = 'stage10_test_' + uuid.uuid4().hex
        with psycopg.connect(**connection_parameters()) as c:
            c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        with open_database(schema=self.schema) as c:
            c.execute('''CREATE TABLE video_recordings (
                video_id TEXT PRIMARY KEY,local_video_path TEXT NOT NULL,camera_id TEXT NOT NULL,
                fps DOUBLE PRECISION NOT NULL,duration_seconds DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now())''')
            c.execute('''CREATE TABLE scenes (id BIGSERIAL PRIMARY KEY,video_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,start_seconds NUMERIC NOT NULL,end_seconds NUMERIC NOT NULL,
                description TEXT NOT NULL,embedding_model TEXT NOT NULL,embedding vector(384) NOT NULL)''')
            c.execute('''CREATE TABLE frame_candidates(id BIGSERIAL PRIMARY KEY,
                video_id TEXT REFERENCES video_recordings(video_id),timestamp_seconds NUMERIC,frame_path TEXT)''')
            c.execute('''CREATE TABLE temporal_window_index(video_id TEXT REFERENCES video_recordings(video_id),
                scene_id BIGINT REFERENCES scenes(id))''')
            c.execute("INSERT INTO video_recordings(video_id,local_video_path,camera_id,fps,duration_seconds) VALUES ('a','a.mp4','CAM-A',10,10),('b','b.mp4','CAM-B',10,10)")
            c.execute("INSERT INTO scenes(video_id,camera_id,start_seconds,end_seconds,description,embedding_model,embedding) VALUES ('a','CAM-A',0,1,'preserved manual scene','minilm',%s)", (np.ones(384),))
            self.original = c.execute('SELECT md5(row_to_json(s)::text) AS digest FROM scenes s').fetchone()['digest']
        migrate(schema=self.schema)
        self.index = VisualIndex(schema=self.schema)
        self.key = encoder_identity('model-A', 'resize224', '3.3', 512)

    def drop_schema(self):
        assert self.schema.startswith('stage10_test_') and len(self.schema) == 45
        with psycopg.connect(**connection_parameters()) as c:
            c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def sample(self, video='a', frame=0, vector=None, **changes):
        v = np.zeros(512); v[0] = 1
        return dict(video_id=video, source_sha256='a'*64, frame_index=frame,
                    source_camera_id='CAM-A' if video=='a' else 'CAM-B',
                    source_video_path=video+'.mp4',source_fps=10,source_duration_seconds=10,
                    timestamp_seconds=frame/10, variant='frame', bbox_xyxy=None,
                    image_path='C:/evidence/frame.png', image_sha256='b'*64,
                    embedding=vector if vector is not None else v,
                    crop_provenance=None) | changes

    def store(self, rows):
        return self.index.store_samples(rows, encoder_key=self.key,
                                        embedding_model='model-A', preprocess_version='resize224')

    def test_additive_migration_preserves_existing_records_and_repeats_safely(self):
        migrate(schema=self.schema)
        with open_database(schema=self.schema, readonly=True) as c:
            self.assertEqual(c.execute('SELECT md5(row_to_json(s)::text) AS digest FROM scenes s').fetchone()['digest'], self.original)
            self.assertEqual(c.execute('SELECT version_num FROM alembic_version').fetchone()['version_num'], '0003_visual_source_snapshot')
            self.assertEqual(c.execute("SELECT format_type(atttypid,atttypmod) AS t FROM pg_attribute WHERE attrelid='visual_embeddings'::regclass AND attname='embedding'").fetchone()['t'], 'vector(512)')

    def test_idempotent_insert_and_source_replacement_rejection(self):
        self.assertEqual(self.store([self.sample()]), 1)
        self.assertEqual(self.store([self.sample()]), 0)
        with self.assertRaises(ValueError):
            self.store([self.sample(source_sha256='c'*64)])
        self.assertEqual(self.index.stats()['total_embeddings'], 1)

    def test_filters_and_one_result_per_source_frame(self):
        low = np.zeros(512); low[1] = 1
        self.store([self.sample(), self.sample(variant='crop', bbox_xyxy=[1, 2, 10, 20],
                         crop_provenance={'class_id':24,'review_status':'unreviewed_detector_output'}),
                    self.sample(frame=20, vector=low), self.sample(video='b')])
        q = self.sample()['embedding']
        rows = self.index.search(q, self.key, limit=10)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(self.index.search(q, self.key, camera_id='CAM-A')), 2)
        self.assertEqual(len(self.index.search(q, self.key, video_id='b')), 1)
        filtered = self.index.search(q, self.key, camera_id='CAM-A', start_seconds=1, end_seconds=3)
        self.assertEqual([r['frame_index'] for r in filtered], [20])
        self.assertEqual(self.index.search(q, 'x'*64), [])

    def test_invalid_rows_roll_back_whole_batch(self):
        for bad in [self.sample(frame=100), self.sample(timestamp_seconds=5),
                    self.sample(variant='crop', bbox_xyxy=[0,0,0,1]),
                    self.sample(embedding=np.ones(384)), self.sample(video='missing')]:
            with self.subTest(bad=bad['video_id']), self.assertRaises(ValueError):
                self.store([self.sample(),bad])
            self.assertEqual(self.index.stats()['total_embeddings'], 0)

    def test_invalid_search_settings(self):
        for settings in [dict(limit=0), dict(start_seconds=-1), dict(start_seconds=3,end_seconds=2), dict(end_seconds=float('nan'))]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.index.search(np.ones(512), self.key, **settings)

    def test_search_provenance_survives_legacy_recording_metadata_overwrite(self):
        self.store([self.sample()])
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE video_recordings SET camera_id='OTHER',local_video_path='replacement.mp4' WHERE video_id='a'")
        rows = self.index.search(np.ones(512),self.key,camera_id='CAM-A')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['camera_id'],'CAM-A')
        self.assertEqual(rows[0]['source_video_path'],'a.mp4')
        self.assertEqual(self.index.search(np.ones(512),self.key,camera_id='OTHER'),[])
        with self.assertRaises(ValueError):
            self.store([self.sample(frame=1)])

    def test_rounded_registry_fps_preserves_actual_source_timing(self):
        fps=30000/1001
        with open_database(schema=self.schema) as c:
            c.execute("UPDATE video_recordings SET fps=29.97,duration_seconds=100 WHERE video_id='a'")
        sample=self.sample(frame=1000,timestamp_seconds=1000/fps,source_fps=fps,source_duration_seconds=100)
        self.assertEqual(self.store([sample]),1)
        row=self.index.search(np.ones(512),self.key)[0]
        self.assertAlmostEqual(row['timestamp_seconds'],33.36666666666667,places=6)
