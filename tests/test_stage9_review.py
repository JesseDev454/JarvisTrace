import hashlib
from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest
import sqlite3

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.stage9 import run
from src.stage9_review import create_app


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.video = root / 'clip.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 60))
        for _ in range(10):
            writer.write(np.zeros((60, 80, 3), np.uint8))
        writer.release()
        labels = root / 'labels'
        labels.mkdir()
        (labels / 'clip_1.txt').write_text('24 .5 .5 .2 .4 .7 17')
        report = root / 'report'
        run(self.video, labels, report, 'CAM', 'clip')
        self.report = report / 'analysis.json'
        self.db = root / 'review.sqlite3'
        self.client = TestClient(create_app(self.report, self.db))
        self.addCleanup(self.client.close)

    def payload(self, **changes):
        return dict(reviewer='Human reviewer', visibility='present', motion='unknown',
                    objects=[dict(object_id='bag-A', bbox_xyxy=[32, 18, 48, 42], surface='chair')],
                    notes='', expected_revision=0) | changes

    def save(self, payload, frame=0):
        return self.client.put(f'/api/frames/{frame}/annotation', json=payload,
                               headers={'X-Review-Client': 'stage9'})

    def test_page_frames_images_and_empty_predictions(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        info = self.client.get('/api/info').json()
        self.assertEqual(info['frame_count'], 10)
        self.assertEqual(self.client.get('/api/frames/0').json()['predictions'][0]['local_track_id'], 17)
        frame = self.client.get('/api/frames/9').json()
        self.assertEqual(frame['predictions'], [])
        self.assertIsNone(frame['annotation'])
        image = self.client.get('/api/frames/9/image')
        self.assertEqual(image.headers['content-type'], 'image/jpeg')
        self.assertTrue(image.content.startswith(b'\xff\xd8'))
        for index in [-1, 10]:
            self.assertEqual(self.client.get(f'/api/frames/{index}').status_code, 404)
            self.assertEqual(self.client.get(f'/api/frames/{index}/image').status_code, 404)

    def test_configured_reviewer_does_not_create_annotations(self):
        client = TestClient(create_app(self.report, self.db, default_reviewer='Goodluck Kassa'))
        self.addCleanup(client.close)
        self.assertEqual(client.get('/api/info').json()['default_reviewer'], 'Goodluck Kassa')
        self.assertEqual(client.get('/api/export').json()['history'], [])
        self.assertEqual(client.get('/api/evaluation').json()['status'], 'awaiting_review')

    def test_annotations_persist_separately_and_preserve_history(self):
        original = self.report.read_bytes()
        self.assertEqual(self.save(self.payload()).status_code, 200)
        self.assertEqual(self.save(self.payload()).status_code, 409)
        self.assertEqual(self.save(self.payload(expected_revision=1, notes='corrected')).status_code, 200)
        reopened = TestClient(create_app(self.report, self.db))
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get('/api/frames/0').json()['annotation']['revision'], 2)
        exported = reopened.get('/api/export').json()
        self.assertEqual(len(exported['history']), 2)
        self.assertEqual(exported['history'][0]['objects'][0]['object_id'], 'bag-A')
        self.assertEqual(self.report.read_bytes(), original)
        scored = reopened.get('/api/evaluation').json()
        self.assertEqual(scored['true_positives'], 1)
        self.assertEqual(scored['precision'], 1)

    def test_reject_bad_annotations_and_cross_origin_writes(self):
        for changes in [dict(reviewer=''), dict(visibility='present', objects=[]),
                        dict(visibility='absent'),
                        dict(objects=[dict(object_id='A', bbox_xyxy=[-1, 0, 10, 10])]),
                        dict(objects=[dict(object_id='A', bbox_xyxy=[0, 0, 90, 70])])]:
            with self.subTest(changes=changes):
                self.assertEqual(self.save(self.payload(**changes)).status_code, 422)
        endpoint = '/api/frames/0/annotation'
        self.assertEqual(self.client.put(endpoint, json=self.payload()).status_code, 403)
        self.assertEqual(self.client.put(endpoint, json=self.payload(),
            headers={'X-Review-Client': 'stage9', 'Origin': 'https://foreign.example'}).status_code, 403)
        self.assertEqual(self.save(self.payload(visibility='absent', objects=[]), frame=9).status_code, 200)

    def test_reject_modified_video_and_cross_report_review_store(self):
        self.save(self.payload())
        original = json.loads(self.report.read_text())
        changed = json.loads(self.report.read_text())
        changed['provenance']['scope_id'] = 'different'
        self.report.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            create_app(self.report, self.db)
        self.report.write_text(json.dumps(original))
        with self.video.open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(ValueError):
            create_app(self.report, self.db)

    def test_backpack_review_rejects_other_target_classes(self):
        report = json.loads(self.report.read_text())
        report['target_class_id'] = 0
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'backpack'):
            create_app(self.report, self.db)

    def test_assisted_labels_are_excluded_from_human_evaluation(self):
        from src.stage9_evaluate import evaluate_report
        self.save(self.payload())
        row = self.client.get('/api/frames/0').json()['annotation']
        row.update(revision=2, reviewer='Codex AI-assisted visual review',
                   annotation_source='ai_assisted_visual_review')
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('INSERT INTO reviews VALUES (?,?,?)', (0, 2, json.dumps(row)))
        human = evaluate_report(self.report, self.db)
        self.assertEqual(human['evaluated_frame_count'], 0)
        self.assertEqual(human['excluded_assisted_frame_count'], 1)
        self.assertIsNone(human['precision'])
        assisted = evaluate_report(self.report, self.db, include_assisted=True)
        self.assertEqual(assisted['evaluated_frame_count'], 1)
        self.assertEqual(assisted['reference_type'], 'includes_ai_assisted_reference')


if __name__ == '__main__':
    unittest.main()
