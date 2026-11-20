"""Exercise the real file-to-report path with a tiny generated video."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from src.stage9 import run


class Stage9CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'clip.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
        writer.release()
        self.labels = self.root / 'labels'
        self.labels.mkdir()
        (self.labels / 'clip_1.txt').write_text('24 .5 .5 .2 .4 .25\n')
        (self.labels / 'clip_10.txt').write_text('24 .5 .5 .2 .4 .75 27\n')
        self.output = self.root / 'result'

    def test_run_preserves_inputs_and_exports_provenance_and_blank_review(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [self.video, *self.labels.glob('*.txt')]}
        report = run(self.video, self.labels, self.output, 'CAM-TEST', 'test-clip', True)
        self.assertEqual(report['analysis']['observation_count'], 2)
        self.assertEqual(report['analysis']['last_observed_candidates'][0]['timestamp_seconds'], .9)
        self.assertEqual(report['provenance']['video_sha256'], before[self.video])
        self.assertFalse(report['database_updated'])
        self.assertEqual(report['analysis']['review_status'], 'unreviewed_detector_output')
        rows = [json.loads(line) for line in (self.output / 'observations.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['camera_id'], 'CAM-TEST')
        self.assertIsNone(rows[0]['local_track_id'])
        self.assertIn(report['provenance']['scope_id'], rows[1]['scoped_track_key'])
        with (self.output / 'human_review.csv').open(newline='', encoding='utf-8-sig') as f:
            review = list(csv.DictReader(f))
        self.assertEqual(len(review), 2)
        self.assertEqual(review[0]['human_object_id'], '')
        self.assertEqual(review[0]['human_surface'], '')
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        self.assertEqual(json.loads((self.output / 'analysis.json').read_text()), report)

    def test_reusing_output_directory_fails_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / 'analysis.json'
        marker.write_text('keep me')
        with self.assertRaises(FileExistsError):
            run(self.video, self.labels, self.output, 'CAM', 'clip')
        self.assertEqual(marker.read_text(), 'keep me')

    def test_missing_corrupt_video_and_bad_labels_leave_no_report_directory(self):
        for path in [self.root / 'absent.avi', self.root / 'bad.mp4']:
            if path.name == 'bad.mp4':
                path.write_text('invalid video')
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                run(path, self.labels, self.output, 'CAM', 'clip')
            self.assertFalse(self.output.exists())
        (self.labels / 'clip_1.txt').write_text('invalid')
        with self.assertRaises(ValueError):
            run(self.video, self.labels, self.output, 'CAM', 'clip')
        self.assertFalse(self.output.exists())

    def test_cli_requires_explicit_export_timing_and_format_assumptions(self):
        args = [sys.executable, '-m', 'src.stage9', '--video', str(self.video),
                '--labels', str(self.labels), '--output', str(self.output),
                '--camera', 'CAM', '--video-id', 'clip']
        rejected = subprocess.run(args, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('--confirm-save-conf', rejected.stderr)
        self.assertFalse(self.output.exists())
        accepted = subprocess.run(args + ['--confirm-save-conf', '--confirm-cfr-stride-one'],
                                  capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn('2 observations', accepted.stdout)
        self.assertTrue((self.output / 'analysis.json').is_file())


if __name__ == '__main__':
    unittest.main()
