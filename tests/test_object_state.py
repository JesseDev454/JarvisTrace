"""Hand-calculated Stage 9 cases; no models, database, GPU or network."""
import math
from pathlib import Path
import tempfile
import unittest

from src.object_state import VideoInfo, analyze_observations, load_yolo_labels


class ObjectStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.labels = Path(self.temp.name)
        self.video = VideoInfo(fps=10, width=80, height=60, frame_count=20)

    def label(self, frame, content):
        (self.labels / f'clip_{frame}.txt').write_text(content, encoding='utf-8')

    def load(self):
        return load_yolo_labels(self.labels, 'clip', self.video)

    def obs(self, frame, x=40, track=1):
        return {'frame_index': frame, 'timestamp_seconds': frame / 10,
                'class_id': 24, 'local_track_id': track, 'confidence': 0.6,
                'bbox_xyxy': [x - 4, 24, x + 4, 36], 'center_xy': [x, 30],
                'source_label': 'clip_1.txt', 'source_line': frame + 1}

    def test_import_offsets_numeric_sort_geometry_and_untracked(self):
        self.label(10, '24 0.5 0.5 0.2 0.4 0.7 27\n')
        self.label(1, '24 0.5 0.5 0.2 0.4 0.25\n0 0.5 0.5 0.1 0.1 0.8 2\n')
        self.label(2, '24 0.5 0.5 0.2 0.4 0.6 17\n')
        result = self.load()
        rows = result['observations']
        self.assertEqual([x['frame_index'] for x in rows], [0, 1, 9])
        self.assertEqual([x['timestamp_seconds'] for x in rows], [0, 0.1, 0.9])
        self.assertEqual(rows[0]['bbox_xyxy'], [32, 18, 48, 42])
        self.assertIsNone(rows[0]['local_track_id'])
        self.assertEqual(rows[2]['local_track_id'], 27)
        self.assertEqual(result['label_file_count'], 3)
        self.assertEqual(result['frames_without_label_file'], [[2, 8], [10, 19]])

    def test_reject_invalid_numeric_metadata(self):
        for kwargs in [dict(fps=math.nan), dict(fps=0), dict(width=0),
                       dict(height=-1), dict(frame_count=0), dict(frame_count=2.5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VideoInfo(**(dict(fps=10, width=80, height=60, frame_count=20) | kwargs))

    def test_reject_malformed_ambiguous_and_nonfinite_rows(self):
        for row in ['24 .5 .5 .2 .4', '24 .5 .5 .2 .4 .7 2 8',
                    '24 nan .5 .2 .4 .7 2', '24 .5 .5 .2 .4 inf 2',
                    '24 .5 .5 .2 .4 1.2 2', '24 .5 .5 -.2 .4 .7 2',
                    '24 0 .5 .2 .4 .7 2', '24 .5 .5 .2 .4 .7 2.5',
                    '24 .5 .5 .2 .4 .7 -1', '24.5 .5 .5 .2 .4 .7 2']:
            with self.subTest(row=row):
                self.label(1, row)
                with self.assertRaisesRegex(ValueError, 'clip_1.txt:1'):
                    self.load()

    def test_reject_out_of_range_foreign_and_duplicate_frames(self):
        self.label(21, '24 .5 .5 .2 .4 .7 1')
        with self.assertRaises(ValueError):
            self.load()
        (self.labels / 'clip_21.txt').unlink()
        (self.labels / 'other_1.txt').write_text('24 .5 .5 .2 .4 .7 1')
        with self.assertRaises(ValueError):
            self.load()
        (self.labels / 'other_1.txt').unlink()
        self.label(1, '24 .5 .5 .2 .4 .7 1\n24 .6 .5 .2 .4 .8 1')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_reject_appended_untracked_duplicates(self):
        self.label(1, '24 .5 .5 .2 .4 .7\n24 .5 .5 .2 .4 .7')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_integer_track_identity_is_not_rounded_through_float(self):
        self.label(1, '24 .5 .5 .2 .4 .7 9007199254740992')
        self.label(2, '24 .5 .5 .2 .4 .7 9007199254740993')
        ids = [r['local_track_id'] for r in self.load()['observations']]
        self.assertEqual(ids, [9007199254740992, 9007199254740993])
        self.label(1, '24 .5 .5 .2 .4 .7 1.00000000000000001')
        with self.assertRaises(ValueError):
            self.load()

    def test_no_labels_is_error_but_no_target_is_valid(self):
        with self.assertRaises(ValueError):
            self.load()
        self.label(1, '0 .5 .5 .2 .4 .7 1')
        self.assertEqual(self.load()['observations'], [])
        report = analyze_observations([], self.video)
        self.assertEqual(report['tracks'], [])
        self.assertEqual(report['last_observed_candidates'], [])
        self.assertEqual(report['frames_without_target_observation'], [[0, 19]])

    def test_extent_detects_slow_drift_not_just_neighbor_jitter(self):
        report = analyze_observations([self.obs(i, x=40+i) for i in range(10)],
                                      self.video, assume_static_camera=True)
        window = report['tracks'][0]['spatial_windows'][0]
        self.assertAlmostEqual(window['center_extent_diagonal_fraction'], .09)
        self.assertEqual(window['state'], 'image_displacement_candidate')
        self.assertAlmostEqual(report['tracks'][0]['net_center_displacement_pixels'], 9)

    def test_low_extent_candidate_requires_duration_and_static_camera_assumption(self):
        rows = [self.obs(i, x=40 + (i % 2)*.2) for i in range(10)]
        report = analyze_observations(rows, self.video, assume_static_camera=True)
        self.assertEqual(report['tracks'][0]['spatial_windows'][0]['state'],
                         'low_image_displacement_candidate')
        unknown = analyze_observations(rows, self.video)
        self.assertEqual(unknown['tracks'][0]['spatial_windows'][0]['state'],
                         'camera_motion_unassessed')
        short = analyze_observations(rows[:2], self.video, assume_static_camera=True)
        self.assertEqual(short['tracks'][0]['spatial_windows'][0]['state'],
                         'insufficient_duration')

    def test_gaps_break_windows_and_are_never_interpolated(self):
        rows = [self.obs(i) for i in [0, 1, 5, 6]]
        report = analyze_observations(rows, self.video, assume_static_camera=True)
        track = report['tracks'][0]
        self.assertEqual(track['gaps'], [[2, 4]])
        self.assertEqual(len(track['spatial_windows']), 2)
        self.assertEqual(report['frames_without_target_observation'], [[2, 4], [7, 19]])
        self.assertEqual([w['observation_count'] for w in track['spatial_windows']], [2, 2])

    def test_displacement_at_window_boundary_is_not_lost(self):
        rows = [self.obs(i, x=20 if i < 10 else 60) for i in range(20)]
        report = analyze_observations(rows, self.video, assume_static_camera=True)
        windows = report['tracks'][0]['spatial_windows']
        self.assertEqual(windows[1]['state'], 'image_displacement_candidate')
        self.assertEqual(windows[1]['first_frame'], 9)
        self.assertEqual(windows[1]['center_extent_pixels'], 40)

    def test_large_adjacent_shift_is_reported_even_at_short_final_window(self):
        rows = [self.obs(i, x=20 if i < 10 else 60) for i in range(11)]
        report = analyze_observations(rows, self.video, assume_static_camera=True)
        self.assertEqual(report['tracks'][0]['spatial_windows'][-1]['state'],
                         'image_displacement_candidate')

    def test_tracks_fragments_and_untracked_rows_never_merge(self):
        rows = [self.obs(0), self.obs(1), self.obs(4, track=2),
                self.obs(5, track=2), self.obs(5, x=60, track=None)]
        report = analyze_observations(rows, self.video)
        self.assertEqual([t['local_track_id'] for t in report['tracks']], [1, 2])
        self.assertEqual(report['untracked_observation_count'], 1)
        self.assertEqual(len(report['last_observed_candidates']), 2)
        self.assertEqual(len(report['possible_fragment_pairs']), 1)
        pair = report['possible_fragment_pairs'][0]
        self.assertEqual((pair['from_track_id'], pair['to_track_id']), (1, 2))
        self.assertEqual(pair['identity_status'], 'unresolved_not_merged')

    def test_concurrent_tracks_remain_ambiguous(self):
        report = analyze_observations([self.obs(0), self.obs(0, track=2)], self.video)
        self.assertEqual(report['concurrent_track_frames'], [[0, 0]])
        self.assertEqual(len(report['last_observed_candidates']), 2)
        self.assertEqual(report['possible_fragment_pairs'], [])

    def test_reject_bad_analysis_settings(self):
        for kwargs in [dict(window_seconds=0), dict(low_extent=math.nan),
                       dict(low_extent=.05, movement_extent=.03),
                       dict(minimum_span_seconds=-1), dict(fragment_gap_seconds=math.inf)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_observations([], self.video, **kwargs)


if __name__ == '__main__':
    unittest.main()
