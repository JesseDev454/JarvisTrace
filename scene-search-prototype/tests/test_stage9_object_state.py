"""Unit tests for Stage 9 deterministic object-state analysis."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_state import (
    analyze_object_state,
    find_last_supported_sighting,
    parse_frame_index,
)


class TestStage9ObjectState(unittest.TestCase):
    def test_parse_frame_index(self):
        self.assertEqual(parse_frame_index('test01_1'), 1)
        self.assertEqual(parse_frame_index('test01_240'), 240)
        with self.assertRaises(ValueError):
            parse_frame_index('no_number')

    def test_parses_untracked_and_tracked_and_isolates_same_frame_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            # Frame 1: untracked backpack (6 cols) at same position as track 17
            (labels / 'clip_1.txt').write_text('24 0.65 0.60 0.11 0.28 0.24\n', encoding='utf-8')
            # Frame 2..7: tracked backpack ID 17 AND tracked person ID 17 at DIFFERENT positions in same frame
            for idx in range(2, 8):
                (labels / f'clip_{idx}.txt').write_text(
                    '0 0.20 0.50 0.30 0.80 0.85 17\n'
                    '24 0.65 0.60 0.11 0.28 0.45 17\n',
                    encoding='utf-8',
                )
            # Frame 10..15: tracked backpack ID 27 at a new position (stationary there)
            for idx in range(10, 16):
                (labels / f'clip_{idx}.txt').write_text(
                    '0 0.25 0.50 0.30 0.80 0.88 17\n'
                    '24 0.80 0.59 0.13 0.29 0.55 27\n',
                    encoding='utf-8',
                )

            report = analyze_object_state(labels, fps=24.0, target_class_id=24)
            self.assertEqual(report['untracked_target_detections'], 1)
            self.assertTrue(report['untracked_summary']['pre_track_continuation_candidate'])
            self.assertEqual(len(report['target_tracks']), 2)
            self.assertEqual(len(report['person_tracks']), 1)

            t17, t27 = report['target_tracks']
            self.assertEqual(t17['track_id'], 17)
            self.assertEqual(t17['motion_state'], 'stationary')
            self.assertEqual(t27['track_id'], 27)
            self.assertEqual(t27['motion_state'], 'stationary')

            gaps = report['gap_and_fragment_analysis']
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0]['kind'], 'fragment_link_candidate')
            self.assertEqual(gaps[0]['gap_frames'], 2)
            self.assertEqual(gaps[0]['overlapping_person_track_ids'], [17])
            self.assertGreater(gaps[0]['spatial_jump'], 0.1)

            last = report['last_supported_sighting']
            self.assertIsNotNone(last)
            self.assertEqual(last['track_id'], 27)
            self.assertEqual(last['last_frame'], 15)

    def test_audits_transient_bytetrack_class_flip_and_motion_onset(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            # Frames 1..5: backpack track 17 stationary at (0.65, 0.60)
            for idx in range(1, 6):
                (labels / f'clip_{idx}.txt').write_text(
                    '24 0.65 0.60 0.16 0.40 0.50 17\n', encoding='utf-8'
                )
            # Frame 6: track 17 transiently misclassified as class 0 at nearly identical box
            (labels / 'clip_6.txt').write_text(
                '0 0.65 0.59 0.16 0.40 0.16 17\n', encoding='utf-8'
            )
            # Frame 7: track 17 returns to class 24 and moves noticeably
            (labels / 'clip_7.txt').write_text(
                '24 0.70 0.55 0.16 0.40 0.40 17\n', encoding='utf-8'
            )

            report = analyze_object_state(labels, fps=24.0, target_class_id=24)
            # Should NOT create a separate person track 17
            self.assertEqual(len(report['person_tracks']), 0)
            self.assertEqual(len(report['target_tracks']), 1)
            t17 = report['target_tracks'][0]
            self.assertEqual(t17['class_flip_frames'], [6])
            self.assertEqual(t17['motion_state'], 'movement_candidate')
            self.assertIn('motion_phase_split', t17)
            self.assertEqual(t17['motion_phase_split']['stationary_until_frame'], 5)
            self.assertEqual(t17['motion_phase_split']['movement_onset_frame'], 7)

    def test_transient_late_track_does_not_override_supported_track(self):
        tracks = [
            {
                'class_id': 24,
                'class_name': 'backpack',
                'track_id': 1,
                'start_frame': 10,
                'end_frame': 100,
                'start_seconds': 0.375,
                'end_seconds': 4.125,
                'observation_count': 50,
                'mean_confidence': 0.50,
                'end_center': [0.8, 0.59],
                'motion_state': 'stationary',
            },
            {
                'class_id': 24,
                'class_name': 'backpack',
                'track_id': 2,
                'start_frame': 200,
                'end_frame': 201,
                'start_seconds': 8.292,
                'end_seconds': 8.333,
                'observation_count': 2,
                'mean_confidence': 0.90,
                'end_center': [0.1, 0.2],
                'motion_state': 'stationary',
            },
        ]
        last = find_last_supported_sighting(tracks, min_observations=5, min_mean_confidence=0.25)
        self.assertIsNotNone(last)
        self.assertEqual(last['track_id'], 1)

        none_supported = find_last_supported_sighting(tracks, min_observations=100)
        self.assertIsNone(none_supported)


if __name__ == '__main__':
    unittest.main()
