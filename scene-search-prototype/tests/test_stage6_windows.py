"""Stage 6 window tests (copied into scene-search-prototype/tests for unified pytest discovery)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal_window_index import select_evenly, make_windows, parse_approved


class Stage6WindowTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id': i + 1, 'timestamp_seconds': float(t), 'frame_path': f'{t}.jpg'}
                     for i, t in enumerate([0, 2, 3, 4, 5, 6, 7, 8, 9])]

    def test_overlapping_windows(self):
        w = make_windows(self.rows, 10.0, 4.0, 3.0, 4, 3)
        self.assertEqual([(x['start_seconds'], x['end_seconds']) for x in w],
                         [(0.0, 4.0), (3.0, 7.0), (6.0, 10.0)])
        self.assertEqual([[r['timestamp_seconds'] for r in x['rows']] for x in w],
                         [[0, 2, 3, 4], [3, 4, 6, 7], [6, 7, 8, 9]])

    def test_select_includes_endpoints(self):
        selected = select_evenly(self.rows, 4)
        self.assertEqual(selected[0]['timestamp_seconds'], 0)
        self.assertEqual(selected[-1]['timestamp_seconds'], 9)
        self.assertEqual(len(selected), 4)

    def test_skip_sparse_windows(self):
        sparse = [self.rows[0]]
        self.assertEqual(make_windows(sparse, 3.0, 2.0, 1.0, 4, 3), [])

    def test_parse_approval(self):
        self.assertEqual(parse_approved('1, 3'), {1, 3})
        with self.assertRaises(ValueError):
            parse_approved('')
        with self.assertRaises(ValueError):
            parse_approved('1,abc')


if __name__ == '__main__':
    unittest.main()
