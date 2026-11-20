from datetime import datetime
import unittest
from src.stage13_dataset import filename_times, timeline


class TimelineTests(unittest.TestCase):
    def test_filename_time_without_invented_timezone(self):
        a,b,c=filename_times('2018-05-18.10-20-00.10-25-00.bus.G507.r13.avi')
        self.assertEqual(a,datetime(2018,5,18,10,20))
        self.assertEqual((b-a).total_seconds(),300)
        self.assertEqual(c,'G507')
        self.assertIsNone(a.tzinfo)

    def test_midnight_and_bad_filename(self):
        a,b,_=filename_times('2018-05-18.23-59-00.00-04-00.bus.G507.r13.avi')
        self.assertEqual((b-a).total_seconds(),300)
        with self.assertRaises(ValueError): filename_times('unknown.avi')

    def test_continuity_uses_filename_and_actual_duration(self):
        rows=[dict(filename='a',filename_start='2018-05-18T10:20:00',filename_end='2018-05-18T10:25:00',duration_seconds=299.8),
              dict(filename='b',filename_start='2018-05-18T10:25:00',filename_end='2018-05-18T10:30:00',duration_seconds=300),
              dict(filename='c',filename_start='2018-05-18T14:45:01',filename_end='2018-05-18T14:50:01',duration_seconds=300)]
        result=timeline(rows)
        self.assertEqual(result[1]['filename_gap_seconds'],0)
        self.assertAlmostEqual(result[1]['media_gap_seconds'],.2)
        self.assertEqual(result[2]['filename_gap_seconds'],15301)
