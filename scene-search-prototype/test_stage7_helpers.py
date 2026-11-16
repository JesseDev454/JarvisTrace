import unittest
from pathlib import Path
from stage7_compare_vlms import time_from_name, WINDOWS

class HelpersTest(unittest.TestCase):
    def test_time_parse(self):
        self.assertEqual(time_from_name(Path('frame_000000096_0000004.000.jpg')), 4.0)
    def test_all_windows(self):
        self.assertEqual([len(x[2]) for x in WINDOWS.values()], [4, 4, 4])
    def test_window_three(self):
        self.assertEqual(WINDOWS[3][2], (6.0, 7.0, 8.0, 9.0))

if __name__ == '__main__':
    unittest.main()
