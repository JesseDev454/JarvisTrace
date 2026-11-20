import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from ingest_video import changed_pixel_fraction, choose_candidate


def test_identical_frames_have_zero_change():
    frame = np.zeros((4, 4), dtype=np.uint8)
    assert changed_pixel_fraction(frame, frame) == 0


def test_completely_different_frames():
    frame1 = np.zeros((4, 4), dtype=np.uint8)
    frame2 = np.full((4, 4), 255, dtype=np.uint8)
    assert changed_pixel_fraction(frame1, frame2) == 1


def test_keep_initial_and_meaningful_change():
    assert choose_candidate(None, 0, 0.08, 5) == 'first-frame'
    assert choose_candidate(0.2, 1, 0.08, 5) == 'visual-change'
    assert choose_candidate(0.02, 1, 0.08, 5) is None
    assert choose_candidate(0.02, 5, 0.08, 5) == 'guard-sample'
