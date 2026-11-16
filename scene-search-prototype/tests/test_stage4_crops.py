"""Test pure crop geometry without loading the AI models."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Stub Stage 3 imports so this file can run in a clean test environment.
stub = ModuleType('auto_caption_frames')
stub.caption_one = lambda *a: None
stub.get_candidates = lambda *a: None
stub.load_vlm = lambda *a: None
sys.modules['auto_caption_frames'] = stub
spec = importlib.util.spec_from_file_location('crop_script', Path(__file__).parents[1] / 'compare_object_crops.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_crop_within_image():
    x1, y1, x2, y2 = m.context_bbox((380, 145, 455, 270), (640, 360))
    assert (x1 <= 380 <= 455 <= x2)
    assert (y1 <= 145 <= 270 <= y2)
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 360
    assert x2 - x1 >= 352


def test_crop_at_edge():
    box = m.context_bbox((600, 280, 637, 358), (640, 360))
    assert box[2] == 640 and box[3] == 360
    assert box[0] <= 600 and box[1] <= 280


def test_select_missing_returns_none():
    dets = [{'class_id': 0, 'confidence': .99, 'xyxy': [0, 0, 1, 1]}]
    assert m.select_best_target(dets) is None


def test_best_backpack():
    dets = [{'class_id': 24, 'confidence': .4},
            {'class_id': 56, 'confidence': .9},
            {'class_id': 24, 'confidence': .7}]
    assert m.select_best_target(dets)['confidence'] == .7
