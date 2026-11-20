import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
import subprocess
import shutil
from src.stage13_prepare import validate_segment,frame_extraction
from src.stage13_prepare import prepare


class SegmentTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required for exact frame mapping regression')
    def test_real_extraction_starts_and_ends_at_selected_frames(self):
        import cv2
        import numpy as np
        from src.stage13_dataset import sha256
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.avi';target=Path(directory)/'derived.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(96,64))
            self.assertTrue(writer.isOpened())
            for frame in range(300):writer.write(np.full((64,96,3),(frame*13)%240,dtype=np.uint8))
            writer.release();digest=sha256(source)
            _,command=frame_extraction(source,target,5,20,10)
            subprocess.run(command,capture_output=True,check=True,timeout=30)
            original=cv2.VideoCapture(str(source));derived=cv2.VideoCapture(str(target))
            try:
                self.assertEqual(derived.get(cv2.CAP_PROP_FRAME_COUNT),200)
                for frame in (0,100,199):
                    original.set(cv2.CAP_PROP_POS_FRAMES,frame+50);a,x=original.read()
                    derived.set(cv2.CAP_PROP_POS_FRAMES,frame);b,y=derived.read()
                    self.assertTrue(a and b)
                    self.assertLess(float(np.abs(x.astype(float)-y.astype(float)).mean()),3)
            finally:original.release();derived.release()
            self.assertEqual(sha256(source),digest)

    def test_frame_based_extraction_ignores_avi_timestamp_origin(self):
        spec,command=frame_extraction(Path('source.avi'),Path('clip.mp4'),250,40,30)
        self.assertEqual(spec['source_frame_start'],7500)
        self.assertEqual(spec['frame_count'],1200)
        self.assertNotIn('-ss',command)
        self.assertIn('trim=start_frame=7500:end_frame=8700,setpts=N/(30*TB)',command)
        self.assertEqual(command[command.index('-frames:v')+1],'1200')

    def test_bounds_and_safe_names(self):
        base=dict(clip_id='g507_01',source_filename='a.avi',start_seconds=20,duration_seconds=40)
        self.assertEqual(validate_segment(base,300),(20,40))
        for change in ({'clip_id':'../unsafe'},{'source_filename':'../a.avi'},
                       {'start_seconds':-1},{'duration_seconds':61},{'start_seconds':280},
                       {'start_seconds':float('nan')}):
            with self.assertRaises(ValueError):validate_segment(base|change,300)

    def test_interrupted_extraction_leaves_no_partial_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'a.avi';source.write_bytes(b'original')
            config=root/'selection.json'
            config.write_text(json.dumps(dict(format='group8-stage13-candidates-v1',selection_basis='test',
                segments=[dict(clip_id='clip1',source_filename='a.avi',start_seconds=0,duration_seconds=40)])))
            def fail(command,**kwargs):
                Path(command[-1]).write_bytes(b'incomplete')
                raise subprocess.TimeoutExpired(command,1)
            with patch('src.stage13_prepare.probe',return_value=dict(duration_seconds=300,sha256='a'*64,fps=30)),patch('src.stage13_prepare.subprocess.run',side_effect=fail):
                for _ in range(2):
                    with self.assertRaises(subprocess.TimeoutExpired):prepare(root,root/'derived',config)
                    self.assertEqual(list((root/'derived'/'clips').rglob('*.mp4')),[])
            self.assertEqual(source.read_bytes(),b'original')
