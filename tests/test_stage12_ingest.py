import io
import base64
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.stage12_ingest import inspect_mp4, safe_filename, segment_ids, segment_start_from_record_path, sha256_file, store_upload


class UploadValidationTests(unittest.TestCase):
    def test_filename_is_reduced_to_a_safe_mp4_basename(self):
        self.assertEqual(safe_filename(r'C:\private\camera\lobby.mp4'), 'lobby.mp4')
        self.assertEqual(safe_filename('..\\secret.mp4'), 'secret.mp4')
        for name in ('../secret.txt', ' ', 'clip.mp4.exe'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_filename(name)

    def test_inspection_requires_one_cfr_h264_video_stream(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 640, 'height': 360,
                               'avg_frame_rate': '30/1', 'r_frame_rate': '30/1', 'time_base': '1/90000'}],
                   'format': {'duration': '12.5', 'format_name': 'mov,mp4,m4a'}}
        ffprobe_result = type('Result', (), {'stdout': json.dumps(payload)})()
        ffmpeg_result = type('Result', (), {'stderr': 'VFR:0.003333 (6/374) min: 2999 max: 3001 avg: 3000'})()
        with patch('src.stage12_ingest.subprocess.run', side_effect=[ffprobe_result, ffmpeg_result]):
            result = inspect_mp4(Path('clip.mp4'))
        self.assertEqual(result['fps'], 30)
        self.assertEqual(result['duration_seconds'], 12.5)

        payload['streams'][0]['codec_name'] = 'hevc'
        with patch('src.stage12_ingest.subprocess.run', return_value=type('Result', (), {'stdout': json.dumps(payload)})()):
            with self.assertRaisesRegex(ValueError, 'H.264'):
                inspect_mp4(Path('clip.mp4'))

    def test_accepts_cfr_segment_with_fmp4_duration_rounding(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 1920, 'height': 1072,
                               'avg_frame_rate': '15375/512', 'r_frame_rate': '30/1', 'time_base': '1/15360'}],
                   'format': {'duration': '68.266656', 'format_name': 'mov,mp4,m4a'}}
        ffprobe_result = type('Result', (), {'stdout': json.dumps(payload)})()
        ffmpeg_result = type('Result', (), {'stderr': 'VFR:0.003333 (6/1794) min: 511 max: 513 avg: 512'})()
        with patch('src.stage12_ingest.subprocess.run', side_effect=[ffprobe_result, ffmpeg_result]):
            result = inspect_mp4(Path('segment.mp4'))
        self.assertEqual(result['fps'], 30.029296875)
        self.assertEqual(result['duration_seconds'], 68.266656)

    def test_accepts_rounded_60_second_fmp4_when_timestamps_are_uniform(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 1920, 'height': 1072,
                               'avg_frame_rate': '54030/1799', 'r_frame_rate': '30/1', 'time_base': '1/90000'}],
                   'format': {'duration': '59.966656', 'format_name': 'mov,mp4,m4a'}}
        ffprobe_result = type('Result', (), {'stdout': json.dumps(payload)})()
        ffmpeg_result = type('Result', (), {'stderr': 'VFR:0.003333 (6/1794) min: 2999 max: 3001 avg: 2999'})()
        with patch('src.stage12_ingest.subprocess.run', side_effect=[ffprobe_result, ffmpeg_result]):
            result = inspect_mp4(Path('segment.mp4'))
        self.assertEqual(result['fps'], 54030 / 1799)

    def test_rejects_dropped_frames_despite_rounded_average_rate(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 1920, 'height': 1072,
                               'avg_frame_rate': '30/1', 'r_frame_rate': '30/1', 'time_base': '1/90000'}],
                   'format': {'duration': '59.966656', 'format_name': 'mov,mp4,m4a'}}
        ffprobe_result = type('Result', (), {'stdout': json.dumps(payload)})()
        ffmpeg_result = type('Result', (), {'stderr': 'VFR:0.019444 (7/353) min: 2999 max: 6000 avg: 3000'})()
        with patch('src.stage12_ingest.subprocess.run', side_effect=[ffprobe_result, ffmpeg_result]):
            with self.assertRaisesRegex(ValueError, 'constant-frame-rate'):
                inspect_mp4(Path('truncated.mp4'))

    def test_rejects_material_frame_rate_mismatch(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 640, 'height': 360,
                               'avg_frame_rate': '24/1', 'r_frame_rate': '30/1', 'time_base': '1/90000'}],
                   'format': {'duration': '12.5', 'format_name': 'mov,mp4,m4a'}}
        completed = type('Result', (), {'stdout': json.dumps(payload)})()
        with patch('src.stage12_ingest.subprocess.run', return_value=completed):
            with self.assertRaisesRegex(ValueError, 'constant-frame-rate'):
                inspect_mp4(Path('variable.mp4'))

    def test_h264_in_a_non_mp4_container_is_rejected_even_with_mp4_extension(self):
        payload = {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 640, 'height': 360,
                               'avg_frame_rate': '30/1', 'r_frame_rate': '30/1'}],
                   'format': {'duration': '12.5', 'format_name': 'matroska,webm'}}
        completed = type('Result', (), {'stdout': json.dumps(payload)})()
        with patch('src.stage12_ingest.subprocess.run', return_value=completed):
            with self.assertRaisesRegex(ValueError, 'MP4 container'):
                inspect_mp4(Path('renamed.mp4'))

    def test_rejects_oversize_without_leaving_a_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, '1 GB'):
                store_upload(io.BytesIO(b'0123456789'), filename='large.mp4', storage_root=Path(temp), max_bytes=4)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_corrupt_video_is_removed_after_probe_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch('src.stage12_ingest.inspect_mp4', side_effect=ValueError('bad video')):
            with self.assertRaisesRegex(ValueError, 'bad video'):
                store_upload(io.BytesIO(b'fake mp4'), filename='test.mp4', storage_root=Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_failed_duplicate_upload_does_not_delete_an_existing_immutable_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / 'fixed-id.mp4'
            existing.write_bytes(b'original verified footage')
            with self.assertRaisesRegex(ValueError, 'probe failed'):
                store_upload(io.BytesIO(b'new bytes'), filename='new.mp4', storage_root=root,
                             video_id='fixed-id', probe=lambda _path: (_ for _ in ()).throw(ValueError('probe failed')))
            self.assertEqual(existing.read_bytes(), b'original verified footage')

    def test_mediamtx_segment_identity_is_idempotent_and_timestamp_is_explicit_utc(self):
        first=segment_ids('CAM-1','CAM-1/1795000000-123456.mp4','a'*64)
        self.assertEqual(first,segment_ids('CAM-1','CAM-1/1795000000-123456.mp4','a'*64))
        self.assertNotEqual(first,segment_ids('CAM-1','CAM-1/1795000001-123456.mp4','a'*64))
        self.assertEqual(segment_start_from_record_path('1795000000-123456.mp4').isoformat(),
                         '2026-11-18T11:06:40.123456+00:00')
        self.assertIsNone(segment_start_from_record_path('not-a-timestamp.mp4'))

    @unittest.skipUnless(shutil.which('ffprobe') and shutil.which('ffmpeg'), 'FFmpeg and FFprobe are required for the real H.264 fixture check')
    def test_real_h264_upload_is_verified_and_published_immutably(self):
        encoded=(Path(__file__).parents[1]/'frontend/tests/fixtures/tiny.mp4.b64').read_text(encoding='utf-8')
        payload=base64.b64decode(encoded)
        with tempfile.TemporaryDirectory() as temp:
            saved=store_upload(io.BytesIO(payload),filename='camera_sample.mp4',storage_root=Path(temp))
            self.assertEqual(saved['filename'],'camera_sample.mp4')
            self.assertGreater(saved['duration_seconds'],0)
            self.assertEqual(saved['source_sha256'],sha256_file(saved['path']))
            duplicate=store_upload(io.BytesIO(payload),filename='same-content.mp4',storage_root=Path(temp),video_id=saved['video_id'])
            self.assertEqual(duplicate['path'],saved['path'])
            self.assertEqual(len(list(Path(temp).glob('*.mp4'))),1)


if __name__ == '__main__':
    unittest.main()
