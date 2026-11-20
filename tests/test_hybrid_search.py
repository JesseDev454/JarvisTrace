"""Hand-calculated window/fusion cases without models or databases."""
import unittest

from src.hybrid_search import filter_windows_by_similarity, fuse_windows


def visual(t,video='a',camera='CAM',sha='a'*64,similarity=.5):
    return dict(video_id=video,camera_id=camera,source_sha256=sha,source_video_path=video+'.mp4',
                duration_seconds=10,timestamp_seconds=t,similarity=similarity,frame_index=round(t*24),variant='frame')


def text(start,end,**kwargs):
    row=visual(start,**kwargs)
    row.pop('timestamp_seconds')
    row.update(start_seconds=start,end_seconds=end,scene_id=15,description='reviewed description',
               provenance_type='reviewed_temporal_caption')
    return row


class FusionTests(unittest.TestCase):
    def test_similarity_gate_requires_a_qualified_enabled_channel(self):
        rows = fuse_windows(
            [visual(1, similarity=.30)], [text(0, 4, similarity=.46)],
            window_seconds=4,
        )
        self.assertEqual(len(filter_windows_by_similarity(
            rows, mode='hybrid', min_visual_similarity=.28, min_text_similarity=.47,
        )), 1)
        self.assertEqual(filter_windows_by_similarity(
            rows, mode='hybrid', min_visual_similarity=.31, min_text_similarity=.47,
        ), [])
        self.assertEqual(filter_windows_by_similarity(
            rows, mode='text', min_visual_similarity=.28, min_text_similarity=.47,
        ), [])
        self.assertEqual(len(filter_windows_by_similarity(
            rows, mode='text', min_visual_similarity=.28, min_text_similarity=.45,
        )), 1)

    def test_similarity_gate_rejects_invalid_thresholds_and_preserves_order(self):
        rows = fuse_windows([visual(1, similarity=.3), visual(5, similarity=.5)], [])
        accepted = filter_windows_by_similarity(rows, mode='visual', min_visual_similarity=.25)
        self.assertEqual([row['start_seconds'] for row in accepted], [0, 4])
        for value in (float('nan'), float('inf'), -1.01, 1.01, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                filter_windows_by_similarity(rows, mode='visual', min_visual_similarity=value)

    def test_one_vote_per_channel_window_and_exact_rrf(self):
        rows=fuse_windows([visual(1),visual(2),visual(5)],[text(0,4)],window_seconds=4,rrf_k=60)
        self.assertEqual([(r['start_seconds'],r['end_seconds']) for r in rows],[(0,4),(4,8)])
        self.assertAlmostEqual(rows[0]['rrf_score'],2/61)
        self.assertEqual(rows[0]['channels']['visual']['rank'],1)
        self.assertEqual(rows[1]['channels']['visual']['rank'],2)
        self.assertAlmostEqual(rows[1]['rrf_score'],1/62)
        self.assertEqual(len(rows[0]['channels']['visual']['evidence']),2)

    def test_boundaries_and_partial_last_window(self):
        rows=fuse_windows([visual(4),visual(9.9)],[],window_seconds=4)
        self.assertEqual([(r['start_seconds'],r['end_seconds']) for r in rows],[(4,8),(8,10)])
        self.assertEqual(fuse_windows([],[],window_seconds=4),[])

    def test_sources_and_cameras_never_merge(self):
        rows=fuse_windows([visual(1),visual(1,video='b'),visual(1,camera='OTHER'),
                           visual(1,sha='b'*64)],[],window_seconds=4)
        self.assertEqual(len(rows),4)

    def test_filtered_bounds_are_clipped_and_end_exclusive(self):
        rows=fuse_windows([visual(5),visual(6),visual(8)],[text(3,7)],window_seconds=4,
                          start_seconds=6,end_seconds=8)
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['start_seconds'],rows[0]['end_seconds']),(6,8))
        self.assertEqual(rows[0]['channels']['visual']['evidence'][0]['timestamp_seconds'],6)
        self.assertEqual(rows[0]['channels']['text']['evidence'][0]['start_seconds'],3)

    def test_missing_channel_and_zero_weight(self):
        rows=fuse_windows([visual(1)],[text(4,8)],visual_weight=0)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['start_seconds'],4)
        self.assertEqual(set(rows[0]['channels']),{'text'})

    def test_invalid_ranges_weights_and_inputs(self):
        for kwargs in [dict(window_seconds=0),dict(rrf_k=-1),dict(text_weight=float('nan')),
                       dict(visual_weight=0,text_weight=0),dict(start_seconds=8,end_seconds=6)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                fuse_windows([],[],**kwargs)
        with self.assertRaises(ValueError):
            fuse_windows([visual(float('nan'))],[])
        with self.assertRaises(ValueError):
            fuse_windows([],[text(4,2)])


if __name__=='__main__':
    unittest.main()
