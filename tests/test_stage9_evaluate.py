"""Evaluation expectations derived from small manually calculated fixtures."""
import unittest

from src.stage9_evaluate import evaluate_observations


def prediction(frame, box=(0, 0, 10, 10), track=17):
    return dict(frame_index=frame, bbox_xyxy=list(box), local_track_id=track)


def annotation(frame, objects=None, visibility='present', revision=1):
    return dict(frame_index=frame, revision=revision, visibility=visibility,
                objects=objects if objects is not None else [dict(object_id='bag-A', bbox_xyxy=[0, 0, 10, 10])])


class EvaluationTests(unittest.TestCase):
    def test_empty_reviews_have_no_accuracy_claim(self):
        r = evaluate_observations([prediction(0)], [], 10)
        self.assertEqual(r['status'], 'awaiting_review')
        self.assertIsNone(r['precision'])
        self.assertIsNone(r['recall'])
        self.assertEqual(r['evaluated_frame_count'], 0)

    def test_only_latest_certain_reviews_count(self):
        rows = [prediction(0), prediction(1), prediction(2)]
        truth = [annotation(0), annotation(0, [], 'absent', 2), annotation(1, visibility='uncertain')]
        r = evaluate_observations(rows, truth, 10)
        self.assertEqual((r['true_positives'], r['false_positives'], r['false_negatives']), (0, 1, 0))
        self.assertEqual(r['precision'], 0)
        self.assertIsNone(r['recall'])
        self.assertEqual(r['uncertain_frame_count'], 1)
        self.assertEqual(r['unreviewed_frame_count'], 8)

    def test_missed_objects_duplicates_and_known_iou(self):
        truth = [annotation(0), annotation(1), annotation(2, [], 'absent')]
        rows = [prediction(0, (0, 0, 5, 10)), prediction(0), prediction(2)]
        r = evaluate_observations(rows, truth, 3)
        self.assertEqual((r['true_positives'], r['false_positives'], r['false_negatives']), (1, 2, 1))
        self.assertAlmostEqual(r['precision'], 1/3)
        self.assertEqual(r['recall'], .5)
        self.assertEqual(r['mean_matched_iou'], 1)

    def test_maximum_cardinality_assignment_avoids_greedy_miss(self):
        # Wide prediction can match either object; tight prediction only object A.
        truth = [annotation(0, [dict(object_id='A', bbox_xyxy=[0, 0, 10, 10]),
                                dict(object_id='B', bbox_xyxy=[5, 0, 15, 10])])]
        r = evaluate_observations([prediction(0, (0, 0, 15, 10)), prediction(0)], truth, 1)
        self.assertEqual(r['true_positives'], 2)
        self.assertEqual(r['false_negatives'], 0)

    def test_tracking_changes_only_between_adjacent_reviewed_frames(self):
        truth = [annotation(f) for f in [0, 1, 2, 3, 5]]
        rows = [prediction(0, track=17), prediction(1, track=27),
                prediction(3, track=None), prediction(5, track=99)]
        r = evaluate_observations(rows, truth, 6)
        self.assertEqual(r['adjacent_reviewed_identity_switches'], 1)
        self.assertEqual(r['adjacent_reviewed_track_gap_starts'], 1)
        self.assertEqual(r['matched_untracked_detections'], 1)
        self.assertEqual(r['motion_evaluation'], 'not_evaluated_requires_interval_ground_truth')

    def test_invalid_settings_and_duplicate_revisions_fail(self):
        for threshold in [0, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                evaluate_observations([], [], 10, iou_threshold=threshold)
        with self.assertRaises(ValueError):
            evaluate_observations([], [annotation(0), annotation(0)], 10)
        with self.assertRaises(ValueError):
            evaluate_observations([], [annotation(10)], 10)
