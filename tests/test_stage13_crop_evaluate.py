import copy
import unittest
from src.stage13_crop_evaluate import evaluate


def manifest():
    return {'format':'group8-stage13-annotations-v1','clips':[{'video_id':'v','duration_seconds':12,'source_sha256':'a'*64,'clip_path':'v.mp4','source_filename':'v.mp4','source_offset_seconds':0,'camera_id':'G507'}], 'queries':[{'query_id':'q','query':'backpack','target_category':'bag','present':True,'review_status':'verified','reviewer':'human','independent_of_predictions':True,'scope_review_complete':True,'scope_video_ids':['v'],'relevant_intervals':[hit(4,8)]}]}


def hit(a,b):
    return {'video_id':'v','start_seconds':a,'end_seconds':b}


def baseline():
    metrics={'returned_candidates':2,'hit_at_1':0,'hit_at_5':1,'mrr':.5,'precision_at_5':.2}
    return {'annotations_sha256':'digest','queries':[{'query_id':'q','query':'backpack','present':True,'modes':{'visual':{'status':'evaluated','metrics':metrics,'results':[hit(0,4),hit(4,8)]}}}], 'aggregates':{'visual':{'positive_queries':1,'negative_queries':0,'hit_at_1':0,'hit_at_5':1,'mrr':.5,'precision_at_5':.2}}}


class CropEvaluationTests(unittest.TestCase):
    def test_mixed_labels_do_not_include_negatives_in_positive_denominators(self):
        m=manifest();negative=copy.deepcopy(m['queries'][0]);negative.update(query_id='n',query='missing laptop',present=False,relevant_intervals=[])
        m['queries'].append(negative);b=baseline()
        old=copy.deepcopy(b['queries'][0]);old.update(query_id='n',query=negative['query'],present=False)
        old['modes']['visual']['metrics']={'returned_candidates':2,'return_rate':1}
        b['queries'].append(old)
        b['aggregates']['visual'].update(negative_queries=1,negative_return_rate=1.,negative_returned_candidates=[2])
        report=evaluate(m,search_fn=lambda *a,**k:{'results':[hit(0,4),hit(4,8)],'candidate_counts':{'visual':2,'text':0}},baseline_report=b,annotations_sha256='digest')
        self.assertEqual(report['aggregates']['combined']['positive_queries'],1)
        self.assertEqual(report['aggregates']['combined']['negative_queries'],1)
        self.assertEqual(report['aggregates']['combined']['mrr'],.5)
        self.assertEqual(report['aggregates']['combined']['negative_return_rate'],1.)

    def test_negatives_report_candidate_return_without_accuracy_or_positive_metrics(self):
        m=manifest();q=m['queries'][0];q.update(query='red suitcase',present=False,relevant_intervals=[])
        b=baseline();old=b['queries'][0];old.update(query=q['query'],present=False)
        old['modes']['visual']['metrics']={'returned_candidates':2,'return_rate':1}
        b['aggregates']['visual']={'positive_queries':0,'negative_queries':1,'negative_return_rate':1.,'negative_returned_candidates':[2]}
        def search(query,**settings):
            results=[hit(0,4),hit(4,8)] if settings['variant']!='crop' else []
            return {'results':results,'candidate_counts':{'visual':len(results),'text':0}}
        report=evaluate(m,search_fn=search,baseline_report=b,annotations_sha256='digest')
        self.assertFalse(report['queries'][0]['present'])
        self.assertIsNone(report['queries'][0]['modes']['full_frame']['first_relevant_rank'])
        self.assertEqual(report['aggregates']['crop_only']['negative_return_rate'],0.)
        self.assertEqual(report['aggregates']['combined']['negative_return_rate'],1.)
        for values in report['aggregates'].values():
            self.assertNotIn('hit_at_1',values)
            self.assertNotIn('accuracy',values)

    def test_variants_warmups_and_only_verified_score(self):
        m=manifest();proposed=copy.deepcopy(m['queries'][0]);proposed.update(query_id='pending',review_status='proposed');m['queries'].append(proposed)
        calls=[]
        def search(query,**settings):
            calls.append((query,settings))
            return {'results':[hit(0,4),hit(4,8)] if settings['variant']=='frame' else [hit(4,8)],'candidate_counts':{'visual':2,'text':0}}
        report=evaluate(m,search_fn=search,baseline_report=baseline(),annotations_sha256='digest')
        self.assertEqual(len(calls),6)
        self.assertEqual([x[1]['variant'] for x in calls],[ 'frame','crop',None,'frame','crop',None])
        for query,settings in calls:
            self.assertEqual(query,'backpack')
            self.assertEqual({k:v for k,v in settings.items() if k!='variant'},dict(visual_weight=1,text_weight=0,window_seconds=4,rrf_k=60,candidate_limit=100,limit=100))
        self.assertEqual(report['queries'][0]['modes']['full_frame']['first_relevant_rank'],2)
        self.assertEqual(report['queries'][0]['modes']['crop_only']['first_relevant_rank'],1)
        self.assertEqual(report['aggregates']['combined']['precision_at_5'],.2)
        self.assertEqual(len(report['skipped_queries']),1)
        self.assertTrue(report['baseline_reproduction']['passed'])

    def test_baseline_drift_rejected_even_when_metrics_match(self):
        def search(query,**settings):
            return {'results':[hit(0,2),hit(4,8)],'candidate_counts':{'visual':2,'text':0}}
        with self.assertRaisesRegex(ValueError,'ranked'):
            evaluate(manifest(),search_fn=search,baseline_report=baseline(),annotations_sha256='digest')

    def test_hash_mismatch_prevents_search(self):
        with self.assertRaisesRegex(ValueError,'SHA256'):
            evaluate(manifest(),search_fn=lambda *a,**k:self.fail('search called'),baseline_report=baseline(),annotations_sha256='different')

    def test_prediction_validation(self):
        for bad in [dict(hit(0,4),source_sha256='b'*64),hit(8,14),dict(hit(0,4),video_id='elsewhere')]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                evaluate(manifest(),search_fn=lambda *a,**k:{'results':[bad],'candidate_counts':{'visual':1,'text':0}},baseline_report=baseline(),annotations_sha256='digest')

    def test_metric_drift_rejected(self):
        b=baseline();b['aggregates']['visual']['mrr']=1
        with self.assertRaisesRegex(ValueError,'aggregate'):
            evaluate(manifest(),search_fn=lambda *a,**k:{'results':[hit(0,4),hit(4,8)],'candidate_counts':{'visual':2,'text':0}},baseline_report=b,annotations_sha256='digest')

class CropCliTests(unittest.TestCase):
    def test_existing_output_rejected_before_any_model_or_database_access(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.stage13_crop_evaluate import main
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json';output.write_text('keep')
            with patch('src.stage13_crop_evaluate.make_runner',side_effect=AssertionError('DB/model access')):
                with self.assertRaises(FileExistsError):
                    main(['--annotations','missing','--schema','stage13_test','--output',str(output)])
            self.assertEqual(output.read_text(),'keep')

    def test_owned_tables_checked_before_runner_initializes(self):
        from unittest.mock import patch
        from src.stage13_crop_evaluate import make_runner
        with patch('src.stage13_index.require_local_tables',side_effect=ValueError('ownership')):
            with patch('src.stage13_evaluate.make_runner',side_effect=AssertionError('model access')):
                with self.assertRaisesRegex(ValueError,'ownership'):
                    make_runner(manifest(),'stage13_test')
