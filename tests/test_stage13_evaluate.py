import copy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from src.stage13_evaluate import validate_manifest, score_query, evaluate


def manifest():
    return {'format':'group8-stage13-annotations-v1','clips':[{'video_id':'clip1','duration_seconds':10,'source_sha256':'a'*64,'clip_path':'clip.mp4','source_filename':'source.mp4','source_offset_seconds':0,'camera_id':'G507'}], 'queries':[{'query_id':'q1','query':'person walking','target_category':'person','present':True,'review_status':'verified','reviewer':'human','independent_of_predictions':True,'scope_video_ids':['clip1'],'scope_review_complete':True,'relevant_intervals':[{'video_id':'clip1','start_seconds':2,'end_seconds':4}],'notes':''}]}


def hit(a,b,video='clip1'):
    return {'video_id':video,'start_seconds':a,'end_seconds':b}


class EvaluationTests(unittest.TestCase):
    def test_overlap_boundary_and_duplicates(self):
        q=manifest()['queries'][0]
        result=score_query(q,[hit(0,2.9),hit(0,2.9),hit(1,3)])
        self.assertEqual(result['hit_at_1'],0)
        self.assertEqual(result['hit_at_5'],1)
        self.assertEqual(result['mrr'],.5)
        self.assertEqual(result['precision_at_5'],.2)
        self.assertEqual(result['returned_candidates'],2)

    def test_negative(self):
        q=manifest()['queries'][0];q.update(present=False,relevant_intervals=[])
        self.assertEqual(score_query(q,[hit(0,2)])['return_rate'],1)
        self.assertEqual(score_query(q,[])['return_rate'],0)

    def test_proposed_is_never_evaluated(self):
        m=manifest();m['queries'][0]['review_status']='proposed'
        r=evaluate(m,search_fn=lambda *a,**k:self.fail('retrieval called'))
        self.assertEqual(r['status'],'awaiting_human_verification')
        self.assertEqual(r['aggregates'],{})
        self.assertEqual(len(r['skipped_queries']),1)

    def test_invalid_verified_evidence(self):
        changes=[{'reviewer':''},{'independent_of_predictions':False},{'scope_review_complete':False},{'scope_video_ids':[]},{'relevant_intervals':[hit(2,11)]},{'present':False},{'relevant_intervals':[hit(1,2,'other')]}]
        for change in changes:
            with self.subTest(change=change):
                m=manifest();m['queries'][0].update(change)
                with self.assertRaises(ValueError):validate_manifest(m)

    def test_modes_and_no_text_coverage(self):
        calls=[]
        def search(query,**kwargs):
            calls.append((query,kwargs))
            return {'results':[hit(2,4)],'candidate_counts':{'visual':1,'text':0}}
        r=evaluate(manifest(),search_fn=search)
        self.assertEqual(r['aggregates']['visual']['hit_at_1'],1)
        self.assertNotIn('text',r['aggregates'])
        self.assertNotIn('hybrid',r['aggregates'])
        self.assertEqual(len(calls),6)
        self.assertTrue(all(c[0]=='person walking' and c[1]['limit']==100 for c in calls))
        self.assertEqual(set(r['warmup_seconds']),{'visual','text','hybrid'})
        self.assertEqual(r['queries'][0]['modes']['visual']['latency_label'],'warm')

    def test_partial_and_complete_text_coverage(self):
        m=manifest()
        second=copy.deepcopy(m['clips'][0]);second['video_id']='clip2'
        m['clips'].append(second)
        m['queries'][0]['scope_video_ids'].append('clip2')
        def search(query,**kwargs):
            return {'results':[hit(2,4)],'candidate_counts':{'visual':1,'text':1}}
        search.text_coverage={'available_video_ids':['clip1'],'missing_video_ids':['clip2'],'complete':False}
        result=evaluate(m,search_fn=search)
        self.assertNotIn('hybrid',result['aggregates'])
        self.assertNotIn('text',result['aggregates'])
        self.assertEqual(result['text_coverage'],search.text_coverage)
        search.text_coverage={'available_video_ids':['clip1','clip2'],'missing_video_ids':[],'complete':True}
        result=evaluate(m,search_fn=search)
        self.assertEqual(result['aggregates']['hybrid']['mrr'],1)
        self.assertEqual(result['aggregates']['text']['mrr'],1)

    def test_out_of_scope_predictions_fail(self):
        with self.assertRaises(ValueError):score_query(manifest()['queries'][0],[hit(0,2,'other')])

    def test_dry_run_never_initializes_retrieval_and_refuses_overwrite(self):
        from src.stage13_evaluate import main
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'annotations.json';output=Path(root)/'report.json'
            source.write_text(json.dumps(manifest()))
            args=['--annotations',str(source),'--schema','stage13_test','--output',str(output),'--dry-run']
            with patch('src.stage13_evaluate.make_runner',side_effect=AssertionError('Model/DB access')):
                report=main(args)
                self.assertEqual(report['status'],'validation_only')
                self.assertEqual(len(report['annotations_sha256']),64)
                with self.assertRaises(FileExistsError):main(args)

    def test_video_must_match_and_short_interval_uses_overlap_coefficient(self):
        q=manifest()['queries'][0]
        self.assertEqual(score_query(q,[hit(2,2.2)])['hit_at_1'],1)
        self.assertEqual(score_query(q,[hit(0,2)])['hit_at_1'],0)

if __name__=='__main__':unittest.main()
