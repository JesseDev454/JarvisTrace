import unittest
from unittest.mock import patch
from src.stage11 import search
from tests.test_hybrid_search import visual, text


class SearchTests(unittest.TestCase):
    def test_comparison_uses_same_candidates_and_preserves_diagnostics(self):
        class Encoder:
            def encode_text(self, query):
                return [1]
        class Images:
            def search(self, *args, **kwargs):
                return [visual(5), visual(1)]
        class Texts:
            def search(self, *args, **kwargs):
                return dict(results=[text(0,4)], excluded_counts={'policy':1})
        with patch('src.stage11._identity', return_value='key'):
            result=search('backpack',encoder=Encoder(),visual_index=Images(),
                          text_index=Texts(),text_embedder=lambda q:[1])
        self.assertEqual(result['comparison']['visual'][0]['start_seconds'],4)
        self.assertEqual(result['comparison']['text'][0]['start_seconds'],0)
        self.assertEqual(result['results'][0]['start_seconds'],0)
        self.assertEqual(result['text_diagnostics']['excluded_counts']['policy'],1)

    def test_invalid_query_before_loading_models(self):
        for query in ('', ' '*4, 'x'*1001, None):
            with self.assertRaises(ValueError):
                search(query)

    def test_disabled_channels_do_not_load_models_or_query_indexes(self):
        class Texts:
            def search(self,*args,**kwargs):
                return {'results':[text(0,4)]}
        with patch('src.stage11._encoder',side_effect=AssertionError('disabled')):
            result=search('bag',visual_weight=0,text_index=Texts(),text_embedder=lambda q:[1])
        self.assertEqual(result['candidate_counts']['visual'],0)
        class Encoder:
            def encode_text(self,q): return [1]
        class Images:
            def search(self,*args,**kwargs): return [visual(1)]
        with patch('src.stage11._identity',return_value='key'), patch('src.stage11.embed_text',side_effect=AssertionError('disabled')):
            result=search('bag',text_weight=0,encoder=Encoder(),visual_index=Images())
        self.assertEqual(result['candidate_counts']['text'],0)
