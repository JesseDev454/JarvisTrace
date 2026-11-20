from unittest.mock import patch
import unittest
from src.stage13_preview import create_preview


class PreviewTests(unittest.TestCase):
    def test_rejects_public_before_loading_models(self):
        with self.assertRaises(ValueError):create_preview('public',[])

    def test_reuses_existing_app_with_isolated_indexes(self):
        with patch('src.stage13_preview.require_local_tables') as guard, \
             patch('src.stage13_preview.open_database') as database, \
             patch('src.stage13_preview.PlaybackCatalog') as catalog, \
             patch('src.stage13_preview.create_app') as app:
            database.return_value.__enter__.return_value.execute.return_value.fetchall.return_value=[]
            create_preview('stage13_example',['root'])
        guard.assert_called_once_with('stage13_example')
        database.assert_called_once_with(schema='stage13_example',readonly=True)
        self.assertEqual(app.call_args.kwargs['search_fn'].keywords['visual_index'].schema,'stage13_example')
