import unittest

from src.stage12_workspace import (
    hash_password, new_session_token, normalize_email, parse_capture_start,
    session_token_hash, verify_password,
)


class WorkspaceSecurityTests(unittest.TestCase):
    def test_password_hashes_are_salted_and_verifiable(self):
        first = hash_password('correct horse battery staple')
        second = hash_password('correct horse battery staple')
        self.assertNotEqual(first, second)
        self.assertTrue(verify_password('correct horse battery staple', first))
        self.assertFalse(verify_password('wrong password', first))
        self.assertFalse(verify_password('correct horse battery staple', 'invalid'))

    def test_password_policy_and_email_normalization(self):
        with self.assertRaises(ValueError):
            hash_password('short')
        self.assertEqual(normalize_email('  Goodluck@Example.Test '), 'goodluck@example.test')
        with self.assertRaises(ValueError):
            normalize_email('not-an-email')

    def test_session_secret_is_opaque_and_only_hash_is_stored(self):
        token = new_session_token()
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(len(session_token_hash(token)), 64)
        self.assertNotEqual(token, session_token_hash(token))

    def test_capture_time_requires_unambiguous_real_local_time(self):
        utc, zone = parse_capture_start('2026-11-19T12:30', 'Africa/Lagos')
        self.assertEqual(zone, 'Africa/Lagos')
        self.assertEqual(utc.isoformat(), '2026-11-19T11:30:00+00:00')
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            parse_capture_start('2026-03-08T02:30', 'America/New_York')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            parse_capture_start('2026-11-01T01:30', 'America/New_York')


if __name__ == '__main__':
    unittest.main()
