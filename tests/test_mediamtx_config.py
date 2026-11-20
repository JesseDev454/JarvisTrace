from pathlib import Path
import unittest

import yaml


class MediaMTXConfigTests(unittest.TestCase):
    def test_enabled_media_listeners_are_loopback_only(self):
        config_path = Path(__file__).parents[1] / "config" / "mediamtx" / "mediamtx.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["rtsp"])
        self.assertEqual(config["rtspAddress"].split(":", 1)[0], "127.0.0.1")
        self.assertEqual(config["rtspTransports"], ["tcp"])
        self.assertFalse(config["rtmp"])
        self.assertEqual(config["hlsAddress"].split(":", 1)[0], "127.0.0.1")
        self.assertEqual(config["webrtcAddress"].split(":", 1)[0], "127.0.0.1")
        self.assertEqual(config["webrtcLocalUDPAddress"].split(":", 1)[0], "127.0.0.1")
        self.assertFalse(config["webrtcIPsFromInterfaces"])
        self.assertEqual(config["webrtcAdditionalHosts"], ["127.0.0.1"])
        self.assertFalse(config["srt"])
        self.assertFalse(config["moq"])


if __name__ == "__main__":
    unittest.main()
