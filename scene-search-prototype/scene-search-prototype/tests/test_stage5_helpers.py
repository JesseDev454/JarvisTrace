import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal_caption_test import make_messages


class TestMessages(unittest.TestCase):
    def test_chronological_image_order(self):
        frames = [{"timestamp_seconds": 0}, {"timestamp_seconds": 4}]
        images = [object(), object()]
        result = make_messages(frames, images)
        parts = result[0]["content"]
        self.assertEqual(parts[1]["text"], "At 0.00 seconds:")
        self.assertIs(parts[2]["image"], images[0])
        self.assertEqual(parts[3]["text"], "At 4.00 seconds:")
        self.assertIs(parts[4]["image"], images[1])
        self.assertIn("Do not invent", parts[-1]["text"])

    def test_mismatched_input_rejected(self):
        with self.assertRaises(ValueError):
            make_messages([{"timestamp_seconds": 0}], [])


if __name__ == "__main__":
    unittest.main()
