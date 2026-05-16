import unittest

from transvideo import seconds_to_hms


class TestSecondsToHms(unittest.TestCase):
    def test_seconds_to_hms(self):
        # Test cases for seconds_to_hms
        self.assertEqual(seconds_to_hms(0), "00:00:00,000")
        self.assertEqual(seconds_to_hms(1), "00:00:01,000")
        self.assertEqual(seconds_to_hms(61), "00:01:01,000")
        self.assertEqual(seconds_to_hms(3661), "01:01:01,000")
        self.assertEqual(seconds_to_hms(3661.123), "01:01:01,123")
        self.assertEqual(seconds_to_hms(86399.999), "23:59:59,999")


if __name__ == "__main__":
    unittest.main()
