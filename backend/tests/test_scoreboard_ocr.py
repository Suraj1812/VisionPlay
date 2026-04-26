from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ai.cricket.scoreboard_ocr import ScoreboardOCR


class ScoreboardOCRTests(unittest.TestCase):
    def test_extract_skips_when_overlay_is_missing(self) -> None:
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        ocr = ScoreboardOCR(ocr_interval=1)

        class FakeReader:
            def readtext(self, *_args, **_kwargs):
                raise AssertionError("OCR should not be called when overlay is missing")

        with patch("ai.cricket.scoreboard_ocr._HAS_OCR", True):
            ocr._reader = FakeReader()
            result = ocr.extract(frame, frame_id=0, fps=25.0)

        self.assertFalse(result.detected)
        self.assertEqual(ocr.overlay_detected_ratio, 0.0)

    def test_extract_enters_cooldown_after_repeated_misses_without_prior_score(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ocr = ScoreboardOCR(ocr_interval=1)

        class FakeReader:
            def readtext(self, *_args, **_kwargs):
                return []

        with patch("ai.cricket.scoreboard_ocr._HAS_OCR", True):
            ocr._reader = FakeReader()
            with patch.object(ScoreboardOCR, "_overlay_present", return_value=True):
                for _ in range(4):
                    ocr.extract(frame, frame_id=0, fps=25.0)

        self.assertGreater(ocr._cooldown_calls, 0)


if __name__ == "__main__":
    unittest.main()
