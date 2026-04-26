from __future__ import annotations

import unittest

import cv2
import numpy as np

from ai.cricket.profile_analyzer import CricketClipProfileAnalyzer


class CricketClipProfileAnalyzerTests(unittest.TestCase):
    def test_build_report_identifies_end_on_cricket_profile(self) -> None:
        analyzer = CricketClipProfileAnalyzer()
        frame = np.full((1080, 1920, 3), (120, 165, 195), dtype=np.uint8)
        cv2.rectangle(frame, (760, 240), (1160, 1040), (150, 185, 205), -1)
        detections = [
            {"bbox": [910, 450, 945, 545], "tracking_id": 1},
            {"bbox": [985, 460, 1022, 548], "tracking_id": 2},
            {"bbox": [760, 680, 948, 1008], "tracking_id": 3},
        ]

        for _ in range(6):
            analyzer.observe(frame, detections, scene_cut=False)

        report = analyzer.build_report()
        self.assertEqual(report.profile, "cricket_end_on_action_cam_v1")
        self.assertTrue(report.specialized)
        self.assertGreater(report.confidence, 0.58)


if __name__ == "__main__":
    unittest.main()
