from __future__ import annotations

import unittest

import numpy as np

from ai.cricket.team_analyzer import TeamAnalyzer


class TeamAnalyzerTests(unittest.TestCase):
    def test_detect_camera_profile_identifies_end_on_actioncam(self) -> None:
        detections = [
            {"bbox": [908, 458, 944, 548], "tracking_id": 1},
            {"bbox": [994, 468, 1028, 552], "tracking_id": 2},
            {"bbox": [760, 684, 948, 1006], "tracking_id": 3},
        ]

        profile = TeamAnalyzer._detect_camera_profile(detections, 1920, 1080)

        self.assertEqual(profile, "action-cam-end-on")

    def test_infer_role_uses_actioncam_layout(self) -> None:
        self.assertEqual(
            TeamAnalyzer._infer_role(0.51, 0.47, 0.09, "action-cam-end-on"),
            "wicketkeeper",
        )
        self.assertEqual(
            TeamAnalyzer._infer_role(0.53, 0.56, 0.11, "action-cam-end-on"),
            "striker",
        )
        self.assertEqual(
            TeamAnalyzer._infer_role(0.44, 0.78, 0.24, "action-cam-end-on"),
            "bowler",
        )

    def test_analyze_exposes_actioncam_profile_in_summary(self) -> None:
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        detections = [
            {"bbox": [908, 458, 944, 548], "tracking_id": 1},
            {"bbox": [994, 468, 1028, 552], "tracking_id": 2},
            {"bbox": [760, 684, 948, 1006], "tracking_id": 3},
        ]
        analyzer = TeamAnalyzer()

        analyzer.analyze(frame, detections, frame_id=0)
        summary = analyzer.get_team_summary()

        self.assertEqual(summary.get("camera_profile"), "action-cam-end-on")


if __name__ == "__main__":
    unittest.main()
