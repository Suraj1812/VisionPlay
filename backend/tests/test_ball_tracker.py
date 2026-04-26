from __future__ import annotations

import unittest

import cv2
import numpy as np

from ai.perception.ball_tracker import SpatioTemporalBallTracker


class BallTrackerTests(unittest.TestCase):
    def test_predict_recovers_motion_candidate_without_yolo_ball(self) -> None:
        tracker = SpatioTemporalBallTracker(fps=25.0)
        frame_a = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_b = frame_a.copy()
        cv2.circle(frame_b, (960, 520), 8, (255, 255, 255), -1)

        first = tracker.predict(frame_a, yolo_ball_bbox=None)
        second = tracker.predict(frame_b, yolo_ball_bbox=None)

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreaterEqual(second[4], 0.4)
        self.assertTrue(920 <= second[0] <= 980)
        self.assertTrue(480 <= second[1] <= 540)


if __name__ == "__main__":
    unittest.main()
