from __future__ import annotations

import unittest

from backend.services.auxiliary_vision_service import (
    AuxiliaryFace,
    AuxiliaryHand,
    AuxiliaryVisionResult,
    AuxiliaryVisionService,
)

class AuxiliaryVisionServiceTests(unittest.TestCase):
    def test_classify_emotion_prefers_smiling(self) -> None:
        emotion = AuxiliaryVisionService._classify_emotion(
            {
                "mouthSmileLeft": 0.72,
                "mouthSmileRight": 0.69,
                "jawOpen": 0.08,
                "eyeBlinkLeft": 0.1,
                "eyeBlinkRight": 0.1,
            }
        )

        self.assertEqual(emotion, "smiling")

    def test_classify_emotion_detects_sleepy(self) -> None:
        emotion = AuxiliaryVisionService._classify_emotion(
            {
                "eyeBlinkLeft": 0.82,
                "eyeBlinkRight": 0.75,
                "mouthSmileLeft": 0.05,
                "mouthSmileRight": 0.04,
            }
        )

        self.assertEqual(emotion, "sleepy")

    def test_classify_emotion_detects_surprised(self) -> None:
        emotion = AuxiliaryVisionService._classify_emotion(
            {
                "jawOpen": 0.4,
                "browInnerUp": 0.2,
                "eyeWideLeft": 0.18,
                "eyeWideRight": 0.17,
            }
        )

        self.assertEqual(emotion, "surprised")

    def test_build_reactions_summarizes_face_and_hand_state(self) -> None:
        result = AuxiliaryVisionResult(
            faces=[
                AuxiliaryFace(
                    bbox=[100.0, 100.0, 200.0, 220.0],
                    confidence=0.91,
                    emotion="smiling",
                    smile_score=0.11,
                    eye_openness=0.93,
                    head_pose="turned_left",
                )
            ],
            hands=[
                AuxiliaryHand(
                    bbox=[220.0, 240.0, 340.0, 400.0],
                    confidence=0.84,
                    handedness="right",
                    gesture="Thumb_Up",
                )
            ],
        )

        reactions = AuxiliaryVisionService._build_reactions(result)

        self.assertIn("Looks smiling", reactions)
        self.assertIn("Looking left", reactions)
        self.assertIn("Gesture Thumb Up", reactions)

    def test_build_metrics_omits_unknown_gesture(self) -> None:
        result = AuxiliaryVisionResult(
            hands=[
                AuxiliaryHand(
                    bbox=[220.0, 240.0, 340.0, 400.0],
                    confidence=0.61,
                    handedness="right",
                    gesture="None",
                )
            ],
        )

        metrics = AuxiliaryVisionService._build_metrics(result)

        self.assertEqual(metrics["handedness"], "right")
        self.assertNotIn("gesture", metrics)

    def test_next_monotonic_timestamp_ms_never_regresses(self) -> None:
        service = AuxiliaryVisionService()

        first = service._next_monotonic_timestamp_ms(120)
        second = service._next_monotonic_timestamp_ms(80)
        third = service._next_monotonic_timestamp_ms(80)

        self.assertEqual(first, 120)
        self.assertEqual(second, 121)
        self.assertEqual(third, 122)

if __name__ == "__main__":
    unittest.main()
