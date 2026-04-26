from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ai.tracking.byte_tracker import TrackedObject, UniversalByteTracker
from backend.services.auxiliary_vision_service import AuxiliaryFace, AuxiliaryHand, AuxiliaryVisionResult
from backend.services.live_detection_service import LiveDetectionService, LiveDetectionSession

class LiveDetectionServiceTests(unittest.TestCase):
    def test_select_focus_item_prefers_centered_detection(self) -> None:
        objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[260, 80, 520, 700],
                confidence=0.81,
                tracking_id=4,
            ),
            TrackedObject(
                object_type="car",
                class_id=2,
                bbox=[10, 420, 160, 620],
                confidence=0.89,
                tracking_id=8,
            ),
        ]

        focus_item = LiveDetectionService._select_focus_item(objects, frame_width=960, frame_height=720)

        self.assertIsNotNone(focus_item)
        self.assertEqual(focus_item.object_type, "person")
        self.assertEqual(focus_item.tracking_id, 4)

    def test_build_reactions_reports_focus_counts_and_lighting(self) -> None:
        focus_item = TrackedObject(
            object_type="person",
            class_id=0,
            bbox=[260, 80, 520, 700],
            confidence=0.81,
            tracking_id=4,
        )
        objects = [
            focus_item,
            TrackedObject(
                object_type="sports ball",
                class_id=32,
                bbox=[540, 320, 560, 340],
                confidence=0.77,
                tracking_id=12,
            ),
        ]

        reactions = LiveDetectionService._build_reactions(
            objects=objects,
            object_counts={"person": 2, "sports ball": 1},
            focus_item=focus_item,
            lighting="low",
            face_reaction=None,
            frame_width=960,
            frame_height=720,
            auxiliary_reactions=[],
        )

        self.assertIn("3 stable objects detected", reactions)
        self.assertIn("Focus locked on Person #4", reactions)
        self.assertIn("Low light scene", reactions)

    def test_recover_handheld_item_from_closeup_person_frame_with_hand_support(self) -> None:
        frame = np.full((900, 1400, 3), 210, dtype=np.uint8)
        frame[280:640, 500:700] = 25
        objects = [
            TrackedObject(object_type="person", class_id=0, bbox=[250, 120, 1080, 760], confidence=0.88, tracking_id=1),
            TrackedObject(
                object_type=LiveDetectionService.HAND_LABEL,
                class_id=-2,
                bbox=[520, 320, 700, 610],
                confidence=0.83,
                tracking_id=None,
            ),
        ]

        recovered = LiveDetectionService()._recover_handheld_item(
            frame,
            objects,
            auxiliary_result=AuxiliaryVisionResult(
                hands=[
                    AuxiliaryHand(
                        bbox=[520, 320, 700, 610],
                        confidence=0.83,
                        handedness="right",
                        gesture=None,
                    )
                ]
            ),
        )

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.object_type, LiveDetectionService.PHONE_LABEL)
        self.assertGreater(recovered.confidence, 0.5)

    def test_focus_prefers_handheld_item_when_present(self) -> None:
        objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[220, 60, 760, 700],
                confidence=0.9,
                tracking_id=4,
            ),
            TrackedObject(
                object_type=LiveDetectionService.HANDHELD_LABEL,
                class_id=-1,
                bbox=[280, 180, 720, 520],
                confidence=0.77,
                tracking_id=None,
            ),
        ]

        focus_item = LiveDetectionService._select_focus_item(objects, frame_width=960, frame_height=720)

        self.assertIsNotNone(focus_item)
        self.assertEqual(focus_item.object_type, LiveDetectionService.HANDHELD_LABEL)

    def test_recover_handheld_item_without_hand_support_returns_none(self) -> None:
        frame = np.full((720, 1280, 3), 205, dtype=np.uint8)
        frame[180:520, 360:900] = 18
        objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[220, 90, 980, 690],
                confidence=0.9,
                tracking_id=7,
            )
        ]

        recovered = LiveDetectionService()._recover_handheld_item(
            frame,
            objects,
            auxiliary_result=AuxiliaryVisionResult(),
        )

        self.assertIsNone(recovered)

    def test_recover_handheld_item_rejects_large_background_rectangle(self) -> None:
        frame = np.full((900, 1400, 3), 214, dtype=np.uint8)
        frame[360:860, 520:1220] = 24
        objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[260, 120, 1080, 840],
                confidence=0.87,
                tracking_id=2,
            ),
            TrackedObject(
                object_type=LiveDetectionService.HAND_LABEL,
                class_id=-2,
                bbox=[540, 420, 780, 690],
                confidence=0.78,
                tracking_id=None,
            ),
        ]

        recovered = LiveDetectionService()._recover_handheld_item(
            frame,
            objects,
            auxiliary_result=AuxiliaryVisionResult(
                hands=[
                    AuxiliaryHand(
                        bbox=[540, 420, 780, 690],
                        confidence=0.78,
                        handedness="right",
                        gesture=None,
                    )
                ]
            ),
        )

        self.assertIsNone(recovered)

    def test_simplify_live_objects_drops_large_person_when_phone_like_device_present(self) -> None:
        service = LiveDetectionService()
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[110, 20, 1180, 700],
                confidence=0.89,
                tracking_id=1,
            ),
            TrackedObject(
                object_type=LiveDetectionService.PHONE_LABEL,
                class_id=-1,
                bbox=[360, 200, 860, 520],
                confidence=0.78,
                tracking_id=None,
            ),
            TrackedObject(
                object_type=LiveDetectionService.FACE_LABEL,
                class_id=-1,
                bbox=[760, 160, 980, 420],
                confidence=0.72,
                tracking_id=None,
            ),
        ]

        simplified = service._simplify_live_objects(frame, objects)

        self.assertEqual(
            [item.object_type for item in simplified],
            [LiveDetectionService.PHONE_LABEL, LiveDetectionService.FACE_LABEL],
        )

    def test_build_reactions_reports_face_and_phone_like_device(self) -> None:
        focus_item = TrackedObject(
            object_type=LiveDetectionService.PHONE_LABEL,
            class_id=-1,
            bbox=[280, 180, 720, 520],
            confidence=0.77,
            tracking_id=None,
        )
        reactions = LiveDetectionService._build_reactions(
            objects=[focus_item],
            object_counts={LiveDetectionService.PHONE_LABEL: 1, LiveDetectionService.FACE_LABEL: 1},
            focus_item=focus_item,
            lighting="balanced",
            face_reaction="Smiling",
            frame_width=960,
            frame_height=720,
            auxiliary_reactions=[],
        )

        self.assertIn("Phone visible", reactions)
        self.assertIn("Face looks smiling", reactions)

    def test_normalize_live_labels_only_maps_cell_phone_to_phone_like_device(self) -> None:
        normalized = LiveDetectionService()._normalize_live_labels(
            [
                TrackedObject(
                    object_type="cell phone",
                    class_id=65,
                    bbox=[260, 180, 420, 420],
                    confidence=0.61,
                    tracking_id=3,
                ),
                TrackedObject(
                    object_type="remote",
                    class_id=66,
                    bbox=[160, 140, 260, 300],
                    confidence=0.58,
                    tracking_id=4,
                )
            ]
        )

        self.assertEqual(normalized[0].object_type, LiveDetectionService.PHONE_LABEL)
        self.assertEqual(normalized[1].object_type, "remote")

    def test_stabilize_derived_objects_reuses_face_tracking_id(self) -> None:
        service = LiveDetectionService()
        session = LiveDetectionSession(tracker=UniversalByteTracker(frame_rate=12), frame_index=1)
        first_frame = [
            TrackedObject(
                object_type=LiveDetectionService.FACE_LABEL,
                class_id=-1,
                bbox=[180, 120, 320, 320],
                confidence=0.84,
                tracking_id=None,
            )
        ]

        stabilized_first = service._stabilize_derived_objects(session, first_frame, frame_width=640, frame_height=480)
        session.frame_index = 2
        second_frame = [
            TrackedObject(
                object_type=LiveDetectionService.FACE_LABEL,
                class_id=-1,
                bbox=[188, 124, 328, 324],
                confidence=0.88,
                tracking_id=None,
            )
        ]
        stabilized_second = service._stabilize_derived_objects(session, second_frame, frame_width=640, frame_height=480)

        self.assertEqual(len(stabilized_first), 1)
        self.assertEqual(len(stabilized_second), 1)
        self.assertIsNotNone(stabilized_first[0].tracking_id)
        self.assertEqual(stabilized_first[0].tracking_id, stabilized_second[0].tracking_id)

    def test_stabilize_derived_objects_keeps_recent_phone_track_as_prediction(self) -> None:
        service = LiveDetectionService()
        session = LiveDetectionSession(tracker=UniversalByteTracker(frame_rate=12), frame_index=1)
        observed_phone = [
            TrackedObject(
                object_type=LiveDetectionService.PHONE_LABEL,
                class_id=-1,
                bbox=[220, 160, 420, 420],
                confidence=0.82,
                tracking_id=None,
            )
        ]

        service._stabilize_derived_objects(session, observed_phone, frame_width=640, frame_height=480)
        session.frame_index = 2
        service._stabilize_derived_objects(session, observed_phone, frame_width=640, frame_height=480)
        session.frame_index = 3
        predicted_only = service._stabilize_derived_objects(session, [], frame_width=640, frame_height=480)

        self.assertEqual(len(predicted_only), 1)
        self.assertEqual(predicted_only[0].object_type, LiveDetectionService.PHONE_LABEL)
        self.assertTrue(predicted_only[0].is_predicted)
        self.assertIsNotNone(predicted_only[0].tracking_id)

    def test_simplify_live_objects_drops_hand_when_it_covers_face_region(self) -> None:
        service = LiveDetectionService()
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        objects = [
            TrackedObject(
                object_type=LiveDetectionService.PHONE_LABEL,
                class_id=-1,
                bbox=[520, 180, 790, 520],
                confidence=0.8,
                tracking_id=100,
            ),
            TrackedObject(
                object_type=LiveDetectionService.FACE_LABEL,
                class_id=-1,
                bbox=[650, 260, 910, 560],
                confidence=0.76,
                tracking_id=101,
            ),
            TrackedObject(
                object_type=LiveDetectionService.HAND_LABEL,
                class_id=-2,
                bbox=[470, 170, 980, 610],
                confidence=0.81,
                tracking_id=102,
            ),
        ]

        simplified = service._simplify_live_objects(frame, objects)

        self.assertEqual(
            [item.object_type for item in simplified],
            [LiveDetectionService.PHONE_LABEL, LiveDetectionService.FACE_LABEL],
        )

    def test_simplify_live_objects_drops_hand_when_it_overlaps_phone(self) -> None:
        service = LiveDetectionService()
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        objects = [
            TrackedObject(
                object_type=LiveDetectionService.PHONE_LABEL,
                class_id=-1,
                bbox=[500, 180, 820, 520],
                confidence=0.83,
                tracking_id=100,
            ),
            TrackedObject(
                object_type=LiveDetectionService.HAND_LABEL,
                class_id=-2,
                bbox=[420, 230, 840, 600],
                confidence=0.9,
                tracking_id=101,
            ),
        ]

        simplified = service._simplify_live_objects(frame, objects)

        self.assertEqual([item.object_type for item in simplified], [LiveDetectionService.PHONE_LABEL])

    def test_simplify_live_objects_keeps_hand_with_meaningful_gesture_near_phone(self) -> None:
        service = LiveDetectionService()
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        hand = TrackedObject(
            object_type=LiveDetectionService.HAND_LABEL,
            class_id=-2,
            bbox=[430, 230, 820, 590],
            confidence=0.68,
            tracking_id=101,
        )
        objects = [
            TrackedObject(
                object_type=LiveDetectionService.PHONE_LABEL,
                class_id=-1,
                bbox=[500, 180, 820, 520],
                confidence=0.83,
                tracking_id=100,
            ),
            hand,
        ]

        from backend.services.auxiliary_vision_service import AuxiliaryHand, AuxiliaryVisionResult

        simplified = service._simplify_live_objects(
            frame,
            objects,
            auxiliary_result=AuxiliaryVisionResult(
                hands=[
                    AuxiliaryHand(
                        bbox=list(hand.bbox),
                        confidence=0.68,
                        handedness="right",
                        gesture="Thumb_Up",
                        gesture_confidence=0.68,
                    )
                ]
            ),
        )

        self.assertEqual(
            [item.object_type for item in simplified],
            [LiveDetectionService.PHONE_LABEL, LiveDetectionService.HAND_LABEL],
        )

    def test_get_auxiliary_result_reuses_recent_result_between_stride_frames(self) -> None:
        service = LiveDetectionService()
        session = LiveDetectionSession(tracker=UniversalByteTracker(frame_rate=12))
        session.frame_index = 1
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        first_result = AuxiliaryVisionResult(
            faces=[
                AuxiliaryFace(
                    bbox=[180.0, 120.0, 320.0, 320.0],
                    confidence=0.86,
                    emotion="neutral",
                    smile_score=0.08,
                    eye_openness=0.92,
                    head_pose="forward",
                )
            ],
            metrics={"emotion": "neutral"},
        )

        with patch(
            "backend.services.live_detection_service.auxiliary_vision_service.analyze_frame",
            return_value=first_result,
        ) as analyze_frame:
            result_one = service._get_auxiliary_result(session, frame)
            session.frame_index = 2
            result_two = service._get_auxiliary_result(session, frame)

        self.assertIs(result_one, first_result)
        self.assertIs(result_two, first_result)
        self.assertEqual(analyze_frame.call_count, 1)

if __name__ == "__main__":
    unittest.main()
