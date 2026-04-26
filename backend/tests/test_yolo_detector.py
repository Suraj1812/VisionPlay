from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from ai.detection.yolo_detector import DetectedObject, YOLODetector

class YOLODetectorTests(unittest.TestCase):
    def test_build_tile_boxes_covers_frame_edges(self) -> None:
        tile_boxes = YOLODetector._build_tile_boxes(1920, 1080)

        self.assertTrue(tile_boxes)
        self.assertEqual(tile_boxes[0][:2], (0, 0))
        self.assertTrue(any(right == 1920 for _, _, right, _ in tile_boxes))
        self.assertTrue(any(bottom == 1080 for _, _, _, bottom in tile_boxes))

    def test_merge_detections_prefers_highest_confidence_overlap(self) -> None:
        merged = YOLODetector._merge_detections(
            [
                DetectedObject("sports ball", 32, [100, 100, 116, 116], 0.42),
                DetectedObject("sports ball", 32, [101, 101, 117, 117], 0.81),
                DetectedObject("person", 0, [300, 300, 420, 760], 0.9),
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].confidence, 0.9)
        self.assertTrue(any(item.object_type == "sports ball" and item.confidence == 0.81 for item in merged))

    def test_class_confidence_threshold_is_stricter_for_small_objects(self) -> None:
        detector = YOLODetector()
        detector.class_names_by_id = {0: "person", 32: "sports ball"}

        with patch.object(
            __import__("ai.detection.yolo_detector", fromlist=["settings"]).settings,
            "small_object_class_names",
            "sports ball",
        ), patch.object(
            __import__("ai.detection.yolo_detector", fromlist=["settings"]).settings,
            "small_object_track_confidence_threshold",
            0.36,
        ):
            self.assertGreater(
                detector._class_confidence_threshold("sports ball"),
                detector._class_confidence_threshold("person"),
            )

    def test_small_object_tile_pass_skips_non_stride_frame_when_full_frame_is_strong(self) -> None:
        detector = YOLODetector()
        detector.small_object_class_ids = [32]

        with patch("ai.detection.yolo_detector.settings.small_object_tile_frame_stride", 2):
            should_run = detector._should_run_small_object_tiles(
                frame_id=1,
                full_frame_detections=[
                    DetectedObject("sports ball", 32, [100, 100, 116, 116], 0.82)
                ],
            )

        self.assertFalse(should_run)

    def test_detect_small_object_tiles_batches_predict_calls(self) -> None:
        class FakeResult:
            def __init__(self) -> None:
                self.boxes = None
                self.names = {32: "sports ball"}

        class FakeModel:
            def __init__(self) -> None:
                self.predict_calls = []

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                return [FakeResult() for _ in kwargs["source"]]

        detector = YOLODetector()
        fake_model = FakeModel()
        detector.model = fake_model                            
        detector.class_names_by_id = {32: "sports ball"}
        detector.small_object_class_ids = [32]

        with patch("ai.detection.yolo_detector.settings.small_object_tile_grid_size", 3):
            detections = detector._detect_small_object_tiles(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertEqual(detections, [])
        self.assertEqual(len(fake_model.predict_calls), 1)
        self.assertIsInstance(fake_model.predict_calls[0]["source"], list)
        self.assertEqual(len(fake_model.predict_calls[0]["source"]), 9)

    def test_should_run_recall_boost_for_empty_low_light_frame(self) -> None:
        frame = np.full((240, 320, 3), 24, dtype=np.uint8)

        self.assertTrue(YOLODetector._should_run_recall_boost(frame, []))

    def test_prepare_recall_frame_preserves_shape(self) -> None:
        detector = YOLODetector()
        frame = np.full((180, 320, 3), 70, dtype=np.uint8)

        enhanced = detector._prepare_recall_frame(frame)

        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, frame.dtype)

    def test_build_region_proposals_targets_high_detail_region(self) -> None:
        detector = YOLODetector()
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        frame[140:320, 250:430] = 22
        frame[170:290, 280:400] = 235

        proposals = detector._build_region_proposals(frame, [])

        self.assertTrue(proposals)
        self.assertTrue(
            any(left <= 250 and top <= 140 and right >= 430 and bottom >= 320 for left, top, right, bottom in proposals)
        )

    def test_build_cricket_focus_boxes_cover_pitch_lane(self) -> None:
        focus_boxes = YOLODetector._build_cricket_focus_boxes(1920, 1080)

        self.assertEqual(len(focus_boxes), 3)
        self.assertTrue(any(left <= 420 and right >= 1500 for left, _, right, _ in focus_boxes))
        self.assertTrue(any(top <= 120 and bottom >= 900 for _, top, _, bottom in focus_boxes))
        self.assertTrue(any(740 <= left <= 780 and 1180 <= right <= 1220 for left, _, right, _ in focus_boxes))

    def test_actioncam_crease_cluster_detects_far_wicket_group(self) -> None:
        detections = [
            DetectedObject("person", 0, [920, 485, 948, 542], 0.62),
            DetectedObject("person", 0, [998, 458, 1037, 549], 0.41),
            DetectedObject("person", 0, [178, 492, 201, 546], 0.66),
        ]

        self.assertTrue(YOLODetector._is_actioncam_crease_cluster(detections, 1920, 1080))

    def test_build_actioncam_ball_boxes_targets_pitch_and_far_crease(self) -> None:
        tile_boxes = YOLODetector._build_actioncam_ball_boxes(1920, 1080)

        self.assertEqual(len(tile_boxes), 2)
        self.assertTrue(any(left <= 620 and right >= 1290 for left, _, right, _ in tile_boxes))
        self.assertTrue(any(top <= 350 and bottom >= 680 for _, top, _, bottom in tile_boxes))

    def test_should_run_actioncam_ball_tiles_requires_cluster_without_ball(self) -> None:
        detector = YOLODetector()
        detector.small_object_class_ids = [32]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [
            DetectedObject("person", 0, [920, 485, 948, 542], 0.62),
            DetectedObject("person", 0, [998, 458, 1037, 549], 0.41),
            DetectedObject("person", 0, [178, 492, 201, 546], 0.66),
        ]

        should_run = detector._should_run_actioncam_ball_tiles(12, frame, detections)

        self.assertTrue(should_run)

    def test_compute_appearance_signature_returns_normalized_vector(self) -> None:
        frame = np.full((120, 120, 3), 60, dtype=np.uint8)
        frame[20:96, 24:92] = (220, 80, 40)

        signature = YOLODetector._compute_appearance_signature(frame, [24, 20, 92, 96])

        self.assertIsNotNone(signature)
        assert signature is not None
        self.assertGreater(len(signature), 10)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in signature)), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
