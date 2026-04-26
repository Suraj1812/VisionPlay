from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from ai.cricket.scoreboard_ocr import ScoreboardData
from ai.pipeline.video_processor import VideoProcessor
from ai.tracking.byte_tracker import TrackedObject

class VideoProcessorTests(unittest.TestCase):
    def test_analyze_live_frame_skips_detector_between_live_stride_frames(self) -> None:
        class FakeTracker:
            def __init__(self) -> None:
                self.frame_rate = 12

            def has_active_tracks(self) -> bool:
                return True

            def update(self, _detections):
                return [
                    TrackedObject(
                        object_type="person",
                        class_id=0,
                        bbox=[200, 120, 360, 620],
                        confidence=0.92,
                        tracking_id=1,
                    )
                ]

            def predict_only(self):
                return [
                    TrackedObject(
                        object_type="person",
                        class_id=0,
                        bbox=[206, 122, 366, 622],
                        confidence=0.9,
                        tracking_id=1,
                        is_predicted=True,
                    )
                ]

        frame = np.full((720, 1280, 3), 160, dtype=np.uint8)
        processor = VideoProcessor()
        processor.detector = Mock()
        processor.detector.detect.return_value = []
        tracker = FakeTracker()
        track_hits: dict[tuple[str, int], int] = {}

        processor.analyze_live_frame(frame, tracker, frame_id=1, track_hits=track_hits)
        predicted_frame = processor.analyze_live_frame(frame, tracker, frame_id=3, track_hits=track_hits)

        self.assertEqual(processor.detector.detect.call_count, 1)
        self.assertEqual(len(predicted_frame), 1)
        self.assertTrue(predicted_frame[0].is_predicted)

    def test_normalize_tracked_objects_rejects_overlay_regions_and_tiny_noise(self) -> None:
        frame = np.full((1080, 1920, 3), (60, 140, 40), dtype=np.uint8)
        frame[900:, :] = (230, 230, 230)
        tracked_objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[80, 935, 250, 1075],
                confidence=0.91,
                tracking_id=1,
            ),
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[840, 420, 980, 900],
                confidence=0.88,
                tracking_id=2,
            ),
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 106, 126],
                confidence=0.95,
                tracking_id=3,
            ),
            TrackedObject(
                object_type="sports ball",
                class_id=32,
                bbox=[1200, 380, 1214, 394],
                confidence=0.85,
                tracking_id=4,
            ),
        ]

        normalized = VideoProcessor._normalize_tracked_objects(
            frame=frame,
            frame_width=1920,
            frame_height=1080,
            tracked_objects=tracked_objects,
        )

        self.assertEqual(
            [(item.object_type, item.tracking_id) for item in normalized],
            [("person", 2), ("sports ball", 4)],
        )

    def test_filter_confirmed_tracks_waits_for_stable_tracks(self) -> None:
        track_hits: dict[tuple[str, int], int] = {}
        with unittest.mock.patch(
            "ai.pipeline.video_processor.settings.small_object_class_names",
            "sports ball",
        ):
            first_pass = VideoProcessor._filter_confirmed_tracks(
                tracked_objects=[
                    TrackedObject(
                        object_type="person",
                        class_id=0,
                        bbox=[400, 300, 520, 760],
                        confidence=0.55,
                        tracking_id=11,
                    ),
                    TrackedObject(
                        object_type="sports ball",
                        class_id=32,
                        bbox=[990, 410, 1004, 424],
                        confidence=0.82,
                        tracking_id=22,
                    ),
                ],
                track_hits=track_hits,
            )
            second_pass = VideoProcessor._filter_confirmed_tracks(
                tracked_objects=[
                    TrackedObject(
                        object_type="sports ball",
                        class_id=32,
                        bbox=[992, 412, 1006, 426],
                        confidence=0.83,
                        tracking_id=22,
                    ),
                    TrackedObject(
                        object_type="person",
                        class_id=0,
                        bbox=[404, 302, 522, 758],
                        confidence=0.57,
                        tracking_id=11,
                    )
                ],
                track_hits=track_hits,
            )
            third_pass = VideoProcessor._filter_confirmed_tracks(
                tracked_objects=[
                    TrackedObject(
                        object_type="person",
                        class_id=0,
                        bbox=[408, 304, 524, 756],
                        confidence=0.58,
                        tracking_id=11,
                    )
                ],
                track_hits=track_hits,
            )

        self.assertEqual(first_pass, [])
        self.assertEqual([(item.object_type, item.tracking_id) for item in second_pass], [("sports ball", 22)])
        self.assertEqual([(item.object_type, item.tracking_id) for item in third_pass], [("person", 11)])

    def test_normalize_tracked_objects_rejects_low_confidence_edge_small_object(self) -> None:
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        tracked_objects = [
            TrackedObject(
                object_type="sports ball",
                class_id=32,
                bbox=[4, 120, 18, 134],
                confidence=0.46,
                tracking_id=7,
            )
        ]

        normalized = VideoProcessor._normalize_tracked_objects(
            frame=frame,
            frame_width=1920,
            frame_height=1080,
            tracked_objects=tracked_objects,
        )

        self.assertEqual(normalized, [])

    def test_scene_cut_detection_resets_on_large_visual_change(self) -> None:
        first_frame = np.full((360, 640, 3), (40, 120, 40), dtype=np.uint8)
        second_frame = np.full((360, 640, 3), (220, 220, 220), dtype=np.uint8)
        processor = VideoProcessor()

        self.assertFalse(processor.cut_detector.is_cut(first_frame))
        self.assertTrue(processor.cut_detector.is_cut(second_frame))

        processor.cut_detector.reset()
        self.assertFalse(processor.cut_detector.is_cut(first_frame.copy()))

    def test_normalize_tracked_objects_keeps_predicted_track_without_visual_support(self) -> None:
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        tracked_objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[420, 240, 520, 620],
                confidence=0.18,
                tracking_id=41,
                is_predicted=True,
            )
        ]

        normalized = VideoProcessor._normalize_tracked_objects(
            frame=frame,
            frame_width=1920,
            frame_height=1080,
            tracked_objects=tracked_objects,
        )

        self.assertEqual(len(normalized), 1)
        self.assertTrue(normalized[0].is_predicted)

    def test_filter_live_tracks_promotes_track_on_second_hit(self) -> None:
        track_hits: dict[tuple[str, int], int] = {}
        tracked_object = TrackedObject(
            object_type="person",
            class_id=0,
            bbox=[360, 220, 520, 760],
            confidence=0.58,
            tracking_id=51,
        )

        first_pass = VideoProcessor._filter_live_tracks([tracked_object], track_hits)
        second_pass = VideoProcessor._filter_live_tracks([tracked_object], track_hits)

        self.assertEqual(first_pass, [])
        self.assertEqual([(item.object_type, item.tracking_id) for item in second_pass], [("person", 51)])

    def test_live_profile_keeps_large_closeup_person(self) -> None:
        frame = np.full((1912, 2940, 3), 180, dtype=np.uint8)
        tracked_objects = [
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[1030.7, 433.5, 2479.3, 1477.5],
                confidence=0.871,
                tracking_id=1,
            )
        ]

        normalized = VideoProcessor._normalize_tracked_objects(
            frame=frame,
            frame_width=2940,
            frame_height=1912,
            tracked_objects=tracked_objects,
            profile="live",
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].object_type, "person")

    def test_retain_cricket_detections_drops_non_cricket_labels(self) -> None:
        detections = [
            TrackedObject(object_type="person", class_id=0, bbox=[0, 0, 10, 10], confidence=0.8, tracking_id=1),
            TrackedObject(object_type="sports ball", class_id=32, bbox=[0, 0, 10, 10], confidence=0.8, tracking_id=2),
            TrackedObject(object_type="boat", class_id=8, bbox=[0, 0, 10, 10], confidence=0.8, tracking_id=3),
            TrackedObject(object_type="baseball bat", class_id=34, bbox=[0, 0, 10, 10], confidence=0.8, tracking_id=4),
        ]

        filtered = VideoProcessor._retain_cricket_detections(detections)

        self.assertEqual(
            [(item.object_type, item.tracking_id) for item in filtered],
            [("person", 1), ("sports ball", 2), ("baseball bat", 4)],
        )

    def test_reset_batch_modules_clears_stale_cricket_state(self) -> None:
        processor = VideoProcessor()
        processor.cut_detector.total_cuts = 4
        processor.scoreboard_ocr._history.append(ScoreboardData(score="10/0", runs=10, wickets=0, overs="2.0", overs_float=2.0))
        processor.team_analyzer._players[1] = unittest.mock.Mock()
        processor.analytics._count = 9
        processor.event_classifier = unittest.mock.Mock()

        processor._reset_batch_modules(30.0)

        self.assertEqual(processor.cut_detector.total_cuts, 0)
        self.assertEqual(processor.scoreboard_ocr.get_score_timeline(), [])
        self.assertEqual(processor.team_analyzer.get_team_summary(), {})
        self.assertEqual(processor.analytics.build_full_analytics()["timeline"], [])
        self.assertIsNone(processor.event_classifier)
        self.assertEqual(processor.ball_tracker.fps, 30.0)

    def test_batch_stride_for_high_fps_video_targets_lower_analysis_rate(self) -> None:
        stride = VideoProcessor._batch_stride_for_fps(60.0, target_fps=15, minimum=1)

        self.assertEqual(stride, 4)

    def test_build_phase_entry_detects_actioncam_striker_ready(self) -> None:
        entry = VideoProcessor._build_phase_entry(
            team_results=[],
            person_detections=[
                {"bbox": [920, 485, 948, 542], "tracking_id": 1, "confidence": 0.62},
                {"bbox": [998, 458, 1037, 549], "tracking_id": 2, "confidence": 0.41},
                {"bbox": [178, 492, 201, 546], "tracking_id": 3, "confidence": 0.66},
            ],
            frame_width=1920,
            frame_height=1080,
            frame_id=0,
            fps=60.0,
        )

        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["phase_key"], "actioncam-striker-ready")
        self.assertIn("far crease", entry["phase_summary"].lower())

    def test_build_phase_entry_detects_actioncam_bowler_approach(self) -> None:
        entry = VideoProcessor._build_phase_entry(
            team_results=[],
            person_detections=[
                {"bbox": [920, 485, 948, 542], "tracking_id": 1, "confidence": 0.62},
                {"bbox": [998, 458, 1037, 549], "tracking_id": 2, "confidence": 0.41},
                {"bbox": [634, 281, 734, 636], "tracking_id": 4, "confidence": 0.8},
            ],
            frame_width=1920,
            frame_height=1080,
            frame_id=300,
            fps=60.0,
        )

        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["phase_key"], "actioncam-bowler-approach")
        self.assertIn("bowler movement", entry["phase_summary"].lower())

if __name__ == "__main__":
    unittest.main()
