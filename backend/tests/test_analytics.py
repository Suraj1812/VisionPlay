from __future__ import annotations

import unittest

from ai.pipeline.analytics import build_track_analytics

class AnalyticsTests(unittest.TestCase):
    def test_build_track_analytics_dedupes_same_frame_points(self) -> None:
        detections = [
            {
                "frame_id": 0,
                "timestamp_ms": 0,
                "object_type": "sports ball",
                "bbox": [10, 10, 20, 20],
                "confidence": 0.35,
                "tracking_id": 7,
            },
            {
                "frame_id": 0,
                "timestamp_ms": 0,
                "object_type": "sports ball",
                "bbox": [11, 11, 21, 21],
                "confidence": 0.95,
                "tracking_id": 7,
            },
            {
                "frame_id": 1,
                "timestamp_ms": 40,
                "object_type": "sports ball",
                "bbox": [14, 13, 24, 23],
                "confidence": 0.75,
                "tracking_id": 7,
            },
            {
                "frame_id": 2,
                "timestamp_ms": 80,
                "object_type": "sports ball",
                "bbox": [18, 15, 28, 25],
                "confidence": 0.78,
                "tracking_id": 7,
            },
            {
                "frame_id": 3,
                "timestamp_ms": 80,
                "object_type": "sports ball",
                "bbox": [5, 5, 5, 9],
                "confidence": 0.75,
                "tracking_id": 7,
            },
        ]

        tracks, summary = build_track_analytics(detections, fps=25.0)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]["path"]), 3)
        self.assertEqual(tracks[0]["path"][0]["x"], 16.0)
        self.assertEqual(summary["tracks_by_type"]["sports ball"], 1)
        self.assertEqual(summary["detections_by_type"]["sports ball"], 4)

    def test_build_track_analytics_discards_short_segment_after_teleport(self) -> None:
        detections = [
            {
                "frame_id": 0,
                "timestamp_ms": 0,
                "object_type": "person",
                "bbox": [100, 100, 180, 300],
                "confidence": 0.8,
                "tracking_id": 11,
            },
            {
                "frame_id": 1,
                "timestamp_ms": 40,
                "object_type": "person",
                "bbox": [108, 106, 188, 306],
                "confidence": 0.82,
                "tracking_id": 11,
            },
            {
                "frame_id": 2,
                "timestamp_ms": 80,
                "object_type": "person",
                "bbox": [900, 400, 980, 600],
                "confidence": 0.81,
                "tracking_id": 11,
            },
        ]

        tracks, summary = build_track_analytics(detections, fps=25.0)

        self.assertEqual(len(tracks), 0)
        self.assertNotIn("person", summary["tracks_by_type"])

    def test_build_track_analytics_rejects_low_confidence_small_object_track(self) -> None:
        detections = [
            {
                "frame_id": 0,
                "timestamp_ms": 0,
                "object_type": "sports ball",
                "bbox": [10, 10, 20, 20],
                "confidence": 0.45,
                "tracking_id": 3,
            },
            {
                "frame_id": 1,
                "timestamp_ms": 40,
                "object_type": "sports ball",
                "bbox": [12, 10, 22, 20],
                "confidence": 0.47,
                "tracking_id": 3,
            },
            {
                "frame_id": 2,
                "timestamp_ms": 80,
                "object_type": "sports ball",
                "bbox": [14, 10, 24, 20],
                "confidence": 0.46,
                "tracking_id": 3,
            },
        ]

        tracks, summary = build_track_analytics(detections, fps=25.0)

        self.assertEqual(tracks, [])
        self.assertNotIn("sports ball", summary["tracks_by_type"])

if __name__ == "__main__":
    unittest.main()
