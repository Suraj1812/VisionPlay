from __future__ import annotations

import unittest

from ai.tracking.byte_tracker import CricketByteTracker, TrackedObject

class CricketByteTrackerTests(unittest.TestCase):
    def test_match_existing_track_prefers_nearby_same_scale_detection(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            7,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.8,
                tracking_id=7,
            ),
        )

        tracker.frame_index = 2
        matched_track_id = tracker._match_existing_track(
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[112, 126, 192, 326],
                confidence=0.79,
                tracking_id=None,
            ),
            used_track_ids=set(),
        )

        self.assertEqual(matched_track_id, 7)

    def test_match_existing_track_rejects_teleport_detection(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            9,
            TrackedObject(
                object_type="sports ball",
                class_id=32,
                bbox=[300, 260, 312, 272],
                confidence=0.7,
                tracking_id=9,
            ),
        )

        tracker.frame_index = 2
        matched_track_id = tracker._match_existing_track(
            TrackedObject(
                object_type="sports ball",
                class_id=32,
                bbox=[1200, 700, 1212, 712],
                confidence=0.71,
                tracking_id=None,
            ),
            used_track_ids=set(),
        )

        self.assertIsNone(matched_track_id)

    def test_inconsistent_tracker_assignment_is_rejected(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            12,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.92,
                tracking_id=12,
            ),
        )

        tracker.frame_index = 2
        self.assertFalse(
            tracker._is_tracker_assignment_consistent(
                12,
                TrackedObject(
                    object_type="person",
                    class_id=0,
                    bbox=[1200, 700, 1280, 900],
                    confidence=0.93,
                    tracking_id=12,
                ),
            )
        )

    def test_predict_only_projects_active_track_forward(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            21,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.9,
                tracking_id=21,
            ),
        )
        tracker.frame_index = 2
        tracker._update_track_state(
            21,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[110, 130, 190, 330],
                confidence=0.88,
                tracking_id=21,
            ),
        )

        predicted = tracker.predict_only()

        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].tracking_id, 21)
        self.assertGreater(predicted[0].bbox[0], 110)
        self.assertGreater(predicted[0].bbox[1], 130)

    def test_has_active_tracks_is_false_after_reset(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            31,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.9,
                tracking_id=31,
            ),
        )

        self.assertTrue(tracker.has_active_tracks())
        tracker.reset()
        self.assertFalse(tracker.has_active_tracks())

    def test_match_existing_track_prefers_appearance_consistent_candidate(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            41,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.9,
                tracking_id=41,
                appearance_signature=(1.0, 0.0, 0.0),
            ),
        )
        tracker._update_track_state(
            42,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[130, 120, 210, 320],
                confidence=0.9,
                tracking_id=42,
                appearance_signature=(0.0, 1.0, 0.0),
            ),
        )

        tracker.frame_index = 2
        matched_track_id = tracker._match_existing_track(
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[128, 124, 208, 324],
                confidence=0.88,
                tracking_id=None,
                appearance_signature=(0.0, 1.0, 0.0),
            ),
            used_track_ids=set(),
        )

        self.assertEqual(matched_track_id, 42)

    def test_inconsistent_assignment_is_rejected_when_appearance_changes_too_much(self) -> None:
        tracker = CricketByteTracker(frame_rate=25.0)
        tracker.frame_index = 1
        tracker._update_track_state(
            51,
            TrackedObject(
                object_type="person",
                class_id=0,
                bbox=[100, 120, 180, 320],
                confidence=0.9,
                tracking_id=51,
                appearance_signature=(1.0, 0.0, 0.0),
            ),
        )

        tracker.frame_index = 2
        self.assertFalse(
            tracker._is_tracker_assignment_consistent(
                51,
                TrackedObject(
                    object_type="person",
                    class_id=0,
                    bbox=[108, 124, 188, 324],
                    confidence=0.88,
                    tracking_id=51,
                    appearance_signature=(0.0, 1.0, 0.0),
                ),
            )
        )

if __name__ == "__main__":
    unittest.main()
