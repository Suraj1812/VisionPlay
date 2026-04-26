from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.database.entities import Detection, Track, Video, VideoStatus
from backend.services.video_service import VideoService

class VideoServiceNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_service = VideoService(db=None)                          

    def test_normalize_processing_result_filters_invalid_records(self) -> None:
        processing_result = {
            "fps": 30,
            "frame_count": 120,
            "width": 1920,
            "height": 1080,
            "detections": [
                {
                    "frame_id": 2,
                    "timestamp_ms": 80,
                    "object_type": "sports ball",
                    "bbox": [200, 100, 190, 90],
                    "confidence": 1.4,
                    "tracking_id": "5",
                },
                {
                    "frame_id": 3,
                    "timestamp_ms": 120,
                    "object_type": "",
                    "bbox": [0, 0, 50, 50],
                    "confidence": 0.5,
                    "tracking_id": None,
                },
                {
                    "frame_id": 4,
                    "timestamp_ms": 160,
                    "object_type": "person",
                    "bbox": [20, 20, 22, 22],
                    "confidence": 0.5,
                    "tracking_id": 9,
                },
            ],
            "tracks": [
                {
                    "tracking_id": 5,
                    "object_type": "sports ball",
                    "path": [
                        {"frame_id": 2, "timestamp_ms": 80, "x": 195, "y": 95, "confidence": 0.9, "scale": 10},
                        {"frame_id": 2, "timestamp_ms": 80, "x": 196, "y": 96, "confidence": 0.8, "scale": 10},
                        {"frame_id": 3, "timestamp_ms": 120, "x": 210, "y": 108, "confidence": 0.88, "scale": 11},
                        {"frame_id": 4, "timestamp_ms": 160, "x": 224, "y": 121, "confidence": 0.87, "scale": 11},
                    ],
                    "distance_px": 9999,
                    "avg_speed_px_s": 9999,
                    "max_speed_px_s": 9999,
                },
                {
                    "tracking_id": -1,
                    "object_type": "person",
                    "path": [],
                    "distance_px": 0,
                    "avg_speed_px_s": 0,
                    "max_speed_px_s": 0,
                },
            ],
            "summary": {},
        }

        normalized = self.video_service._normalize_processing_result(processing_result)

        self.assertEqual(len(normalized["detections"]), 1)
        self.assertEqual(normalized["detections"][0]["bbox"], [190.0, 90.0, 200.0, 100.0])
        self.assertEqual(normalized["detections"][0]["confidence"], 1.0)
        self.assertEqual(len(normalized["tracks"]), 1)
        self.assertEqual(len(normalized["tracks"][0]["path"]), 3)
        self.assertEqual(normalized["summary"]["tracks_by_type"]["sports ball"], 1)
        self.assertEqual(normalized["summary"]["events"][0]["event_type"], "entry")

    def test_truncate_error_message_limits_length(self) -> None:
        message = "failure " * 200
        truncated = self.video_service._truncate_error_message(message)

        self.assertLessEqual(len(truncated), 500)
        self.assertTrue(truncated.endswith("..."))

    def test_normalize_processing_result_recomputes_summary_metrics(self) -> None:
        processing_result = {
            "fps": 10,
            "frame_count": 3,
            "width": 1920,
            "height": 1080,
            "detections": [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0,
                    "object_type": "sports ball",
                    "bbox": [10, 10, 20, 20],
                    "confidence": 0.9,
                    "tracking_id": 1,
                },
                {
                    "frame_id": 1,
                    "timestamp_ms": 100,
                    "object_type": "sports ball",
                    "bbox": [20, 10, 30, 20],
                    "confidence": 0.9,
                    "tracking_id": 1,
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 200,
                    "object_type": "sports ball",
                    "bbox": [40, 10, 50, 20],
                    "confidence": 0.9,
                    "tracking_id": 1,
                },
                {
                    "frame_id": 0,
                    "timestamp_ms": 0,
                    "object_type": "sports ball",
                    "bbox": [100, 100, 110, 110],
                    "confidence": 0.7,
                    "tracking_id": 2,
                },
                {
                    "frame_id": 1,
                    "timestamp_ms": 100,
                    "object_type": "sports ball",
                    "bbox": [200, 100, 210, 110],
                    "confidence": 0.7,
                    "tracking_id": 2,
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 200,
                    "object_type": "sports ball",
                    "bbox": [300, 100, 310, 110],
                    "confidence": 0.7,
                    "tracking_id": 2,
                },
                {
                    "frame_id": 0,
                    "timestamp_ms": 0,
                    "object_type": "person",
                    "bbox": [300, 300, 340, 340],
                    "confidence": 0.8,
                    "tracking_id": 8,
                },
                {
                    "frame_id": 1,
                    "timestamp_ms": 100,
                    "object_type": "person",
                    "bbox": [300, 330, 340, 370],
                    "confidence": 0.8,
                    "tracking_id": 8,
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 200,
                    "object_type": "person",
                    "bbox": [300, 360, 340, 400],
                    "confidence": 0.8,
                    "tracking_id": 8,
                },
            ],
            "tracks": [
                {
                    "tracking_id": 1,
                    "object_type": "sports ball",
                    "path": [
                        {"frame_id": 0, "timestamp_ms": 0, "x": 0, "y": 0, "confidence": 0.9, "scale": 10},
                        {"frame_id": 1, "timestamp_ms": 100, "x": 10, "y": 0, "confidence": 0.9, "scale": 10},
                        {"frame_id": 2, "timestamp_ms": 200, "x": 30, "y": 0, "confidence": 0.9, "scale": 10},
                    ],
                },
                {
                    "tracking_id": 2,
                    "object_type": "sports ball",
                    "path": [
                        {"frame_id": 0, "timestamp_ms": 0, "x": 0, "y": 0, "confidence": 0.7, "scale": 10},
                        {"frame_id": 1, "timestamp_ms": 100, "x": 100, "y": 0, "confidence": 0.7, "scale": 10},
                        {"frame_id": 2, "timestamp_ms": 200, "x": 200, "y": 0, "confidence": 0.7, "scale": 10},
                    ],
                },
                {
                    "tracking_id": 8,
                    "object_type": "person",
                    "path": [
                        {"frame_id": 0, "timestamp_ms": 0, "x": 0, "y": 0, "confidence": 0.8, "scale": 40},
                        {"frame_id": 1, "timestamp_ms": 100, "x": 0, "y": 30, "confidence": 0.8, "scale": 40},
                        {"frame_id": 2, "timestamp_ms": 200, "x": 0, "y": 60, "confidence": 0.8, "scale": 40},
                    ],
                },
            ],
            "summary": {
                "peak_speed_px_s": 1,
                "tracks_by_type": {"person": 99},
                "frame_errors": "5",
            },
        }

        normalized = self.video_service._normalize_processing_result(processing_result)

        self.assertEqual(normalized["summary"]["peak_speed_px_s"], 1000.0)
        self.assertEqual(normalized["summary"]["peak_speed_track_id"], 2)
        self.assertEqual(normalized["summary"]["tracks_by_type"]["sports ball"], 2)
        self.assertEqual(normalized["summary"]["tracks_by_type"]["person"], 1)
        self.assertEqual(normalized["summary"]["frame_errors"], 5)
        self.assertTrue(normalized["summary"]["primary_tracks"])

    def test_normalize_processing_result_repairs_track_timestamps_and_clamps_events(self) -> None:
        processing_result = {
            "fps": 10,
            "frame_count": 3,
            "width": 1280,
            "height": 720,
            "detections": [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0,
                    "object_type": "person",
                    "bbox": [100, 100, 160, 280],
                    "confidence": 0.82,
                    "tracking_id": 1,
                },
                {
                    "frame_id": 1,
                    "timestamp_ms": 100,
                    "object_type": "person",
                    "bbox": [104, 102, 164, 282],
                    "confidence": 0.8,
                    "tracking_id": 1,
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 200,
                    "object_type": "person",
                    "bbox": [108, 104, 168, 284],
                    "confidence": 0.79,
                    "tracking_id": 1,
                },
            ],
            "tracks": [
                {
                    "tracking_id": 1,
                    "object_type": "person",
                    "path": [
                        {"frame_id": 0, "timestamp_ms": 100, "x": 130, "y": 190, "confidence": 0.82, "scale": 180},
                        {"frame_id": 1, "timestamp_ms": 40, "x": 134, "y": 192, "confidence": 0.8, "scale": 180},
                        {"frame_id": 2, "timestamp_ms": 20, "x": 138, "y": 194, "confidence": 0.79, "scale": 180},
                    ],
                }
            ],
            "summary": {
                "events": [
                    {
                        "event_type": "exit",
                        "object_type": "unknown",
                        "start_frame": 20,
                        "end_frame": 30,
                        "tracking_ids": [1],
                        "details": {
                            "confidence": float("inf"),
                            "note": " finished "
                        },
                    },
                    {
                        "event_type": "interaction",
                        "object_type": "multiple",
                        "start_frame": 0,
                        "end_frame": 1,
                        "tracking_ids": [999],
                        "details": {},
                    },
                ]
            },
        }

        normalized = self.video_service._normalize_processing_result(processing_result)

        self.assertEqual(
            [point["timestamp_ms"] for point in normalized["tracks"][0]["path"]],
            [100, 200, 300],
        )
        self.assertEqual(len(normalized["summary"]["events"]), 1)
        self.assertEqual(normalized["summary"]["events"][0]["event_type"], "exit")
        self.assertEqual(normalized["summary"]["events"][0]["object_type"], "person")
        self.assertEqual(normalized["summary"]["events"][0]["start_frame"], 2)
        self.assertEqual(normalized["summary"]["events"][0]["end_frame"], 2)
        self.assertEqual(normalized["summary"]["events"][0]["details"], {"note": "finished"})

    def test_normalize_processing_result_preserves_cricket_ball_evidence_without_strict_track(self) -> None:
        processing_result = {
            "fps": 25,
            "frame_count": 8,
            "width": 1920,
            "height": 1080,
            "detections": [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0,
                    "object_type": "sports ball",
                    "bbox": [900, 420, 914, 434],
                    "confidence": 0.42,
                    "tracking_id": 99999,
                    "class_id": 32,
                },
                {
                    "frame_id": 1,
                    "timestamp_ms": 40,
                    "object_type": "sports ball",
                    "bbox": [930, 414, 944, 428],
                    "confidence": 0.42,
                    "tracking_id": 99999,
                    "class_id": 32,
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 80,
                    "object_type": "sports ball",
                    "bbox": [968, 406, 982, 420],
                    "confidence": 0.42,
                    "tracking_id": 99999,
                    "class_id": 32,
                },
            ],
            "tracks": [],
            "summary": {},
            "cricket": {
                "ball_trajectory": [[907.0, 427.0], [937.0, 421.0], [975.0, 413.0]],
                "events": [
                    {"event_type": "ball_released", "frame_id": 0},
                    {"event_type": "bat_impact", "frame_id": 2},
                ],
            },
        }

        normalized = self.video_service._normalize_processing_result(processing_result)

        self.assertEqual(len(normalized["detections"]), 3)
        self.assertEqual(len(normalized["tracks"]), 1)
        self.assertEqual(normalized["tracks"][0]["object_type"], "sports ball")
        self.assertEqual(normalized["summary"]["detections_by_type"]["sports ball"], 3)
        self.assertEqual(normalized["summary"]["tracks_by_type"]["sports ball"], 1)

    def test_build_track_results_keeps_cricket_ball_track_in_results_payload(self) -> None:
        detections = [
            Detection(
                video_id="video-1",
                frame_id=0,
                timestamp_ms=0,
                object_type="sports ball",
                bbox=[900.0, 420.0, 914.0, 434.0],
                confidence=0.42,
                tracking_id=99999,
            ),
            Detection(
                video_id="video-1",
                frame_id=1,
                timestamp_ms=40,
                object_type="sports ball",
                bbox=[930.0, 414.0, 944.0, 428.0],
                confidence=0.42,
                tracking_id=99999,
            ),
            Detection(
                video_id="video-1",
                frame_id=2,
                timestamp_ms=80,
                object_type="sports ball",
                bbox=[968.0, 406.0, 982.0, 420.0],
                confidence=0.42,
                tracking_id=99999,
            ),
        ]
        tracks = [
            Track(
                video_id="video-1",
                tracking_id=99999,
                object_type="sports ball",
                frame_start=0,
                frame_end=2,
                path=[
                    {"frame_id": 0, "timestamp_ms": 0, "x": 907.0, "y": 427.0, "confidence": 0.55, "scale": 14.0},
                    {"frame_id": 1, "timestamp_ms": 40, "x": 937.0, "y": 421.0, "confidence": 0.55, "scale": 14.0},
                    {"frame_id": 2, "timestamp_ms": 80, "x": 975.0, "y": 413.0, "confidence": 0.55, "scale": 14.0},
                ],
                distance_px=0.0,
                avg_speed_px_s=0.0,
                max_speed_px_s=0.0,
            )
        ]

        track_results = self.video_service._build_track_results(
            detections=detections,
            tracks=tracks,
            fps=25.0,
            summary_payload={
                "cricket": {
                    "ball_trajectory": [[907.0, 427.0], [937.0, 421.0], [975.0, 413.0]],
                    "events": [
                        {"event_type": "ball_released", "frame_id": 0},
                        {"event_type": "bat_impact", "frame_id": 2},
                    ],
                }
            },
        )

        self.assertEqual(len(track_results), 1)
        self.assertEqual(track_results[0].object_type, "sports ball")

class VideoServiceDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp_dir.name) / 'test.db'}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db = self.SessionLocal()
        self.video_service = VideoService(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_delete_video_removes_database_record_and_files(self) -> None:
        video = self._create_video(status=VideoStatus.COMPLETED)

        deleted_video_id = self.video_service.delete_video(video.id)
        self.db.commit()

        self.assertEqual(deleted_video_id, video.id)
        self.assertIsNone(self.db.get(Video, video.id))
        self.assertFalse(Path(video.original_path).exists())
        self.assertFalse(Path(video.processed_path).exists())

    def test_delete_videos_by_statuses_keeps_non_matching_rows(self) -> None:
        completed_video = self._create_video(status=VideoStatus.COMPLETED)
        failed_video = self._create_video(status=VideoStatus.FAILED)
        processing_video = self._create_video(status=VideoStatus.PROCESSING)

        deleted_ids = self.video_service.delete_videos_by_statuses(
            {VideoStatus.COMPLETED, VideoStatus.FAILED}
        )
        self.db.commit()

        self.assertCountEqual(deleted_ids, [completed_video.id, failed_video.id])
        self.assertIsNone(self.db.get(Video, completed_video.id))
        self.assertIsNone(self.db.get(Video, failed_video.id))
        self.assertIsNotNone(self.db.get(Video, processing_video.id))

    def test_claim_next_pending_video_promotes_oldest_pending_row(self) -> None:
        first_video = self._create_video(status=VideoStatus.PENDING)
        second_video = self._create_video(status=VideoStatus.PENDING)
        first_video.uploaded_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        second_video.uploaded_at = datetime.now(timezone.utc)
        self.db.commit()

        claimed = self.video_service.claim_next_pending_video()

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.id, first_video.id)
        self.assertEqual(claimed.status, VideoStatus.PROCESSING)
        self.assertIsNotNone(claimed.started_at)
        self.assertEqual(self.db.get(Video, second_video.id).status, VideoStatus.PENDING)

    def test_recover_stale_processing_videos_requeues_stuck_jobs(self) -> None:
        stale_video = self._create_video(status=VideoStatus.PROCESSING)
        stale_video.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        self.db.commit()

        recovered_ids = self.video_service.recover_stale_processing_videos(30)

        self.assertEqual(recovered_ids, [stale_video.id])
        refreshed = self.db.get(Video, stale_video.id)
        self.assertEqual(refreshed.status, VideoStatus.PENDING)
        self.assertIsNone(refreshed.started_at)

    def _create_video(self, status: VideoStatus) -> Video:
        base_name = f"{status.value}-{uuid.uuid4()}"
        original_path = Path(self.temp_dir.name) / f"{base_name}-source.mp4"
        original_path.write_bytes(b"source")
        processed_path = Path(self.temp_dir.name) / f"{base_name}-processed.mp4"
        processed_path.write_bytes(b"processed")

        video = Video(
            filename=f"{base_name}.mp4",
            original_path=str(original_path),
            processed_path=str(processed_path),
            status=status,
            summary={},
        )
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

if __name__ == "__main__":
    unittest.main()
