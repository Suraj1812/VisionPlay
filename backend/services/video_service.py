from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from math import hypot, isfinite
from typing import Any

from fastapi import HTTPException
from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from ai.pipeline.analytics import infer_track_events, summarize_tracks_and_detections
from backend.database.entities import Detection, Track, Video, VideoStatus
from backend.models.schemas import (
    DetectionResponse,
    EventResponse,
    FrameResultResponse,
    ResultsResponse,
    StatusResponse,
    TrackPointResponse,
    TrackResponse,
    UploadResponse,
    VideoMetadataResponse,
)
from backend.services.storage_service import storage_service
from backend.utils.config import settings


class VideoService:
    def __init__(self, db: Session):
        self.db = db

    def create_video(self, filename: str, original_path: str) -> Video:
        video = Video(
            filename=storage_service.sanitize_filename(filename),
            original_path=original_path,
        )
        self.db.add(video)
        self.db.flush()
        return video

    def get_video(self, video_id: str) -> Video | None:
        return self.db.get(Video, video_id)

    def require_video(self, video_id: str) -> Video:
        video = self.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return video

    def delete_video(self, video_id: str) -> str:
        video = self.require_video(video_id)
        return self._delete_video_record(video)

    def delete_videos_by_statuses(self, statuses: Iterable[VideoStatus | str]) -> list[str]:
        normalized_statuses = [
            status if isinstance(status, VideoStatus) else VideoStatus(str(status).strip().lower())
            for status in statuses
            if str(status).strip()
        ]
        if not normalized_statuses:
            return []

        videos = list(
            self.db.scalars(select(Video).where(Video.status.in_(normalized_statuses)))
        )
        return [self._delete_video_record(video) for video in videos]

    def delete_all_videos(self) -> list[str]:
        videos = list(self.db.scalars(select(Video)))
        return [self._delete_video_record(video) for video in videos]

    def claim_next_pending_video(self) -> Video | None:
        pending_video_id = self.db.scalar(
            select(Video.id)
            .where(Video.status == VideoStatus.PENDING)
            .order_by(Video.uploaded_at.asc())
            .limit(1)
        )
        if pending_video_id is None:
            return None

        claimed_at = datetime.now(timezone.utc)
        claim_result = self.db.execute(
            update(Video)
            .where(Video.id == pending_video_id, Video.status == VideoStatus.PENDING)
            .values(
                status=VideoStatus.PROCESSING,
                started_at=claimed_at,
                completed_at=None,
                error_message=None,
            )
        )
        self.db.commit()
        if claim_result.rowcount != 1:
            return None
        return self.require_video(pending_video_id)

    def recover_stale_processing_videos(self, stale_after_seconds: int) -> list[str]:
        threshold = datetime.now(timezone.utc) - timedelta(seconds=max(stale_after_seconds, 1))
        stale_videos = list(
            self.db.scalars(
                select(Video).where(
                    Video.status == VideoStatus.PROCESSING,
                    or_(Video.started_at.is_(None), Video.started_at < threshold),
                )
            )
        )
        recovered_ids: list[str] = []
        for video in stale_videos:
            storage_service.delete_file(video.processed_path)
            video.processed_path = None
            video.status = VideoStatus.PENDING
            video.started_at = None
            video.completed_at = None
            video.error_message = None
            recovered_ids.append(video.id)
        if recovered_ids:
            self.db.commit()
        return recovered_ids

    def delete_expired_videos(self) -> list[str]:
        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []

        if settings.retention_completed_days > 0:
            completed_before = now - timedelta(days=settings.retention_completed_days)
            completed_videos = list(
                self.db.scalars(
                    select(Video).where(
                        Video.status == VideoStatus.COMPLETED,
                        Video.completed_at.is_not(None),
                        Video.completed_at < completed_before,
                    )
                )
            )
            expired_ids.extend(self._delete_video_record(video) for video in completed_videos)

        if settings.retention_failed_days > 0:
            failed_before = now - timedelta(days=settings.retention_failed_days)
            failed_videos = list(
                self.db.scalars(
                    select(Video).where(
                        Video.status == VideoStatus.FAILED,
                        Video.completed_at.is_not(None),
                        Video.completed_at < failed_before,
                    )
                )
            )
            expired_ids.extend(self._delete_video_record(video) for video in failed_videos)

        if expired_ids:
            self.db.commit()
        return expired_ids

    def mark_processing(self, video: Video) -> None:
        video.status = VideoStatus.PROCESSING
        video.started_at = datetime.now(timezone.utc)
        video.completed_at = None
        video.error_message = None

    def mark_failed(self, video: Video, message: str) -> None:
        video.status = VideoStatus.FAILED
        video.error_message = self._truncate_error_message(message)
        video.completed_at = datetime.now(timezone.utc)

    def store_processing_results(
        self,
        video_id: str,
        processed_path: str,
        processing_result: dict[str, Any],
    ) -> Video:
        video = self.require_video(video_id)
        normalized_result = self._normalize_processing_result(processing_result)

        self.db.execute(delete(Detection).where(Detection.video_id == video_id))
        self.db.execute(delete(Track).where(Track.video_id == video_id))

        video.processed_path = processed_path
        video.status = VideoStatus.COMPLETED
        video.completed_at = datetime.now(timezone.utc)
        video.error_message = None
        video.fps = normalized_result["fps"]
        video.frame_count = normalized_result["frame_count"]
        video.width = normalized_result["width"]
        video.height = normalized_result["height"]
        video.summary = normalized_result["summary"]

        cricket_data = processing_result.get("cricket")
        if isinstance(cricket_data, dict) and cricket_data:
            summary_dict = video.summary if isinstance(video.summary, dict) else {}
            summary_dict["cricket"] = cricket_data
            video.summary = summary_dict

        detections = [
            Detection(
                video_id=video_id,
                frame_id=item["frame_id"],
                timestamp_ms=item["timestamp_ms"],
                object_type=item["object_type"],
                bbox=item["bbox"],
                confidence=item["confidence"],
                tracking_id=item["tracking_id"],
            )
            for item in normalized_result["detections"]
        ]
        tracks = [
            Track(
                video_id=video_id,
                tracking_id=item["tracking_id"],
                object_type=item["object_type"],
                frame_start=item["frame_start"],
                frame_end=item["frame_end"],
                path=item["path"],
                distance_px=item["distance_px"],
                avg_speed_px_s=item["avg_speed_px_s"],
                max_speed_px_s=item["max_speed_px_s"],
            )
            for item in normalized_result["tracks"]
        ]

        self.db.add_all(detections)
        self.db.add_all(tracks)
        return video

    def build_upload_response(self, video: Video) -> UploadResponse:
        return UploadResponse(video_id=video.id, status=video.status.value)

    def build_status_response(self, video: Video, processing_progress: int = 0) -> StatusResponse:
        return StatusResponse(
            video_id=video.id,
            status=video.status.value,
            processing_progress=max(0, min(int(processing_progress), 100)),
            error_message=video.error_message,
            uploaded_at=video.uploaded_at,
            started_at=video.started_at,
            completed_at=video.completed_at,
        )

    def build_results_response(self, video_id: str) -> ResultsResponse:
        video = self.require_video(video_id)
        if video.status != VideoStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="Video processing is not complete")

        detections = list(
            self.db.scalars(
                select(Detection)
                .where(Detection.video_id == video_id)
                .order_by(Detection.frame_id.asc(), Detection.id.asc())
            )
        )
        tracks = list(
            self.db.scalars(
                select(Track)
                .where(Track.video_id == video_id)
                .order_by(Track.distance_px.desc(), Track.object_type.asc(), Track.tracking_id.asc())
            )
        )

        frame_results = self._build_frame_results(detections)
        track_results = self._build_track_results(
            detections=detections,
            tracks=tracks,
            fps=float(video.fps or 0.0),
            summary_payload=video.summary if isinstance(video.summary, dict) else {},
        )
        events_payload = self._normalize_events(
            (video.summary or {}).get("events") if isinstance(video.summary, dict) else None,
            fallback_tracks=track_results,
        )
        summary = self._build_summary(
            detections=[
                {
                    "frame_id": item.frame_id,
                    "timestamp_ms": item.timestamp_ms,
                    "object_type": item.object_type,
                    "bbox": [float(value) for value in item.bbox],
                    "confidence": float(item.confidence),
                    "tracking_id": item.tracking_id,
                }
                for item in detections
            ],
            tracks=track_results,
            summary_payload=video.summary,
            fps=float(video.fps or 0.0),
            events=events_payload,
        )

        metadata = VideoMetadataResponse(
            id=video.id,
            filename=video.filename,
            status=video.status.value,
            fps=video.fps,
            frame_count=video.frame_count,
            width=video.width,
            height=video.height,
            uploaded_at=video.uploaded_at,
            started_at=video.started_at,
            completed_at=video.completed_at,
            source_video_url=storage_service.media_url_for(video.original_path),
            processed_video_url=storage_service.media_url_for(video.processed_path),
        )

        cricket_data = {}
        if isinstance(video.summary, dict):
            cricket_data = video.summary.get("cricket", {})

        return ResultsResponse(
            video=metadata,
            summary=summary,
            events=[
                EventResponse(
                    event_type=event["event_type"],
                    start_frame=event["start_frame"],
                    end_frame=event["end_frame"],
                    duration_frames=event["duration_frames"],
                    object_type=event["object_type"],
                    tracking_ids=event["tracking_ids"],
                    details=event["details"],
                )
                for event in events_payload
            ],
            frames=frame_results,
            tracks=track_results,
            cricket=cricket_data if isinstance(cricket_data, dict) else {},
        )

    def _delete_video_record(self, video: Video) -> str:
        video_id = video.id
        storage_service.delete_file(video.original_path)
        storage_service.delete_file(video.processed_path)
        self.db.delete(video)
        return video_id

    def _build_frame_results(self, detections: list[Detection]) -> list[FrameResultResponse]:
        grouped_frames: dict[int, list[DetectionResponse]] = defaultdict(list)
        timestamps_by_frame: dict[int, int] = {}

        for detection in detections:
            grouped_frames[detection.frame_id].append(
                DetectionResponse(
                    frame_id=detection.frame_id,
                    timestamp_ms=detection.timestamp_ms,
                    object_type=detection.object_type,
                    class_id=-1,
                    bbox=[float(value) for value in detection.bbox],
                    confidence=float(detection.confidence),
                    tracking_id=detection.tracking_id,
                )
            )
            timestamps_by_frame[detection.frame_id] = detection.timestamp_ms

        return [
            FrameResultResponse(
                frame_id=frame_id,
                timestamp_ms=timestamps_by_frame[frame_id],
                detections=grouped_frames[frame_id],
            )
            for frame_id in sorted(grouped_frames.keys())
        ]

    def _build_track_results(
        self,
        detections: list[Detection],
        tracks: list[Track],
        fps: float,
        summary_payload: dict[str, Any],
    ) -> list[TrackResponse]:
        cricket_payload = summary_payload.get("cricket") if isinstance(summary_payload, dict) else {}
        strong_cricket_ball_evidence = self._has_strong_cricket_ball_evidence(
            cricket_payload if isinstance(cricket_payload, dict) else {}
        )
        detections_by_track: dict[tuple[str, int], list[Detection]] = defaultdict(list)
        for detection in detections:
            if detection.tracking_id is None:
                continue
            detections_by_track[(detection.object_type, detection.tracking_id)].append(detection)

        primary_track_keys = {
            (str(item.get("object_type", "")).strip().lower(), int(item.get("tracking_id")))
            for item in summary_payload.get("primary_tracks", [])
            if isinstance(item, dict)
            and str(item.get("object_type", "")).strip()
            and self._coerce_int(item.get("tracking_id"), default=-1, minimum=0) >= 0
        }

        track_results: list[TrackResponse] = []
        for track in tracks:
            path = self._normalize_track_path(track.path, fps)
            if not path:
                continue

            distance_px, avg_speed_px_s, max_speed_px_s = self._compute_track_metrics(path, fps)
            duration_ms = self._compute_duration_ms(path, fps)
            track_detections = detections_by_track.get((track.object_type, track.tracking_id), [])
            confidences = [float(item.confidence) for item in track_detections] or [
                float(point.get("confidence", 0.0)) for point in path
            ]
            if strong_cricket_ball_evidence and track.object_type in settings.small_object_class_name_list:
                confidence_floor = min(settings.min_small_object_track_avg_confidence + 0.05, 0.58)
                confidences = [max(confidence, confidence_floor) for confidence in confidences]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            if not self._is_track_qualified(path, distance_px, duration_ms, fps):
                continue
            if avg_confidence < self._minimum_avg_confidence(track.object_type):
                continue

            max_confidence = max(confidences, default=0.0)
            primary_score = self._compute_primary_score(
                distance_px=distance_px,
                duration_ms=duration_ms,
                avg_confidence=avg_confidence,
                detection_count=len(path),
            )
            track_results.append(
                TrackResponse(
                    tracking_id=track.tracking_id,
                    object_type=track.object_type,
                    class_id=-1,
                    frame_start=path[0]["frame_id"],
                    frame_end=path[-1]["frame_id"],
                    duration_frames=path[-1]["frame_id"] - path[0]["frame_id"] + 1,
                    duration_ms=round(max(duration_ms, 0.0), 2),
                    time_in_frame_ms=round(max(duration_ms, 0.0), 2),
                    detection_count=len(path),
                    avg_confidence=round(avg_confidence, 4),
                    max_confidence=round(max_confidence, 4),
                    path=[
                        TrackPointResponse(
                            frame_id=point["frame_id"],
                            timestamp_ms=point["timestamp_ms"],
                            x=float(point["x"]),
                            y=float(point["y"]),
                            confidence=float(point.get("confidence", 0.0)),
                            scale=float(point.get("scale", 0.0)),
                        )
                        for point in path
                    ],
                    distance_px=round(max(distance_px, 0.0), 2),
                    avg_speed_px_s=round(max(avg_speed_px_s, 0.0), 2),
                    max_speed_px_s=round(max(max_speed_px_s, 0.0), 2),
                    primary_score=round(primary_score, 2),
                    is_primary=(track.object_type, track.tracking_id) in primary_track_keys,
                )
            )

        track_results.sort(key=lambda item: (-item.primary_score, item.object_type, item.tracking_id))
        return track_results

    def _normalize_processing_result(self, processing_result: dict[str, Any]) -> dict[str, Any]:
        fps = self._coerce_float(processing_result.get("fps"), default=25.0, minimum=0.1)
        frame_count = self._coerce_int(processing_result.get("frame_count"), default=0, minimum=0)
        width = self._coerce_int(processing_result.get("width"), default=0, minimum=0)
        height = self._coerce_int(processing_result.get("height"), default=0, minimum=0)
        cricket_payload = (
            processing_result.get("cricket")
            if isinstance(processing_result.get("cricket"), dict)
            else {}
        )

        detections = [
            normalized
            for item in processing_result.get("detections", [])
            if (normalized := self._normalize_detection(item, width, height)) is not None
        ]
        tracks = [
            normalized
            for item in processing_result.get("tracks", [])
            if (normalized := self._normalize_track(item, fps)) is not None
        ]
        tracks.extend(self._build_cricket_small_object_fallback_tracks(detections, tracks, fps, cricket_payload))

        track_keys = {(track["object_type"], track["tracking_id"]) for track in tracks}
        detections = [
            item
            for item in detections
            if item["tracking_id"] is not None
            and (
                (item["object_type"], item["tracking_id"]) in track_keys
                or self._should_preserve_cricket_detection(item, cricket_payload)
            )
        ]
        detections.sort(key=lambda item: (item["frame_id"], item["object_type"], -item["confidence"]))
        tracks.sort(key=lambda item: (-item["primary_score"], item["object_type"], item["tracking_id"]))

        summary = self._build_summary(
            detections=detections,
            tracks=tracks,
            summary_payload=processing_result.get("summary"),
            fps=fps,
            events=self._normalize_events(
                (processing_result.get("summary") or {}).get("events")
                if isinstance(processing_result.get("summary"), dict)
                else None,
                fallback_tracks=tracks,
            ),
        )

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "detections": detections,
            "tracks": tracks,
            "summary": summary,
        }

    def _build_cricket_small_object_fallback_tracks(
        self,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]],
        fps: float,
        cricket_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self._has_strong_cricket_ball_evidence(cricket_payload):
            return []

        existing_keys = {
            (track["object_type"], track["tracking_id"])
            for track in tracks
            if isinstance(track, dict)
        }
        grouped_detections: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in detections:
            tracking_id = item.get("tracking_id")
            object_type = str(item.get("object_type", "")).strip().lower()
            if tracking_id is None or object_type not in settings.small_object_class_name_list:
                continue
            grouped_detections[(object_type, tracking_id)].append(item)

        fallback_tracks: list[dict[str, Any]] = []
        confidence_floor = min(settings.min_small_object_track_avg_confidence + 0.05, 0.58)
        for (object_type, tracking_id), grouped_items in grouped_detections.items():
            if (object_type, tracking_id) in existing_keys:
                continue

            deduped_by_frame: dict[int, dict[str, Any]] = {}
            for item in grouped_items:
                existing = deduped_by_frame.get(item["frame_id"])
                if existing is None or item["confidence"] >= existing["confidence"]:
                    deduped_by_frame[item["frame_id"]] = item
            ordered_items = [deduped_by_frame[frame_id] for frame_id in sorted(deduped_by_frame)]
            if len(ordered_items) < 2:
                continue

            path = self._stabilize_track_path_timestamps(
                [
                    {
                        "frame_id": item["frame_id"],
                        "timestamp_ms": item["timestamp_ms"],
                        "x": round((item["bbox"][0] + item["bbox"][2]) / 2.0, 2),
                        "y": round((item["bbox"][1] + item["bbox"][3]) / 2.0, 2),
                        "confidence": round(max(float(item["confidence"]), confidence_floor), 4),
                        "scale": round(
                            max(
                                float(item["bbox"][2]) - float(item["bbox"][0]),
                                float(item["bbox"][3]) - float(item["bbox"][1]),
                            ),
                            2,
                        ),
                    }
                    for item in ordered_items
                ],
                fps,
            )
            if len(path) < 2:
                continue

            confidences = [float(point.get("confidence", 0.0)) for point in path]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            if avg_confidence < min(settings.min_small_object_track_avg_confidence, 0.34):
                continue

            distance_px, avg_speed_px_s, max_speed_px_s = self._compute_track_metrics(path, fps)
            duration_ms = self._compute_duration_ms(path, fps)
            if distance_px < max(settings.min_track_distance_px * 0.5, 4.0) and duration_ms < 250.0:
                continue

            primary_score = self._compute_primary_score(
                distance_px=distance_px,
                duration_ms=duration_ms,
                avg_confidence=avg_confidence,
                detection_count=len(path),
            )
            fallback_tracks.append(
                {
                    "tracking_id": tracking_id,
                    "object_type": object_type,
                    "class_id": self._coerce_int(
                        ordered_items[0].get("class_id"),
                        default=-1,
                    ),
                    "frame_start": path[0]["frame_id"],
                    "frame_end": path[-1]["frame_id"],
                    "duration_frames": path[-1]["frame_id"] - path[0]["frame_id"] + 1,
                    "duration_ms": round(max(duration_ms, 0.0), 2),
                    "time_in_frame_ms": round(max(duration_ms, 0.0), 2),
                    "detection_count": len(path),
                    "avg_confidence": round(avg_confidence, 4),
                    "max_confidence": round(max(confidences, default=0.0), 4),
                    "path": path,
                    "distance_px": round(max(distance_px, 0.0), 2),
                    "avg_speed_px_s": round(max(avg_speed_px_s, 0.0), 2),
                    "max_speed_px_s": round(max(max_speed_px_s, 0.0), 2),
                    "primary_score": round(primary_score, 2),
                    "is_primary": False,
                }
            )

        return fallback_tracks

    def _should_preserve_cricket_detection(
        self,
        detection: dict[str, Any],
        cricket_payload: dict[str, Any],
    ) -> bool:
        object_type = str(detection.get("object_type", "")).strip().lower()
        if object_type not in settings.small_object_class_name_list:
            return False
        return self._has_strong_cricket_ball_evidence(cricket_payload)

    @staticmethod
    def _has_strong_cricket_ball_evidence(cricket_payload: dict[str, Any]) -> bool:
        if not isinstance(cricket_payload, dict) or not cricket_payload:
            return False

        ball_trajectory = cricket_payload.get("ball_trajectory")
        if isinstance(ball_trajectory, list) and len(ball_trajectory) >= 8:
            return True

        raw_events = cricket_payload.get("events")
        if not isinstance(raw_events, list):
            return False

        ball_event_types = {
            "ball_released",
            "ball_bounced",
            "bat_impact",
            "wicket",
        }
        cricket_ball_events = 0
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type", "")).strip().lower()
            if event_type in ball_event_types:
                cricket_ball_events += 1
            if cricket_ball_events >= 2:
                return True
        return False

    def _normalize_detection(
        self,
        item: Any,
        width: int,
        height: int,
    ) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        object_type = str(item.get("object_type", "")).strip().lower()
        if not object_type:
            return None

        bbox = self._normalize_bbox(item.get("bbox"), width, height)
        if bbox is None:
            return None

        tracking_id_raw = item.get("tracking_id")
        tracking_id = None
        if tracking_id_raw not in (None, ""):
            tracking_id = self._coerce_int(tracking_id_raw, default=-1, minimum=0)
            if tracking_id < 0:
                tracking_id = None

        return {
            "frame_id": self._coerce_int(item.get("frame_id"), default=0, minimum=0),
            "timestamp_ms": self._coerce_int(item.get("timestamp_ms"), default=0, minimum=0),
            "object_type": object_type,
            "class_id": self._coerce_int(item.get("class_id"), default=-1),
            "bbox": bbox,
            "confidence": round(
                min(max(self._coerce_float(item.get("confidence"), default=0.0), 0.0), 1.0),
                4,
            ),
            "tracking_id": tracking_id,
        }

    def _normalize_track(self, item: Any, fps: float) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        object_type = str(item.get("object_type", "")).strip().lower()
        if not object_type:
            return None

        tracking_id = self._coerce_int(item.get("tracking_id"), default=-1, minimum=0)
        if tracking_id < 0:
            return None

        path = self._normalize_track_path(item.get("path"), fps)
        if not path:
            return None

        distance_px, avg_speed_px_s, max_speed_px_s = self._compute_track_metrics(path, fps)
        duration_ms = self._compute_duration_ms(path, fps)
        if not self._is_track_qualified(path, distance_px, duration_ms, fps):
            return None

        confidences = [float(point.get("confidence", 0.0)) for point in path]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_confidence = max(confidences, default=0.0)
        primary_score = self._coerce_float(item.get("primary_score"), default=0.0)
        if avg_confidence < self._minimum_avg_confidence(object_type):
            return None
        if primary_score <= 0:
            primary_score = self._compute_primary_score(
                distance_px=distance_px,
                duration_ms=duration_ms,
                avg_confidence=avg_confidence,
                detection_count=len(path),
            )

        return {
            "tracking_id": tracking_id,
            "object_type": object_type,
            "class_id": self._coerce_int(item.get("class_id"), default=-1),
            "frame_start": path[0]["frame_id"],
            "frame_end": path[-1]["frame_id"],
            "duration_frames": path[-1]["frame_id"] - path[0]["frame_id"] + 1,
            "duration_ms": round(max(duration_ms, 0.0), 2),
            "time_in_frame_ms": round(max(duration_ms, 0.0), 2),
            "detection_count": len(path),
            "avg_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "path": path,
            "distance_px": round(max(distance_px, 0.0), 2),
            "avg_speed_px_s": round(max(avg_speed_px_s, 0.0), 2),
            "max_speed_px_s": round(max(max_speed_px_s, 0.0), 2),
            "primary_score": round(primary_score, 2),
            "is_primary": bool(item.get("is_primary", False)),
        }

    def _normalize_track_path(self, path_payload: Any, fps: float) -> list[dict[str, Any]]:
        if not isinstance(path_payload, list):
            return []

        deduped_by_frame: dict[int, dict[str, Any]] = {}
        for point in path_payload:
            if not isinstance(point, dict):
                continue

            x = self._coerce_float(point.get("x"), default=float("nan"))
            y = self._coerce_float(point.get("y"), default=float("nan"))
            if not isfinite(x) or not isfinite(y):
                continue

            frame_id = self._coerce_int(point.get("frame_id"), default=-1, minimum=0)
            if frame_id < 0:
                continue

            deduped_by_frame[frame_id] = {
                "frame_id": frame_id,
                "timestamp_ms": self._coerce_int(point.get("timestamp_ms"), default=0, minimum=0),
                "x": round(x, 2),
                "y": round(y, 2),
                "confidence": round(
                    min(max(self._coerce_float(point.get("confidence"), default=0.0), 0.0), 1.0),
                    4,
                ),
                "scale": round(max(self._coerce_float(point.get("scale"), default=0.0), 0.0), 2),
            }

        ordered_points = [deduped_by_frame[frame_id] for frame_id in sorted(deduped_by_frame)]
        return self._stabilize_track_path_timestamps(ordered_points, fps)

    def _build_summary(
        self,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]] | list[TrackResponse],
        summary_payload: Any,
        fps: float,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        frame_errors = self._coerce_int(summary_payload.get("frame_errors"), default=0, minimum=0)
        normalized_events = events or self._normalize_events(summary_payload.get("events"), fallback_tracks=tracks)

        normalized_tracks: list[dict[str, Any]] = []
        for track in tracks:
            if isinstance(track, TrackResponse):
                normalized_tracks.append(track.model_dump())
            elif isinstance(track, dict):
                normalized_tracks.append(track)

        summary = summarize_tracks_and_detections(
            detections=detections,
            tracks=normalized_tracks,
            fps=fps,
            frame_errors=frame_errors,
            events=normalized_events,
        )

        return summary

    def _normalize_events(
        self,
        raw_events: Any,
        fallback_tracks: list[dict[str, Any]] | list[TrackResponse],
    ) -> list[dict[str, Any]]:
        normalized_tracks = [
            track.model_dump() if isinstance(track, TrackResponse) else track
            for track in fallback_tracks
            if isinstance(track, (TrackResponse, dict))
        ]
        known_track_ids = {
            self._coerce_int(track.get("tracking_id"), default=-1, minimum=0)
            for track in normalized_tracks
            if isinstance(track, dict)
        }
        known_track_ids.discard(-1)
        track_ranges: dict[int, tuple[int, int]] = {}
        track_types: dict[int, set[str]] = defaultdict(set)
        for track in normalized_tracks:
            if not isinstance(track, dict):
                continue
            tracking_id = self._coerce_int(track.get("tracking_id"), default=-1, minimum=0)
            if tracking_id < 0:
                continue
            frame_start = self._coerce_int(track.get("frame_start"), default=0, minimum=0)
            frame_end = self._coerce_int(track.get("frame_end"), default=frame_start, minimum=frame_start)
            current_range = track_ranges.get(tracking_id)
            if current_range is None:
                track_ranges[tracking_id] = (frame_start, frame_end)
            else:
                track_ranges[tracking_id] = (
                    min(current_range[0], frame_start),
                    max(current_range[1], frame_end),
                )
            object_type = str(track.get("object_type", "")).strip().lower()
            if object_type:
                track_types[tracking_id].add(object_type)

        events: list[dict[str, Any]] = []
        seen_event_keys: set[tuple[str, int, int, str, tuple[int, ...]]] = set()
        if isinstance(raw_events, list):
            for event in raw_events:
                if not isinstance(event, dict):
                    continue
                object_type = str(event.get("object_type", "")).strip().lower() or "unknown"
                start_frame = self._coerce_int(event.get("start_frame"), default=0, minimum=0)
                end_frame = self._coerce_int(event.get("end_frame"), default=start_frame, minimum=start_frame)
                tracking_ids_raw = event.get("tracking_ids", [])
                tracking_ids = []
                if isinstance(tracking_ids_raw, list):
                    tracking_ids = [
                        tracking_id
                        for tracking_id in (
                            self._coerce_int(value, default=-1, minimum=0)
                            for value in tracking_ids_raw
                        )
                        if tracking_id in known_track_ids
                    ]

                event_type = str(event.get("event_type", "")).strip().lower() or "observation"
                if tracking_ids:
                    if event_type == "interaction":
                        clamped_start = max(track_ranges[tracking_id][0] for tracking_id in tracking_ids)
                        clamped_end = min(track_ranges[tracking_id][1] for tracking_id in tracking_ids)
                        if clamped_end < clamped_start:
                            continue
                        object_type = "multiple"
                    else:
                        clamped_start = min(track_ranges[tracking_id][0] for tracking_id in tracking_ids)
                        clamped_end = max(track_ranges[tracking_id][1] for tracking_id in tracking_ids)
                        associated_types = {
                            label
                            for tracking_id in tracking_ids
                            for label in track_types.get(tracking_id, set())
                        }
                        if len(associated_types) == 1:
                            object_type = next(iter(associated_types))

                    start_frame = min(max(start_frame, clamped_start), clamped_end)
                    end_frame = min(max(end_frame, start_frame), clamped_end)
                elif event_type in {"entry", "exit", "interaction"}:
                    continue

                normalized_event = {
                    "event_type": event_type,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": max(end_frame - start_frame + 1, 1),
                    "object_type": object_type,
                    "tracking_ids": tracking_ids,
                    "details": self._normalize_event_details(event.get("details")),
                }
                event_key = (
                    normalized_event["event_type"],
                    normalized_event["start_frame"],
                    normalized_event["end_frame"],
                    normalized_event["object_type"],
                    tuple(normalized_event["tracking_ids"]),
                )
                if event_key in seen_event_keys:
                    continue
                seen_event_keys.add(event_key)
                events.append(normalized_event)

        if not events:
            return infer_track_events(normalized_tracks)

        events.sort(key=lambda item: (item["start_frame"], item["event_type"], item["object_type"]))
        return events

    @staticmethod
    def _stabilize_track_path_timestamps(
        ordered_points: list[dict[str, Any]],
        fps: float,
    ) -> list[dict[str, Any]]:
        if not ordered_points:
            return []

        stabilized_points: list[dict[str, Any]] = []
        for index, point in enumerate(ordered_points):
            stabilized_point = dict(point)
            fallback_timestamp = (
                int(round((point["frame_id"] / max(fps, 1.0)) * 1000.0))
                if fps > 0
                else max(VideoService._coerce_int(point.get("timestamp_ms"), default=0), 0)
            )
            timestamp_ms = max(
                VideoService._coerce_int(point.get("timestamp_ms"), default=fallback_timestamp),
                0,
            )

            if index == 0:
                stabilized_point["timestamp_ms"] = timestamp_ms if timestamp_ms > 0 else fallback_timestamp
                stabilized_points.append(stabilized_point)
                continue

            previous_point = stabilized_points[-1]
            frame_delta = max(point["frame_id"] - previous_point["frame_id"], 1)
            minimum_step_ms = (
                max(int(round((frame_delta / max(fps, 1.0)) * 1000.0)), 1)
                if fps > 0
                else frame_delta
            )
            stabilized_point["timestamp_ms"] = max(
                timestamp_ms,
                fallback_timestamp,
                previous_point["timestamp_ms"] + minimum_step_ms,
            )
            stabilized_points.append(stabilized_point)

        return stabilized_points

    @staticmethod
    def _normalize_event_details(details: Any) -> dict[str, Any]:
        if not isinstance(details, dict):
            return {}

        normalized_details: dict[str, Any] = {}
        for key, value in details.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            if isinstance(value, bool):
                normalized_details[normalized_key] = value
                continue
            if isinstance(value, int):
                normalized_details[normalized_key] = value
                continue
            if isinstance(value, float):
                if isfinite(value):
                    normalized_details[normalized_key] = round(value, 4)
                continue
            if isinstance(value, str):
                normalized_details[normalized_key] = value.strip()
                continue
            if isinstance(value, list):
                normalized_values: list[Any] = []
                for item in value:
                    if isinstance(item, (bool, int)):
                        normalized_values.append(item)
                    elif isinstance(item, float) and isfinite(item):
                        normalized_values.append(round(item, 4))
                    elif isinstance(item, str):
                        normalized_values.append(item.strip())
                if normalized_values:
                    normalized_details[normalized_key] = normalized_values

        return normalized_details

    @staticmethod
    def _compute_track_metrics(path: list[dict[str, Any]], fps: float) -> tuple[float, float, float]:
        distance_px = 0.0

        for previous, current in zip(path, path[1:]):
            distance = hypot(current["x"] - previous["x"], current["y"] - previous["y"])
            distance_px += distance

        speed_samples = VideoService._compute_speed_samples(path, fps)
        avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
        max_speed = max(speed_samples, default=0.0)
        return distance_px, avg_speed, max_speed

    @staticmethod
    def _compute_speed_samples(path: list[dict[str, Any]], fps: float) -> list[float]:
        speed_samples: list[float] = []

        for previous, current in zip(path, path[1:]):
            distance = hypot(current["x"] - previous["x"], current["y"] - previous["y"])
            seconds = 0.0
            timestamp_delta_ms = current["timestamp_ms"] - previous["timestamp_ms"]
            if timestamp_delta_ms > 0:
                seconds = timestamp_delta_ms / 1000.0
            elif fps > 0:
                frame_delta = max(current["frame_id"] - previous["frame_id"], 1)
                seconds = frame_delta / fps

            if seconds > 0:
                speed_samples.append(distance / seconds)

        return speed_samples

    @staticmethod
    def _compute_duration_ms(path: list[dict[str, Any]], fps: float) -> float:
        if len(path) <= 1:
            return round(1000.0 / max(fps, 1.0), 2) if path else 0.0

        timestamp_delta_ms = path[-1]["timestamp_ms"] - path[0]["timestamp_ms"]
        if timestamp_delta_ms > 0:
            return float(timestamp_delta_ms)

        frame_delta = max(path[-1]["frame_id"] - path[0]["frame_id"], 1)
        return round((frame_delta / max(fps, 1.0)) * 1000.0, 2)

    @staticmethod
    def _compute_primary_score(
        distance_px: float,
        duration_ms: float,
        avg_confidence: float,
        detection_count: int,
    ) -> float:
        return (
            (distance_px * 0.03)
            + (duration_ms / 500.0)
            + (avg_confidence * 10.0)
            + (detection_count * 0.5)
        )

    @staticmethod
    def _is_track_qualified(
        path: list[dict[str, Any]],
        distance_px: float,
        duration_ms: float,
        fps: float,
    ) -> bool:
        if len(path) < settings.min_track_points:
            return False

        if distance_px >= settings.min_track_distance_px:
            return True

        minimum_static_duration_ms = max(
            int((settings.min_track_points / max(fps, 1.0)) * 1000.0 * 2.0),
            750,
        )
        return duration_ms >= minimum_static_duration_ms

    @staticmethod
    def _minimum_avg_confidence(object_type: str) -> float:
        if object_type in settings.small_object_class_name_list:
            return settings.min_small_object_track_avg_confidence
        return settings.min_track_avg_confidence

    @staticmethod
    def _normalize_bbox(bbox: Any, width: int, height: int) -> list[float] | None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None

        if not all(isfinite(value) for value in [x1, y1, x2, y2]):
            return None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if width > 0:
            left = min(max(left, 0.0), float(width))
            right = min(max(right, 0.0), float(width))
        if height > 0:
            top = min(max(top, 0.0), float(height))
            bottom = min(max(bottom, 0.0), float(height))

        if right - left <= 0 or bottom - top <= 0:
            return None

        return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]

    @staticmethod
    def _truncate_error_message(message: str) -> str:
        sanitized = " ".join(str(message).split())
        if len(sanitized) <= settings.max_error_message_length:
            return sanitized
        return f"{sanitized[: settings.max_error_message_length - 3]}..."

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int | None = None) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None:
            parsed = max(parsed, minimum)
        return parsed

    @staticmethod
    def _coerce_float(value: Any, default: float, minimum: float | None = None) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(parsed):
            return default
        if minimum is not None:
            parsed = max(parsed, minimum)
        return parsed
