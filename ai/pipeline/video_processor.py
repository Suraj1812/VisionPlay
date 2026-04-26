from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ai.detection.yolo_detector import YOLODetector
from ai.pipeline.analytics import build_track_analytics
from ai.tracking.byte_tracker import TrackedObject, UniversalByteTracker
from ai.perception.camera_cuts import CameraCutDetector
from ai.perception.ball_tracker import SpatioTemporalBallTracker
from ai.cricket.scoreboard_ocr import ScoreboardOCR
from ai.cricket.event_classifier import CricketEventClassifier
from ai.cricket.team_analyzer import TeamAnalyzer
from ai.cricket.analytics_aggregator import CricketAnalyticsAggregator
from ai.cricket.profile_analyzer import CricketClipProfileAnalyzer
from ai.cricket.audio_transcriber import CricketAudioTranscriber
from ai.cricket.insight_builder import build_cricket_package
from backend.utils.config import settings


logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self) -> None:
        self.detector = YOLODetector()
        self.cut_detector = CameraCutDetector()
        self.ball_tracker = SpatioTemporalBallTracker()
        self.scoreboard_ocr = ScoreboardOCR()
        self.event_classifier: CricketEventClassifier | None = None
        self.team_analyzer = TeamAnalyzer()
        self.analytics = CricketAnalyticsAggregator()
        self.profile_analyzer = CricketClipProfileAnalyzer()
        self.audio_transcriber = CricketAudioTranscriber()

    def analyze_live_frame(
        self,
        frame: np.ndarray,
        tracker: UniversalByteTracker,
        frame_id: int,
        track_hits: dict[tuple[str, int], int],
    ) -> list[TrackedObject]:
        should_run_detector = (
            frame_id <= 1
            or settings.live_detection_frame_stride <= 1
            or frame_id % settings.live_detection_frame_stride == 0
            or not tracker.has_active_tracks()
        )

        if should_run_detector:
            detected_objects = self.detector.detect(frame, frame_id=frame_id)
            tracked_objects = tracker.update(detected_objects)
        else:
            tracked_objects = tracker.predict_only()
            if not tracked_objects:
                detected_objects = self.detector.detect(frame, frame_id=frame_id)
                tracked_objects = tracker.update(detected_objects)

        normalized = self._normalize_tracked_objects(
            frame=frame,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            tracked_objects=tracked_objects,
            profile="live",
        )
        return self._filter_live_tracks(normalized, track_hits)

    def process_video(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_value if fps_value > 0 else 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._reset_batch_modules(fps)
        base_detection_stride = self._batch_stride_for_fps(
            fps,
            settings.target_detection_fps,
            minimum=settings.detection_frame_stride,
        )
        specialized_detection_stride = self._batch_stride_for_fps(
            fps,
            settings.cricket_specialized_detection_fps,
            minimum=1,
        )
        base_cricket_analysis_stride = self._batch_stride_for_fps(
            fps,
            settings.target_cricket_analysis_fps,
            minimum=1,
        )
        specialized_cricket_analysis_stride = self._batch_stride_for_fps(
            fps,
            settings.cricket_specialized_analysis_fps,
            minimum=1,
        )
        scoreboard_stride = self._batch_stride_for_fps(
            fps,
            settings.target_scoreboard_analysis_fps,
            minimum=1,
        )

        tracker = UniversalByteTracker(frame_rate=fps)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        writer: cv2.VideoWriter | None = None
        detections_payload: list[dict[str, Any]] = []
        heuristic_phase_timeline: list[dict[str, Any]] = []
        active_phase_entry: dict[str, Any] | None = None
        frame_id = 0
        frame_errors = 0
        track_hits: dict[tuple[str, int], int] = {}
        profile_report = self.profile_analyzer.build_report()

        if progress_callback is not None and frame_count > 0:
            progress_callback(0, frame_count)

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if writer is None:
                    if width <= 0 or height <= 0:
                        height, width = frame.shape[:2]

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to write processed video to {output_path}")

                    self.event_classifier = CricketEventClassifier(
                        fps=fps, frame_width=width, frame_height=height
                    )

                try:
                    scene_cut = self.cut_detector.is_cut(frame)
                    if scene_cut:
                        tracker.reset()
                        track_hits.clear()
                        self.ball_tracker.reset()
                        if self.event_classifier:
                            self.event_classifier.reset()
                        self.analytics.on_event({
                            "event_type": "camera_cut",
                            "frame_id": frame_id,
                            "timestamp_ms": int((frame_id / fps) * 1000),
                        })
                    is_replay = self.cut_detector.is_replay_segment()
                    detection_stride = (
                        specialized_detection_stride
                        if profile_report.specialized
                        else base_detection_stride
                    )
                    cricket_analysis_stride = (
                        specialized_cricket_analysis_stride
                        if profile_report.specialized
                        else base_cricket_analysis_stride
                    )

                    should_run_detector = (
                        scene_cut
                        or frame_id == 0
                        or detection_stride <= 1
                        or frame_id % detection_stride == 0
                        or not tracker.has_active_tracks()
                    )
                    if should_run_detector:
                        detected_objects = self.detector.detect(frame, frame_id=frame_id)
                        tracked_objects = tracker.update(detected_objects)
                    else:
                        tracked_objects = tracker.predict_only()
                    frame_detections = self._normalize_tracked_objects(
                        frame=frame,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        tracked_objects=tracked_objects,
                    )
                    frame_detections = self._filter_confirmed_tracks(
                        tracked_objects=frame_detections,
                        track_hits=track_hits,
                    )
                    frame_detections = self._retain_cricket_detections(frame_detections)

                    yolo_ball = next(
                        (item.bbox for item in frame_detections if item.object_type == "sports ball"),
                        None,
                    )
                    should_run_cricket_analysis = (
                        scene_cut
                        or frame_id == 0
                        or cricket_analysis_stride <= 1
                        or frame_id % cricket_analysis_stride == 0
                    )
                    predicted_ball = self.ball_tracker.predict(frame, yolo_ball) if should_run_cricket_analysis else None
                    if predicted_ball and not yolo_ball:
                        frame_detections.append(
                            TrackedObject(
                                object_type="sports ball",
                                class_id=32,
                                bbox=predicted_ball[:4],
                                confidence=predicted_ball[4],
                                tracking_id=99999,
                                is_predicted=True,
                                appearance_signature=None,
                            )
                        )

                    player_dets = [
                        {"bbox": item.bbox, "tracking_id": item.tracking_id, "confidence": item.confidence}
                        for item in frame_detections
                        if item.object_type == "person"
                    ]
                    team_results = self.team_analyzer.analyze(frame, player_dets, frame_id) if should_run_cricket_analysis else []
                    self.profile_analyzer.observe(frame, player_dets, scene_cut=scene_cut)
                    profile_report = self.profile_analyzer.build_report()
                    if should_run_cricket_analysis:
                        active_phase_entry = self._update_phase_timeline(
                            timeline=heuristic_phase_timeline,
                            active_entry=active_phase_entry,
                            team_results=team_results,
                            person_detections=player_dets,
                            frame_width=frame.shape[1],
                            frame_height=frame.shape[0],
                            frame_id=frame_id,
                            fps=fps,
                        )

                    tid_to_team = {p.tracking_id: p for p in team_results}

                    ball_det = next(
                        ({"bbox": item.bbox, "confidence": item.confidence}
                         for item in frame_detections
                         if item.object_type == "sports ball"),
                        None,
                    )
                    bat_dets = [
                        {"bbox": item.bbox}
                        for item in frame_detections
                        if "bat" in item.object_type
                    ]

                    ball_velocity = self.ball_tracker.get_velocity()
                    ball_acceleration = self.ball_tracker.get_acceleration()

                    if self.event_classifier and not is_replay and should_run_cricket_analysis:
                        cricket_events = self.event_classifier.process_frame(
                            frame_id=frame_id,
                            ball_detection=ball_det,
                            player_detections=player_dets,
                            bat_detections=bat_dets,
                            fps=fps,
                            ball_velocity=ball_velocity,
                            ball_acceleration=ball_acceleration,
                        )
                        for ev in cricket_events:
                            ev_dict = {
                                "event_type": ev.event_type.value,
                                "frame_id": ev.frame_id,
                                "timestamp_ms": ev.timestamp_ms,
                                "confidence": ev.confidence,
                                "details": ev.details,
                            }
                            self.analytics.on_event(ev_dict)
                            logger.info(
                                "CRICKET [%s] frame=%d conf=%.2f %s",
                                ev.event_type.value, ev.frame_id, ev.confidence, ev.details,
                            )

                    should_run_scoreboard = (
                        not is_replay
                        and (
                            frame_id == 0
                            or scoreboard_stride <= 1
                            or frame_id % scoreboard_stride == 0
                        )
                    )
                    if should_run_scoreboard:
                        score_data = self.scoreboard_ocr.extract(
                            frame,
                            frame_id=frame_id,
                            fps=fps,
                            force=not profile_report.specialized,
                        )
                        if score_data.detected and score_data.runs >= 0:
                            logger.info(
                                "SCOREBOARD: %s overs=%s rr=%.1f",
                                score_data.score, score_data.overs, score_data.run_rate,
                            )

                except Exception:
                    frame_errors += 1
                    logger.exception("Detection or tracking failed at frame %s", frame_id)
                    frame_detections = []

                for item in frame_detections:
                    det_dict = self._serialize_detection(frame_id=frame_id, fps=fps, item=item)
                    if item.object_type == "person" and item.tracking_id in tid_to_team:
                        p_info = tid_to_team[item.tracking_id]
                        det_dict["team_id"] = p_info.team_id
                        det_dict["role"] = p_info.role
                    detections_payload.append(det_dict)
                    self._draw_annotation(frame, item)

                writer.write(frame)
                frame_id += 1
                effective_total = frame_count if frame_count > 0 else frame_id
                if progress_callback is not None and effective_total > 0:
                    progress_callback(frame_id, effective_total)

        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if frame_id == 0:
            raise RuntimeError("The uploaded video contains no readable frames")

        effective_total = frame_count if frame_count > 0 else frame_id
        if progress_callback is not None and effective_total > 0:
            progress_callback(effective_total, effective_total)

        tracks, summary = build_track_analytics(detections_payload, fps)
        summary["frame_errors"] = frame_errors

        if active_phase_entry is not None:
            heuristic_phase_timeline.append(active_phase_entry)
        cricket_analytics = self.analytics.build_full_analytics()
        if not cricket_analytics.get("timeline") and heuristic_phase_timeline:
            cricket_analytics["timeline"] = heuristic_phase_timeline
        score_timeline = self.scoreboard_ocr.get_score_timeline()
        team_summary = self.team_analyzer.get_team_summary()
        event_summary = (
            self.event_classifier.get_delivery_summary() if self.event_classifier else {}
        )
        visual_events = self.event_classifier.get_all_events() if self.event_classifier else []
        last_score = self.scoreboard_ocr.last_score
        transcript = self.audio_transcriber.transcribe(video_path, profile=profile_report.profile)
        cricket_package = build_cricket_package(
            heuristic_analytics=cricket_analytics,
            score_timeline=score_timeline,
            team_summary=team_summary,
            delivery_summary=event_summary,
            visual_events=visual_events,
            last_score=last_score,
            camera_cuts=self.cut_detector.total_cuts,
            ball_trajectory=self.ball_tracker.get_trajectory(),
            transcript=transcript.to_dict(),
            profile_report=profile_report.to_dict(),
            fps=fps,
            frame_width=width,
            frame_height=height,
        )

        return {
            "fps": fps,
            "frame_count": frame_count if frame_count > 0 else frame_id,
            "width": width,
            "height": height,
            "detections": detections_payload,
            "tracks": tracks,
            "summary": summary,
            "cricket": cricket_package,
        }

    @staticmethod
    def _normalize_tracked_objects(
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        tracked_objects: list[TrackedObject],
        profile: str = "batch",
    ) -> list[TrackedObject]:
        normalized: list[TrackedObject] = []
        seen_keys: set[tuple[str, int | None, tuple[float, float, float, float]]] = set()

        for item in tracked_objects:
            x1, y1, x2, y2 = item.bbox
            left, right = sorted((max(0.0, x1), min(float(frame_width), x2)))
            top, bottom = sorted((max(0.0, y1), min(float(frame_height), y2)))

            if right - left < settings.min_detection_box_size_px or bottom - top < settings.min_detection_box_size_px:
                continue

            if not VideoProcessor._is_plausible_detection(
                frame=frame,
                frame_width=frame_width,
                frame_height=frame_height,
                item=item,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                profile=profile,
            ):
                continue

            normalized_bbox = (
                round(left, 2),
                round(top, 2),
                round(right, 2),
                round(bottom, 2),
            )
            identity = (item.object_type, item.tracking_id, normalized_bbox)
            if identity in seen_keys:
                continue

            seen_keys.add(identity)
            normalized.append(
                TrackedObject(
                    object_type=item.object_type,
                    class_id=item.class_id,
                    bbox=list(normalized_bbox),
                    confidence=min(max(float(item.confidence), 0.0), 1.0),
                    tracking_id=item.tracking_id,
                    is_predicted=item.is_predicted,
                    appearance_signature=item.appearance_signature,
                )
            )

        return normalized

    def _reset_batch_modules(self, fps: float) -> None:
        self.cut_detector.reset()
        self.ball_tracker = SpatioTemporalBallTracker(fps=fps)
        self.scoreboard_ocr.reset()
        self.team_analyzer.reset()
        self.analytics.reset()
        self.profile_analyzer.reset()
        self.event_classifier = None

    @staticmethod
    def _batch_stride_for_fps(fps: float, target_fps: int, minimum: int = 1) -> int:
        safe_target = max(int(target_fps), 1)
        computed = int(round(max(fps, 1.0) / safe_target))
        return max(minimum, computed, 1)

    @staticmethod
    def _update_phase_timeline(
        timeline: list[dict[str, Any]],
        active_entry: dict[str, Any] | None,
        team_results: list[Any],
        person_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        frame_id: int,
        fps: float,
    ) -> dict[str, Any] | None:
        next_entry = VideoProcessor._build_phase_entry(
            team_results=team_results,
            person_detections=person_detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_id=frame_id,
            fps=fps,
        )
        if next_entry is None:
            if active_entry is not None:
                timeline.append(active_entry)
            return None

        if active_entry is None:
            return next_entry

        if active_entry.get("phase_key") == next_entry.get("phase_key"):
            active_entry["ts_end"] = next_entry["ts_end"]
            active_entry["detail"] = next_entry["detail"]
            return active_entry

        timeline.append(active_entry)
        return next_entry

    @staticmethod
    def _build_phase_entry(
        team_results: list[Any],
        person_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        frame_id: int,
        fps: float,
    ) -> dict[str, Any] | None:
        actioncam_entry = VideoProcessor._build_actioncam_phase_entry(
            person_detections=person_detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_id=frame_id,
            fps=fps,
        )
        if actioncam_entry is not None:
            return actioncam_entry

        role_counts: dict[str, int] = {}
        for player in team_results:
            role = str(getattr(player, "role", "fielder") or "fielder")
            role_counts[role] = role_counts.get(role, 0) + 1

        batter_count = role_counts.get("batter", 0)
        bowler_count = role_counts.get("bowler", 0)
        wicketkeeper_count = role_counts.get("wicketkeeper", 0)
        fielder_count = role_counts.get("fielder", 0)
        person_count = len(person_detections)

        if batter_count and bowler_count:
            phase_key = "batter-bowler"
            phase_summary = f"Approximate batter and bowler lanes are visible with {person_count} tracked players."
        elif batter_count:
            phase_key = "batter-setup"
            phase_summary = f"Approximate batting setup is visible with {person_count} tracked players."
        elif bowler_count:
            phase_key = "bowler-approach"
            phase_summary = f"Approximate bowler lane is visible with {person_count} tracked players."
        elif wicketkeeper_count and person_count >= 3:
            phase_key = "keeper-field"
            phase_summary = f"Wicketkeeper and field setup are visible with {person_count} tracked players."
        elif person_count >= 3 or fielder_count >= 2:
            phase_key = "field-set"
            phase_summary = f"Field setup is visible with {person_count} tracked players."
        else:
            return None

        timestamp_ms = int((max(frame_id, 0) / max(fps, 1.0)) * 1000)
        phase_bucket = max((frame_id // max(int(fps * 4), 1)) + 1, 1)
        return {
            "ball": phase_bucket,
            "over": f"0.{phase_bucket}",
            "runs": 0,
            "boundary": False,
            "six": False,
            "wicket": False,
            "dot": False,
            "shot": "unknown",
            "length": "unknown",
            "line": "unknown",
            "zone": 0,
            "phase_key": phase_key,
            "phase_summary": phase_summary,
            "detail": f"{batter_count} batter · {bowler_count} bowler · {wicketkeeper_count} keeper",
            "confidence": 0.52,
            "ts_start": timestamp_ms,
            "ts_end": timestamp_ms + 2400,
        }

    @staticmethod
    def _build_actioncam_phase_entry(
        person_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        frame_id: int,
        fps: float,
    ) -> dict[str, Any] | None:
        if len(person_detections) < 2:
            return None

        normalized_players: list[dict[str, float]] = []
        for detection in person_detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            normalized_players.append(
                {
                    "cx": (x1 + x2) / 2.0 / max(frame_width, 1),
                    "cy": (y1 + y2) / 2.0 / max(frame_height, 1),
                    "w": width / max(frame_width, 1),
                    "h": height / max(frame_height, 1),
                }
            )

        if len(normalized_players) < 2:
            return None

        crease_players = [
            player
            for player in normalized_players
            if 0.42 <= player["cx"] <= 0.62 and 0.36 <= player["cy"] <= 0.60 and 0.03 <= player["h"] <= 0.18
        ]
        support_players = [
            player
            for player in normalized_players
            if 0.30 <= player["cx"] <= 0.72 and 0.34 <= player["cy"] <= 0.68 and player["h"] <= 0.20
        ]
        wide_fielders = [
            player
            for player in normalized_players
            if player["cx"] <= 0.18 or player["cx"] >= 0.82
        ]
        close_runners = [
            player
            for player in normalized_players
            if 0.24 <= player["cx"] <= 0.82 and player["cy"] >= 0.38 and player["h"] >= 0.16
        ]

        if not crease_players and len(support_players) < 3:
            return None

        if close_runners:
            phase_key = "actioncam-bowler-approach"
            phase_summary = "Bowler movement is visible from the camera end while the striker holds the far crease."
            detail = "Camera-end run-up"
            confidence = 0.68
        elif len(crease_players) >= 2:
            phase_key = "actioncam-striker-ready"
            phase_summary = "Striker and wicket area are set at the far crease."
            detail = "Far-crease set"
            confidence = 0.64
        elif len(support_players) >= 3:
            phase_key = "actioncam-field-reset"
            phase_summary = "Field is resetting around the striker in the wide action-cam view."
            detail = "Field reset"
            confidence = 0.6
        else:
            phase_key = "actioncam-striker-hold"
            phase_summary = "Striker remains set at the far crease between deliveries."
            detail = "Striker hold"
            confidence = 0.58

        timestamp_ms = int((max(frame_id, 0) / max(fps, 1.0)) * 1000)
        phase_bucket = max((frame_id // max(int(fps * 4), 1)) + 1, 1)
        return {
            "ball": phase_bucket,
            "over": f"0.{phase_bucket}",
            "runs": 0,
            "boundary": False,
            "six": False,
            "wicket": False,
            "dot": False,
            "shot": "unknown",
            "length": "unknown",
            "line": "unknown",
            "zone": 0,
            "phase_key": phase_key,
            "phase_summary": phase_summary,
            "detail": detail,
            "confidence": confidence,
            "camera_profile": "action-cam-end-on",
            "ts_start": timestamp_ms,
            "ts_end": timestamp_ms + 2400,
        }

    @staticmethod
    def _retain_cricket_detections(tracked_objects: list[TrackedObject]) -> list[TrackedObject]:
        allowed_labels = set(settings.tracked_class_name_list)
        cricket_objects: list[TrackedObject] = []
        for item in tracked_objects:
            label = item.object_type.strip().lower()
            if label in allowed_labels or "bat" in label:
                cricket_objects.append(item)
        return cricket_objects

    @staticmethod
    def _is_plausible_detection(
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        item: TrackedObject,
        left: float,
        top: float,
        right: float,
        bottom: float,
        profile: str = "batch",
    ) -> bool:
        box_width = max(right - left, 0.0)
        box_height = max(bottom - top, 0.0)
        box_area = box_width * box_height
        if box_area <= 0:
            return False

        frame_area = max(float(frame_width * frame_height), 1.0)
        area_ratio = box_area / frame_area
        if area_ratio > VideoProcessor._max_area_ratio_for_label(item.object_type, profile=profile):
            return False

        if (
            box_width < VideoProcessor._min_box_size_for_label(item.object_type)
            or box_height < VideoProcessor._min_box_size_for_label(item.object_type)
        ):
            return False

        aspect_ratio = box_width / max(box_height, 1.0)
        min_aspect_ratio, max_aspect_ratio = VideoProcessor._aspect_ratio_bounds_for_label(
            item.object_type,
            profile=profile,
        )
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            return False

        if not item.is_predicted and item.confidence < VideoProcessor._min_confidence_for_label(item.object_type):
            return False

        if profile != "live" and VideoProcessor._overlaps_ignored_regions(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            return False

        if profile != "live" and VideoProcessor._is_low_confidence_edge_detection(
            object_type=item.object_type,
            confidence=item.confidence,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            return False

        if item.is_predicted:
            return True

        if item.confidence >= VideoProcessor._instant_confidence_for_label(item.object_type):
            return True

        return VideoProcessor._has_visual_support(frame, item.object_type, left, top, right, bottom)

    @staticmethod
    def _overlaps_ignored_regions(
        left: float,
        top: float,
        right: float,
        bottom: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        ignore_regions = [
            (0.0, 0.0, float(frame_width), frame_height * settings.overlay_ignore_top_ratio),
            (
                0.0,
                frame_height * (1.0 - settings.overlay_ignore_bottom_ratio),
                float(frame_width),
                float(frame_height),
            ),
            (0.0, 0.0, frame_width * settings.overlay_ignore_side_ratio, float(frame_height)),
            (
                frame_width * (1.0 - settings.overlay_ignore_side_ratio),
                0.0,
                float(frame_width),
                float(frame_height),
            ),
        ]
        box_area = max(right - left, 0.0) * max(bottom - top, 0.0)
        if box_area <= 0:
            return True

        for region_left, region_top, region_right, region_bottom in ignore_regions:
            inter_left = max(left, region_left)
            inter_top = max(top, region_top)
            inter_right = min(right, region_right)
            inter_bottom = min(bottom, region_bottom)
            overlap_area = max(inter_right - inter_left, 0.0) * max(inter_bottom - inter_top, 0.0)
            if overlap_area / box_area >= settings.overlay_overlap_threshold:
                return True
        return False

    @staticmethod
    def _has_visual_support(
        frame: np.ndarray,
        object_type: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> bool:
        x1 = max(int(round(left)), 0)
        y1 = max(int(round(top)), 0)
        x2 = max(int(round(right)), x1 + 1)
        y2 = max(int(round(bottom)), y1 + 1)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return False

        grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        intensity_std = float(grayscale.std())
        edges = cv2.Canny(grayscale, 80, 180)
        edge_ratio = float((edges > 0).mean())
        if VideoProcessor._is_small_object(object_type):
            return intensity_std >= 8.5 or edge_ratio >= 0.02
        return intensity_std >= 12.0 or edge_ratio >= 0.015

    @staticmethod
    def _filter_confirmed_tracks(
        tracked_objects: list[TrackedObject],
        track_hits: dict[tuple[str, int], int],
    ) -> list[TrackedObject]:
        confirmed: list[TrackedObject] = []

        for item in tracked_objects:
            if item.tracking_id is None:
                continue

            track_key = (item.object_type, item.tracking_id)
            hits = track_hits.get(track_key, 0) + 1
            track_hits[track_key] = hits

            if hits >= VideoProcessor._min_hits_for_label(item.object_type) or item.confidence >= VideoProcessor._instant_confidence_for_label(item.object_type):
                confirmed.append(item)

        return confirmed

    @staticmethod
    def _filter_live_tracks(
        tracked_objects: list[TrackedObject],
        track_hits: dict[tuple[str, int], int],
    ) -> list[TrackedObject]:
        confirmed: list[TrackedObject] = []

        for item in tracked_objects:
            if item.tracking_id is None:
                continue

            track_key = (item.object_type, item.tracking_id)
            hits = track_hits.get(track_key, 0) + 1
            track_hits[track_key] = hits

            required_hits = (
                2 if VideoProcessor._is_small_object(item.object_type) else max(settings.min_track_hits - 1, 2)
            )
            if item.is_predicted and hits < required_hits:
                continue

            if hits >= required_hits or item.confidence >= VideoProcessor._instant_confidence_for_label(item.object_type):
                confirmed.append(item)

        return confirmed

    @staticmethod
    def _serialize_detection(frame_id: int, fps: float, item: TrackedObject) -> dict[str, Any]:
        return {
            "frame_id": frame_id,
            "timestamp_ms": int((frame_id / fps) * 1000) if fps else 0,
            "object_type": item.object_type,
            "class_id": item.class_id,
            "bbox": [round(value, 2) for value in item.bbox],
            "confidence": round(item.confidence, 4),
            "tracking_id": item.tracking_id,
        }

    @staticmethod
    def _draw_annotation(frame: np.ndarray, item: TrackedObject) -> None:
        x1, y1, x2, y2 = [int(value) for value in item.bbox]
        color = VideoProcessor._color_for_label(item.object_type)
        label = f"{item.object_type.title()} ID {item.tracking_id if item.tracking_id is not None else 'NA'}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_height = text_height + baseline + 10
        label_width = text_width + 16
        label_left = min(max(x1, 0), max(frame.shape[1] - label_width, 0))
        label_top = max(y1 - label_height - 4, 0)
        if label_top == 0:
            label_top = min(y1 + 4, max(frame.shape[0] - label_height, 0))
        label_right = min(label_left + label_width, frame.shape[1])
        label_bottom = min(label_top + label_height, frame.shape[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (label_left + 8, label_bottom - baseline - 5),
            font,
            font_scale,
            (12, 18, 26),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _color_for_label(label: str) -> tuple[int, int, int]:
        seed = sum((index + 1) * ord(char) for index, char in enumerate(label))
        blue = 70 + (seed % 120)
        green = 90 + ((seed // 3) % 120)
        red = 80 + ((seed // 7) % 120)
        return int(blue), int(green), int(red)

    @staticmethod
    def _min_confidence_for_label(object_type: str) -> float:
        threshold = settings.class_confidence_threshold_map.get(object_type, settings.min_detection_confidence)
        if VideoProcessor._is_small_object(object_type):
            threshold = max(threshold, settings.small_object_track_confidence_threshold)
        return threshold

    @staticmethod
    def _instant_confidence_for_label(object_type: str) -> float:
        if VideoProcessor._is_small_object(object_type):
            return max(settings.instant_track_confidence, settings.small_object_track_confidence_threshold + 0.2)
        return settings.instant_track_confidence

    @staticmethod
    def _min_hits_for_label(object_type: str) -> int:
        if VideoProcessor._is_small_object(object_type):
            return max(settings.min_track_hits - 1, 2)
        return settings.min_track_hits

    @staticmethod
    def _min_box_size_for_label(object_type: str) -> float:
        return max(
            settings.class_min_box_size_map.get(object_type, settings.min_detection_box_size_px),
            settings.min_detection_box_size_px,
        )

    @staticmethod
    def _max_area_ratio_for_label(object_type: str, profile: str = "batch") -> float:
        base_ratio = min(
            settings.class_max_area_ratio_map.get(object_type, settings.max_detection_area_ratio),
            settings.max_detection_area_ratio,
        )
        if profile == "live":
            if VideoProcessor._is_small_object(object_type):
                return base_ratio
            return max(base_ratio, 0.78)
        return min(base_ratio, 0.9)

    @staticmethod
    def _aspect_ratio_bounds_for_label(
        object_type: str,
        profile: str = "batch",
    ) -> tuple[float, float]:
        base_bounds = settings.class_aspect_ratio_range_map.get(object_type, (0.16, 6.5))
        if profile == "live":
            if VideoProcessor._is_small_object(object_type):
                return base_bounds
            return (min(base_bounds[0], 0.12), max(base_bounds[1], 1.9))
        return base_bounds

    @staticmethod
    def _is_low_confidence_edge_detection(
        object_type: str,
        confidence: float,
        left: float,
        top: float,
        right: float,
        bottom: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        margin_x = frame_width * settings.edge_ignore_margin_ratio
        margin_y = frame_height * settings.edge_ignore_margin_ratio
        touches_edge = (
            left <= margin_x
            or top <= margin_y
            or right >= frame_width - margin_x
            or bottom >= frame_height - margin_y
        )
        if not touches_edge:
            return False
        if VideoProcessor._is_small_object(object_type):
            return confidence < max(settings.edge_low_confidence_reject_threshold, settings.small_object_track_confidence_threshold + 0.05)
        return confidence < settings.edge_low_confidence_reject_threshold

    @staticmethod
    def _is_small_object(object_type: str) -> bool:
        return object_type.strip().lower() in settings.small_object_class_name_list
