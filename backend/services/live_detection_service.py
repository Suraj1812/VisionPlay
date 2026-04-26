from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
import time
import uuid
from typing import Any

import cv2
import numpy as np

from ai.tracking.byte_tracker import TrackedObject, UniversalByteTracker
from backend.services.auxiliary_vision_service import AuxiliaryVisionResult, auxiliary_vision_service
from backend.services.processing_service import processing_service
from backend.utils.config import settings


@dataclass
class LiveDetectionSession:
    tracker: UniversalByteTracker
    lock: Any = field(default_factory=Lock, repr=False)
    track_hits: dict[tuple[str, int], int] = field(default_factory=dict)
    derived_tracks: dict[int, "DerivedLiveTrack"] = field(default_factory=dict)
    next_derived_tracking_id: int = 500000
    frame_index: int = 0
    last_seen_at: float = field(default_factory=time.monotonic)
    last_completed_at: float = 0.0
    smoothed_processing_latency_ms: float = 0.0
    smoothed_detection_fps: float = 0.0
    last_auxiliary_result: AuxiliaryVisionResult = field(default_factory=AuxiliaryVisionResult)
    last_auxiliary_frame_index: int = 0


@dataclass
class DerivedLiveTrack:
    object_type: str
    bbox: list[float]
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)
    last_seen_frame: int = 0
    hit_count: int = 1


class LiveDetectionService:
    HANDHELD_LABEL = "handheld_item"
    PHONE_LABEL = "phone_like_device"
    FACE_LABEL = "face"
    HAND_LABEL = "hand"

    def __init__(self) -> None:
        self._sessions: dict[str, LiveDetectionSession] = {}
        self._lock = Lock()
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._profile_face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._smile_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def detect_frame_bytes(self, payload: bytes, session_id: str | None = None) -> dict[str, object]:
        started_at = time.perf_counter()
        frame = self._decode_frame(payload)
        session_key = self._resolve_session_id(session_id)
        session = self._get_or_create_session(session_key)
        with session.lock:
            session.frame_index += 1
            session.last_seen_at = time.monotonic()

            objects = processing_service.processor.analyze_live_frame(
                frame=frame,
                tracker=session.tracker,
                frame_id=session.frame_index,
                track_hits=session.track_hits,
            )
            auxiliary_result = self._get_auxiliary_result(
                session=session,
                frame=frame,
            )
            objects = self._augment_live_objects(frame, objects, auxiliary_result=auxiliary_result)
            objects = self._stabilize_derived_objects(
                session=session,
                objects=objects,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            focus_item = self._select_focus_item(objects, frame_width=frame.shape[1], frame_height=frame.shape[0])
            object_counts = dict(sorted(Counter(item.object_type for item in objects).items()))
            lighting = self._classify_lighting(frame)
            face_reaction = self._infer_face_reaction(frame, objects, auxiliary_result=auxiliary_result)
            processing_latency_ms = (time.perf_counter() - started_at) * 1000.0
            metrics = self._update_session_metrics(session, processing_latency_ms)
            metrics.update(auxiliary_result.metrics)
            metrics["auxiliary_vision_enabled"] = auxiliary_vision_service.available
            metrics["processing_latency_ms"] = round(processing_latency_ms, 2)
            reactions = self._build_reactions(
                objects=objects,
                object_counts=object_counts,
                focus_item=focus_item,
                lighting=lighting,
                face_reaction=face_reaction,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                auxiliary_reactions=auxiliary_result.reactions,
            )

            return {
                "session_id": session_key,
                "frame_index": session.frame_index,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
                "focus_label": focus_item.object_type if focus_item is not None else None,
                "focus_tracking_id": focus_item.tracking_id if focus_item is not None else None,
                "lighting": lighting,
                "reactions": reactions,
                "object_counts": object_counts,
                "metrics": metrics,
                "objects": [
                    self._serialize_live_object(item, auxiliary_result)
                    for item in objects
                ],
            }

    def _get_or_create_session(self, session_id: str) -> LiveDetectionSession:
        with self._lock:
            self._prune_stale_sessions()
            session = self._sessions.get(session_id)
            if session is not None:
                return session

            session = LiveDetectionSession(
                tracker=UniversalByteTracker(frame_rate=settings.live_tracker_frame_rate),
            )
            self._sessions[session_id] = session
            return session

    def _prune_stale_sessions(self) -> None:
        now = time.monotonic()
        stale_keys = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen_at > settings.live_session_ttl_seconds
        ]
        for session_id in stale_keys:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _build_auxiliary_timestamp_ms(session: LiveDetectionSession) -> int:
        frame_rate = max(session.tracker.frame_rate, 1.0)
        return int((session.frame_index / frame_rate) * 1000)

    def _get_auxiliary_result(
        self,
        session: LiveDetectionSession,
        frame: np.ndarray,
    ) -> AuxiliaryVisionResult:
        should_refresh = (
            session.last_auxiliary_frame_index <= 0
            or session.frame_index - session.last_auxiliary_frame_index >= settings.live_aux_frame_stride
            or (
                not session.last_auxiliary_result.faces
                and not session.last_auxiliary_result.hands
            )
        )
        if not should_refresh:
            return session.last_auxiliary_result

        auxiliary_result = auxiliary_vision_service.analyze_frame(
            frame,
            timestamp_ms=self._build_auxiliary_timestamp_ms(session),
        )
        session.last_auxiliary_result = auxiliary_result
        session.last_auxiliary_frame_index = session.frame_index
        return auxiliary_result

    @staticmethod
    def _decode_frame(payload: bytes) -> np.ndarray:
        if not payload:
            raise ValueError("Camera frame payload is empty")

        buffer = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Unable to decode live camera frame")
        return frame

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str:
        if session_id and session_id.strip():
            return session_id.strip()
        return uuid.uuid4().hex

    def _augment_live_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult,
    ) -> list[TrackedObject]:
        augmented = self._normalize_live_labels(objects)
        augmented = self._append_auxiliary_face_objects(augmented, auxiliary_result)
        augmented = self._append_auxiliary_hand_objects(augmented, auxiliary_result)
        recovered_item = self._recover_handheld_item(
            frame,
            augmented,
            auxiliary_result=auxiliary_result,
        )
        if recovered_item is None:
            return self._simplify_live_objects(
                frame,
                self._append_face_objects(frame, augmented),
                auxiliary_result=auxiliary_result,
            )

        for item in augmented:
            if self._bbox_iou(item.bbox, recovered_item.bbox) >= 0.7:
                return self._simplify_live_objects(
                    frame,
                    self._append_face_objects(frame, augmented),
                    auxiliary_result=auxiliary_result,
                )

        augmented.append(recovered_item)
        return self._simplify_live_objects(
            frame,
            self._append_face_objects(frame, augmented),
            auxiliary_result=auxiliary_result,
        )

    def _stabilize_derived_objects(
        self,
        session: LiveDetectionSession,
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> list[TrackedObject]:
        if not objects:
            predicted = self._build_predicted_derived_objects(
                session=session,
                matched_track_ids=set(),
            )
            self._prune_derived_tracks(session)
            return predicted

        stabilized: list[TrackedObject] = []
        matched_track_ids: set[int] = set()

        for item in objects:
            if item.tracking_id is not None or item.object_type not in self._derived_live_labels():
                stabilized.append(item)
                continue

            track_id = self._match_derived_track(
                session=session,
                item=item,
                matched_track_ids=matched_track_ids,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            track = session.derived_tracks.get(track_id)

            if track is None:
                session.derived_tracks[track_id] = DerivedLiveTrack(
                    object_type=item.object_type,
                    bbox=[round(float(value), 2) for value in item.bbox],
                    confidence=float(item.confidence),
                    last_seen_frame=session.frame_index,
                )
                stabilized.append(
                    TrackedObject(
                        object_type=item.object_type,
                        class_id=item.class_id,
                        bbox=[round(float(value), 2) for value in item.bbox],
                        confidence=item.confidence,
                        tracking_id=track_id,
                        is_predicted=item.is_predicted,
                        appearance_signature=item.appearance_signature,
                    )
                )
                matched_track_ids.add(track_id)
                continue

            previous_center = self._bbox_center(track.bbox)
            smoothed_bbox = self._blend_bbox(track.bbox, item.bbox, settings.live_aux_track_smoothing_factor)
            current_center = self._bbox_center(smoothed_bbox)
            frame_gap = max(session.frame_index - track.last_seen_frame, 1)
            velocity = (
                (current_center[0] - previous_center[0]) / frame_gap,
                (current_center[1] - previous_center[1]) / frame_gap,
            )
            track.object_type = item.object_type
            track.bbox = smoothed_bbox
            track.confidence = float(item.confidence)
            track.velocity = velocity
            track.last_seen_frame = session.frame_index
            track.hit_count += 1
            matched_track_ids.add(track_id)
            stabilized.append(
                TrackedObject(
                    object_type=item.object_type,
                    class_id=item.class_id,
                    bbox=smoothed_bbox,
                    confidence=item.confidence,
                    tracking_id=track_id,
                    is_predicted=item.is_predicted,
                    appearance_signature=item.appearance_signature,
                )
            )

        predicted = self._build_predicted_derived_objects(
            session=session,
            matched_track_ids=matched_track_ids,
        )
        self._prune_derived_tracks(session)
        return stabilized + predicted

    def _build_predicted_derived_objects(
        self,
        session: LiveDetectionSession,
        matched_track_ids: set[int],
    ) -> list[TrackedObject]:
        predicted: list[TrackedObject] = []
        for track_id, track in session.derived_tracks.items():
            if track_id in matched_track_ids:
                continue

            frame_gap = session.frame_index - track.last_seen_frame
            if frame_gap <= 0 or frame_gap > settings.live_aux_track_ttl_frames:
                continue
            if track.hit_count < settings.live_aux_track_min_hits:
                continue

            predicted_bbox = self._project_bbox(track.bbox, track.velocity, frame_gap)
            confidence = max(track.confidence - (frame_gap * 0.08), 0.18)
            predicted.append(
                TrackedObject(
                    object_type=track.object_type,
                    class_id=-3,
                    bbox=predicted_bbox,
                    confidence=round(confidence, 4),
                    tracking_id=track_id,
                    is_predicted=True,
                )
            )
        return predicted

    def _match_derived_track(
        self,
        session: LiveDetectionSession,
        item: TrackedObject,
        matched_track_ids: set[int],
        frame_width: int,
        frame_height: int,
    ) -> int:
        best_track_id: int | None = None
        best_score = float("-inf")
        max_center_distance = self._frame_diagonal(frame_width, frame_height) * settings.live_aux_track_center_distance_ratio
        item_center = self._bbox_center(item.bbox)

        for track_id, track in session.derived_tracks.items():
            if track_id in matched_track_ids or track.object_type != item.object_type:
                continue

            frame_gap = max(session.frame_index - track.last_seen_frame, 1)
            if frame_gap > settings.live_aux_track_ttl_frames:
                continue

            projected_bbox = self._project_bbox(track.bbox, track.velocity, frame_gap)
            iou = self._bbox_iou(projected_bbox, item.bbox)
            center_distance = self._center_distance(self._bbox_center(projected_bbox), item_center)
            if iou < settings.live_aux_track_min_iou and center_distance > max_center_distance:
                continue

            score = (iou * 1.6) + (1.0 - min(center_distance / max(max_center_distance, 1.0), 1.0))
            score += min(track.hit_count * 0.08, 0.3)
            score += max(0.0, 0.12 - (frame_gap * 0.03))
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is not None:
            return best_track_id

        track_id = session.next_derived_tracking_id
        session.next_derived_tracking_id += 1
        return track_id

    def _prune_derived_tracks(self, session: LiveDetectionSession) -> None:
        stale_ids = [
            track_id
            for track_id, track in session.derived_tracks.items()
            if session.frame_index - track.last_seen_frame > settings.live_aux_track_ttl_frames
        ]
        for track_id in stale_ids:
            session.derived_tracks.pop(track_id, None)

    def _append_auxiliary_face_objects(
        self,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult,
    ) -> list[TrackedObject]:
        augmented = list(objects)
        if any(item.object_type == self.FACE_LABEL for item in augmented):
            return augmented
        for face in auxiliary_result.faces:
            face_object = TrackedObject(
                object_type=self.FACE_LABEL,
                class_id=-1,
                bbox=list(face.bbox),
                confidence=face.confidence,
                tracking_id=None,
                is_predicted=False,
            )
            if any(self._bbox_iou(face_object.bbox, item.bbox) >= 0.72 for item in augmented):
                continue
            augmented.append(face_object)
        return augmented

    def _append_auxiliary_hand_objects(
        self,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult,
    ) -> list[TrackedObject]:
        augmented = list(objects)
        for hand in auxiliary_result.hands:
            has_meaningful_gesture = self._has_meaningful_gesture(hand.gesture)
            if not has_meaningful_gesture and hand.confidence < 0.5:
                continue

            hand_object = TrackedObject(
                object_type=self.HAND_LABEL,
                class_id=-2,
                bbox=list(hand.bbox),
                confidence=hand.confidence,
                tracking_id=None,
                is_predicted=False,
            )
            if any(
                item.object_type == self.HAND_LABEL and self._bbox_iou(hand_object.bbox, item.bbox) >= 0.78
                for item in augmented
            ):
                continue
            augmented.append(hand_object)
        return augmented

    def _normalize_live_labels(
        self,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        normalized: list[TrackedObject] = []
        for item in objects:
            object_type = item.object_type
            if object_type == "cell phone":
                object_type = self.PHONE_LABEL
            normalized.append(
                TrackedObject(
                    object_type=object_type,
                    class_id=item.class_id,
                    bbox=list(item.bbox),
                    confidence=item.confidence,
                    tracking_id=item.tracking_id,
                    is_predicted=item.is_predicted,
                    appearance_signature=item.appearance_signature,
                )
            )
        return normalized

    def _append_face_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        augmented = list(objects)
        for face_object in self._detect_face_objects(frame, augmented):
            if any(self._bbox_iou(face_object.bbox, item.bbox) >= 0.68 for item in augmented):
                continue
            augmented.append(face_object)
        return augmented

    def _detect_face_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        if self._face_detector.empty() and self._profile_face_detector.empty():
            return []
        if any(item.object_type == self.FACE_LABEL for item in objects):
            return []

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)
        min_size = (
            max(int(frame.shape[1] * 0.08), 42),
            max(int(frame.shape[0] * 0.12), 42),
        )
        candidates = self._collect_face_candidates(grayscale, min_size=min_size)
        if len(candidates) == 0:
            return []

        frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
        person_boxes = [item.bbox for item in objects if item.object_type == "person"]
        face_objects: list[TrackedObject] = []

        for x, y, width, height in candidates:
            area_ratio = (width * height) / frame_area
            if area_ratio < 0.012 or area_ratio > 0.18:
                continue

            center_x = x + (width / 2.0)
            center_y = y + (height / 2.0)
            normalized_x = center_x / max(frame.shape[1], 1)
            normalized_y = center_y / max(frame.shape[0], 1)
            if not (0.18 <= normalized_x <= 0.88 and 0.12 <= normalized_y <= 0.92):
                continue

            aspect_ratio = max(width, height) / max(min(width, height), 1.0)
            if aspect_ratio > 1.45:
                continue

            bbox = [float(x), float(y), float(x + width), float(y + height)]
            if person_boxes and not any(self._bbox_iou(bbox, box) >= 0.02 for box in person_boxes):
                continue

            confidence = min(max((area_ratio * 8.5) + 0.38, 0.45), 0.92)
            face_objects.append(
                TrackedObject(
                    object_type=self.FACE_LABEL,
                    class_id=-1,
                    bbox=[round(value, 2) for value in bbox],
                    confidence=round(confidence, 4),
                    tracking_id=None,
                    is_predicted=False,
                )
            )

        face_objects.sort(key=lambda item: item.confidence, reverse=True)
        return face_objects[:2]

    def _collect_face_candidates(
        self,
        grayscale: np.ndarray,
        min_size: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        candidate_boxes: list[tuple[int, int, int, int]] = []

        if not self._face_detector.empty():
            candidate_boxes.extend(
                tuple(int(value) for value in candidate)
                for candidate in self._face_detector.detectMultiScale(
                    grayscale,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=min_size,
                )
            )

        if not self._profile_face_detector.empty():
            candidate_boxes.extend(
                tuple(int(value) for value in candidate)
                for candidate in self._profile_face_detector.detectMultiScale(
                    grayscale,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=min_size,
                )
            )

            flipped = cv2.flip(grayscale, 1)
            flipped_candidates = self._profile_face_detector.detectMultiScale(
                flipped,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=min_size,
            )
            width = grayscale.shape[1]
            for candidate in flipped_candidates:
                x, y, box_width, box_height = [int(value) for value in candidate]
                candidate_boxes.append((width - (x + box_width), y, box_width, box_height))

        deduped: list[tuple[int, int, int, int]] = []
        for candidate in candidate_boxes:
            bbox = [
                float(candidate[0]),
                float(candidate[1]),
                float(candidate[0] + candidate[2]),
                float(candidate[1] + candidate[3]),
            ]
            if any(
                self._bbox_iou(
                    bbox,
                    [
                        float(existing[0]),
                        float(existing[1]),
                        float(existing[0] + existing[2]),
                        float(existing[1] + existing[3]),
                    ],
                )
                >= 0.52
                for existing in deduped
            ):
                continue
            deduped.append(candidate)

        return deduped

    def _recover_handheld_item(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult,
    ) -> TrackedObject | None:
        if any(item.object_type in {self.HANDHELD_LABEL, self.PHONE_LABEL, "cell phone", "remote"} for item in objects):
            return None

        frame_height, frame_width = frame.shape[:2]
        primary_person = self._select_primary_person(objects, frame_width, frame_height)
        if primary_person is None:
            return None

        support_hand_boxes = [
            list(item.bbox)
            for item in objects
            if item.object_type == self.HAND_LABEL and item.confidence >= 0.52
        ]
        support_hand_boxes.extend(
            list(hand.bbox)
            for hand in auxiliary_result.hands
            if hand.confidence >= 0.52 or self._has_meaningful_gesture(hand.gesture)
        )
        if not support_hand_boxes:
            return None

        person_x1, person_y1, person_x2, person_y2 = primary_person.bbox
        search_regions: list[list[float]] = []
        for hand_bbox in support_hand_boxes:
            hand_x1, hand_y1, hand_x2, hand_y2 = hand_bbox
            hand_width = max(hand_x2 - hand_x1, 1.0)
            hand_height = max(hand_y2 - hand_y1, 1.0)
            center_x = (hand_x1 + hand_x2) / 2.0
            center_y = (hand_y1 + hand_y2) / 2.0
            region_width = max(hand_width * 2.8, 170.0)
            region_height = max(hand_height * 2.4, 170.0)
            search_regions.append(
                [
                    max(center_x - (region_width / 2.0), person_x1 - (hand_width * 0.45), 0.0),
                    max(center_y - (region_height / 2.0), person_y1 - (hand_height * 0.35), 0.0),
                    min(center_x + (region_width / 2.0), person_x2 + (hand_width * 0.45), float(frame_width)),
                    min(center_y + (region_height / 2.0), person_y2 + (hand_height * 0.35), float(frame_height)),
                ]
            )

        left = int(max(min(region[0] for region in search_regions), 0.0))
        top = int(max(min(region[1] for region in search_regions), 0.0))
        right = int(min(max(region[2] for region in search_regions), float(frame_width)))
        bottom = int(min(max(region[3] for region in search_regions), float(frame_height)))
        if right - left < 120 or bottom - top < 120:
            return None

        roi = frame[top:bottom, left:right]
        grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_threshold = int(np.clip(np.percentile(grayscale, 26), 55, 95))
        mask = cv2.inRange(grayscale, 0, dark_threshold)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        roi_height, roi_width = roi.shape[:2]
        roi_area = max(float(roi_height * roi_width), 1.0)
        roi_center = (roi_width / 2.0, roi_height / 2.0)
        best_bbox: list[float] | None = None
        best_label = self.HANDHELD_LABEL
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < roi_area * 0.012 or area > roi_area * 0.24:
                continue

            box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
            if min(box_width, box_height) < 72:
                continue
            if box_width > roi_width * 0.72 or box_height > roi_height * 0.72:
                continue

            aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
            if aspect_ratio < 1.2 or aspect_ratio > 2.22:
                continue

            fill_ratio = area / max(float(box_width * box_height), 1.0)
            if fill_ratio < 0.56:
                continue

            center_x = box_x + (box_width / 2.0)
            center_y = box_y + (box_height / 2.0)
            distance_to_center = ((center_x - roi_center[0]) ** 2 + (center_y - roi_center[1]) ** 2) ** 0.5
            max_distance = max((roi_center[0] ** 2 + roi_center[1] ** 2) ** 0.5, 1.0)
            center_bias = 1.0 - min(distance_to_center / max_distance, 1.0)
            if center_bias < 0.18:
                continue

            candidate_bbox = [
                float(left + box_x),
                float(top + box_y),
                float(left + box_x + box_width),
                float(top + box_y + box_height),
            ]
            candidate_area = self._bbox_area(candidate_bbox)
            if candidate_area / max(float(frame_width * frame_height), 1.0) > 0.12:
                continue

            touches_roi_edge = (
                box_x <= 6
                or box_y <= 6
                or box_x + box_width >= roi_width - 6
                or box_y + box_height >= roi_height - 6
            )
            if touches_roi_edge:
                continue

            max_hand_iou = max(
                (self._bbox_iou(candidate_bbox, hand_bbox) for hand_bbox in support_hand_boxes),
                default=0.0,
            )
            max_hand_center_bias = max(
                (
                    1.0
                    - min(
                        self._center_distance(
                            self._bbox_center(candidate_bbox),
                            self._bbox_center(hand_bbox),
                        )
                        / max(self._frame_diagonal(frame_width, frame_height) * 0.24, 1.0),
                        1.0,
                    )
                )
                for hand_bbox in support_hand_boxes
            )
            if max_hand_iou < 0.015 and max_hand_center_bias < 0.4:
                continue

            score = (
                (area / roi_area) * 1.15
                + fill_ratio * 0.72
                + center_bias * 0.28
                + max_hand_iou * 1.5
                + max_hand_center_bias * 0.9
                + (0.22 if 1.42 <= aspect_ratio <= 2.08 else 0.0)
            )
            if score > best_score:
                best_score = score
                best_label = self.PHONE_LABEL if 1.46 <= aspect_ratio <= 2.12 else self.HANDHELD_LABEL
                best_bbox = candidate_bbox

        if best_bbox is None or best_score < 1.42:
            return None

        return TrackedObject(
            object_type=best_label,
            class_id=-1,
            bbox=[round(value, 2) for value in best_bbox],
            confidence=round(min(max(best_score / 2.6, 0.52), 0.88), 4),
            tracking_id=None,
            is_predicted=False,
        )

    def _simplify_live_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult | None = None,
    ) -> list[TrackedObject]:
        if not objects:
            return objects

        frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
        derived_items = [
            item
            for item in objects
            if item.object_type in {self.HANDHELD_LABEL, self.PHONE_LABEL, self.FACE_LABEL, self.HAND_LABEL}
        ]
        if not derived_items:
            return objects

        filtered: list[TrackedObject] = []
        face_boxes = [
            item.bbox
            for item in objects
            if item.object_type == self.FACE_LABEL
        ]
        face_or_phone_boxes = [
            item.bbox
            for item in objects
            if item.object_type in {self.PHONE_LABEL, self.FACE_LABEL, self.HANDHELD_LABEL}
        ]
        phone_like_boxes = [
            item.bbox
            for item in objects
            if item.object_type in {self.PHONE_LABEL, self.HANDHELD_LABEL}
        ]
        for item in objects:
            if item.object_type == self.HAND_LABEL:
                matched_hand = (
                    self._match_auxiliary_hand(item.bbox, auxiliary_result)
                    if auxiliary_result is not None
                    else None
                )
                has_meaningful_gesture = self._has_meaningful_gesture(
                    matched_hand.gesture if matched_hand is not None else None
                )
                hand_confidence = matched_hand.confidence if matched_hand is not None else item.confidence
                hand_area_ratio = self._bbox_area(item.bbox) / frame_area
                max_phone_overlap = max((self._bbox_iou(item.bbox, other_bbox) for other_bbox in phone_like_boxes), default=0.0)
                max_face_overlap = max((self._bbox_iou(item.bbox, other_bbox) for other_bbox in face_boxes), default=0.0)
                hand_area = self._bbox_area(item.bbox)
                contains_focus_signal = any(
                    self._bbox_contains(item.bbox, other_bbox, tolerance=12.0)
                    and hand_area > self._bbox_area(other_bbox) * 1.55
                    for other_bbox in face_or_phone_boxes
                )
                overwhelms_phone = any(
                    self._bbox_iou(item.bbox, other_bbox) >= 0.5
                    and hand_area > self._bbox_area(other_bbox) * 1.18
                    for other_bbox in phone_like_boxes
                )
                overwhelms_face = any(
                    self._bbox_iou(item.bbox, other_bbox) >= 0.55
                    and hand_area > self._bbox_area(other_bbox) * 1.45
                    for other_bbox in face_boxes
                )

                if hand_area_ratio > 0.2 and not has_meaningful_gesture:
                    continue
                if contains_focus_signal and not has_meaningful_gesture and hand_confidence < 0.82:
                    continue
                if overwhelms_phone and not has_meaningful_gesture:
                    continue
                if overwhelms_face and not has_meaningful_gesture:
                    continue
                if max_phone_overlap >= 0.38 and not has_meaningful_gesture and hand_confidence < 0.74:
                    continue
                if max_face_overlap >= 0.48 and not has_meaningful_gesture and hand_confidence < 0.7:
                    continue
                filtered.append(item)
                continue

            if item.object_type != "person":
                filtered.append(item)
                continue

            left, top, right, bottom = item.bbox
            person_area = max((right - left) * (bottom - top), 1.0)
            person_area_ratio = person_area / frame_area
            if person_area_ratio < 0.34:
                filtered.append(item)
                continue

            contains_derived = [
                derived
                for derived in derived_items
                if self._bbox_contains(item.bbox, derived.bbox) or self._bbox_iou(item.bbox, derived.bbox) >= 0.08
            ]
            if not contains_derived:
                filtered.append(item)
                continue

            derived_cover_ratio = max(
                self._bbox_area(derived.bbox) / person_area for derived in contains_derived
            )
            has_face_or_phone = any(
                derived.object_type in {self.PHONE_LABEL, self.FACE_LABEL, self.HAND_LABEL} for derived in contains_derived
            )
            if has_face_or_phone and derived_cover_ratio >= 0.04:
                continue

            filtered.append(item)

        return filtered or objects

    @staticmethod
    def _select_primary_person(
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> TrackedObject | None:
        people = [item for item in objects if item.object_type == "person"]
        if not people:
            return None

        return LiveDetectionService._select_focus_item(people, frame_width=frame_width, frame_height=frame_height)

    @staticmethod
    def _select_focus_item(
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> TrackedObject | None:
        if not objects:
            return None

        frame_area = max(float(frame_width * frame_height), 1.0)
        frame_center = (frame_width / 2.0, frame_height / 2.0)
        best_item: TrackedObject | None = None
        best_score = float("-inf")

        for item in objects:
            x1, y1, x2, y2 = item.bbox
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            area_ratio = (width * height) / frame_area
            center_x = x1 + (width / 2.0)
            center_y = y1 + (height / 2.0)
            distance_to_center = ((center_x - frame_center[0]) ** 2 + (center_y - frame_center[1]) ** 2) ** 0.5
            max_distance = max((frame_center[0] ** 2 + frame_center[1] ** 2) ** 0.5, 1.0)
            center_bias = 1.0 - min(distance_to_center / max_distance, 1.0)
            score = (
                item.confidence * 1.25
                + min(area_ratio * 12.0, 0.9)
                + center_bias * 0.55
                + LiveDetectionService._focus_priority(item.object_type)
                + (0.12 if not item.is_predicted else 0.0)
            )
            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    @staticmethod
    def _classify_lighting(frame: np.ndarray) -> str:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(grayscale.mean())
        if brightness < 72:
            return "low"
        if brightness > 185:
            return "bright"
        return "balanced"

    @staticmethod
    def _build_reactions(
        objects: list[TrackedObject],
        object_counts: dict[str, int],
        focus_item: TrackedObject | None,
        lighting: str,
        face_reaction: str | None,
        frame_width: int,
        frame_height: int,
        auxiliary_reactions: list[str],
    ) -> list[str]:
        total_objects = sum(object_counts.values())
        reactions: list[str] = []

        if total_objects == 0:
            reactions.append("Scanning the scene")
        else:
            reactions.append(f"{total_objects} stable object{'s' if total_objects != 1 else ''} detected")

        if focus_item is not None:
            focus_label = LiveDetectionService._humanize_label(focus_item.object_type)
            if focus_item.tracking_id is not None:
                reactions.append(f"Focus locked on {focus_label} #{focus_item.tracking_id}")
            else:
                reactions.append(f"Focus locked on {focus_label}")

        if object_counts.get(LiveDetectionService.PHONE_LABEL, 0):
            reactions.append("Phone visible")
        elif object_counts.get(LiveDetectionService.HANDHELD_LABEL, 0):
            reactions.append("Handheld item visible")

        if object_counts.get(LiveDetectionService.FACE_LABEL, 0):
            reactions.append(
                f"Face looks {face_reaction.lower()}" if face_reaction else "Face detected"
            )
        if object_counts.get(LiveDetectionService.HAND_LABEL, 0) and not any(
            reaction.lower().startswith("gesture ") or "hand visible" in reaction.lower()
            for reaction in auxiliary_reactions
        ):
            reactions.append("Hand visible")

        if lighting == "low":
            reactions.append("Low light scene")
        elif lighting == "bright":
            reactions.append("High brightness scene")

        for reaction in auxiliary_reactions:
            if reaction not in reactions:
                reactions.append(reaction)

        return reactions[:5]

    @staticmethod
    def _focus_priority(object_type: str) -> float:
        if object_type == LiveDetectionService.PHONE_LABEL:
            return 0.82
        if object_type == LiveDetectionService.HANDHELD_LABEL:
            return 0.72
        if object_type == LiveDetectionService.FACE_LABEL:
            return 0.5
        if object_type == LiveDetectionService.HAND_LABEL:
            return 0.4
        if object_type in {"cell phone", "remote", "book", "bottle", "cup", "laptop"}:
            return 0.28
        if object_type == "person":
            return 0.02
        return 0.1

    @staticmethod
    def _humanize_label(object_type: str) -> str:
        labels = {
            LiveDetectionService.PHONE_LABEL: "Phone",
            LiveDetectionService.HANDHELD_LABEL: "Handheld",
            LiveDetectionService.FACE_LABEL: "Face",
            LiveDetectionService.HAND_LABEL: "Hand",
        }
        if object_type in labels:
            return labels[object_type]
        return object_type.replace("_", " ").title()

    def _infer_face_reaction(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        auxiliary_result: AuxiliaryVisionResult,
    ) -> str | None:
        if auxiliary_result.faces:
            emotion = auxiliary_result.faces[0].emotion.replace("_", " ").strip().title()
            return emotion or None
        faces = [item for item in objects if item.object_type == self.FACE_LABEL]
        if not faces:
            return None
        if self._smile_detector.empty():
            return None

        face = max(faces, key=lambda item: self._bbox_area(item.bbox))
        x1, y1, x2, y2 = [int(round(value)) for value in face.bbox]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        if x2 - x1 < 18 or y2 - y1 < 18:
            return "Face visible"

        face_roi = frame[y1:y2, x1:x2]
        grayscale = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)
        smile_candidates = self._smile_detector.detectMultiScale(
            grayscale,
            scaleFactor=1.7,
            minNeighbors=18,
            minSize=(max((x2 - x1) // 5, 14), max((y2 - y1) // 7, 10)),
        )
        if len(smile_candidates) > 0:
            return "Smiling"
        return "Neutral"

    def _update_session_metrics(
        self,
        session: LiveDetectionSession,
        processing_latency_ms: float,
    ) -> dict[str, float]:
        alpha = 0.3
        if session.smoothed_processing_latency_ms <= 0:
            session.smoothed_processing_latency_ms = processing_latency_ms
        else:
            session.smoothed_processing_latency_ms = (
                session.smoothed_processing_latency_ms * (1.0 - alpha)
            ) + (processing_latency_ms * alpha)

        completed_at = time.monotonic()
        if session.last_completed_at > 0:
            delta = max(completed_at - session.last_completed_at, 1e-4)
            instantaneous_fps = 1.0 / delta
            if session.smoothed_detection_fps <= 0:
                session.smoothed_detection_fps = instantaneous_fps
            else:
                session.smoothed_detection_fps = (
                    session.smoothed_detection_fps * (1.0 - alpha)
                ) + (instantaneous_fps * alpha)
        session.last_completed_at = completed_at

        return {
            "smoothed_processing_latency_ms": round(session.smoothed_processing_latency_ms, 2),
            "inference_fps": round(session.smoothed_detection_fps, 2),
        }

    def _serialize_live_object(
        self,
        item: TrackedObject,
        auxiliary_result: AuxiliaryVisionResult,
    ) -> dict[str, object]:
        details: dict[str, object] = {}
        if item.object_type == self.FACE_LABEL:
            matched_face = self._match_auxiliary_face(item.bbox, auxiliary_result)
            if matched_face is not None:
                details = {
                    "emotion": matched_face.emotion,
                    "smile_score": matched_face.smile_score,
                    "eye_openness": matched_face.eye_openness,
                    "head_pose": matched_face.head_pose,
                }
        elif item.object_type == self.HAND_LABEL:
            matched_hand = self._match_auxiliary_hand(item.bbox, auxiliary_result)
            if matched_hand is not None:
                normalized_gesture = (matched_hand.gesture or "").strip().lower()
                normalized_handedness = (matched_hand.handedness or "").strip().lower()
                if normalized_gesture not in {"", "none", "unknown"}:
                    details["gesture"] = matched_hand.gesture
                if normalized_handedness not in {"", "unknown"}:
                    details["handedness"] = matched_hand.handedness

        return {
            "object_type": item.object_type,
            "class_id": item.class_id,
            "bbox": [round(float(value), 2) for value in item.bbox],
            "confidence": round(float(item.confidence), 4),
            "tracking_id": item.tracking_id,
            "is_predicted": bool(item.is_predicted),
            "details": details,
        }

    def _match_auxiliary_face(
        self,
        bbox: list[float],
        auxiliary_result: AuxiliaryVisionResult,
    ):
        best_match = None
        best_iou = 0.0
        for face in auxiliary_result.faces:
            iou = self._bbox_iou(bbox, face.bbox)
            if iou > best_iou:
                best_match = face
                best_iou = iou
        return best_match

    def _match_auxiliary_hand(
        self,
        bbox: list[float],
        auxiliary_result: AuxiliaryVisionResult,
    ):
        best_match = None
        best_iou = 0.0
        for hand in auxiliary_result.hands:
            iou = self._bbox_iou(bbox, hand.bbox)
            if iou > best_iou:
                best_match = hand
                best_iou = iou
        return best_match

    @staticmethod
    def _has_meaningful_gesture(gesture: str | None) -> bool:
        normalized_gesture = (gesture or "").strip().lower()
        return normalized_gesture not in {"", "none", "unknown"}

    @classmethod
    def _derived_live_labels(cls) -> set[str]:
        return {
            cls.HANDHELD_LABEL,
            cls.PHONE_LABEL,
            cls.FACE_LABEL,
            cls.HAND_LABEL,
        }

    @staticmethod
    def _blend_bbox(previous_bbox: list[float], current_bbox: list[float], alpha: float) -> list[float]:
        clamped_alpha = max(0.0, min(alpha, 1.0))
        return [
            round((previous * (1.0 - clamped_alpha)) + (current * clamped_alpha), 2)
            for previous, current in zip(previous_bbox, current_bbox)
        ]

    @staticmethod
    def _project_bbox(
        bbox: list[float],
        velocity: tuple[float, float],
        frame_gap: int,
    ) -> list[float]:
        offset_x = velocity[0] * frame_gap
        offset_y = velocity[1] * frame_gap
        return [
            round(bbox[0] + offset_x, 2),
            round(bbox[1] + offset_y, 2),
            round(bbox[2] + offset_x, 2),
            round(bbox[3] + offset_y, 2),
        ]

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        return (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )

    @staticmethod
    def _frame_diagonal(frame_width: int, frame_height: int) -> float:
        return ((frame_width ** 2) + (frame_height ** 2)) ** 0.5

    @staticmethod
    def _center_distance(left_center: tuple[float, float], right_center: tuple[float, float]) -> float:
        return ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5

    @staticmethod
    def _has_edge_pressure(
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        margin_x = frame_width * 0.08
        margin_y = frame_height * 0.08
        for item in objects:
            left, top, right, bottom = item.bbox
            if left <= margin_x or top <= margin_y or right >= frame_width - margin_x or bottom >= frame_height - margin_y:
                return True
        return False

    @staticmethod
    def _bbox_iou(left_bbox: list[float], right_bbox: list[float]) -> float:
        left_x1, left_y1, left_x2, left_y2 = left_bbox
        right_x1, right_y1, right_x2, right_y2 = right_bbox

        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_area = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
        if inter_area <= 0:
            return 0.0

        left_area = max(left_x2 - left_x1, 0.0) * max(left_y2 - left_y1, 0.0)
        right_area = max(right_x2 - right_x1, 0.0) * max(right_y2 - right_y1, 0.0)
        union_area = left_area + right_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _bbox_area(bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    @staticmethod
    def _bbox_contains(outer_bbox: list[float], inner_bbox: list[float], tolerance: float = 8.0) -> bool:
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_bbox
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_bbox
        return (
            inner_x1 >= outer_x1 - tolerance
            and inner_y1 >= outer_y1 - tolerance
            and inner_x2 <= outer_x2 + tolerance
            and inner_y2 <= outer_y2 + tolerance
        )


live_detection_service = LiveDetectionService()
