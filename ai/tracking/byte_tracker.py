from __future__ import annotations

from dataclasses import dataclass
import logging
from math import hypot

import numpy as np
import supervision as sv

from ai.detection.yolo_detector import DetectedObject
from backend.utils.config import settings


logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    object_type: str
    class_id: int
    bbox: list[float]
    confidence: float
    tracking_id: int | None
    is_predicted: bool = False
    appearance_signature: tuple[float, ...] | None = None


@dataclass
class TrackState:
    object_type: str
    class_id: int
    bbox: list[float]
    center: tuple[float, float]
    velocity: tuple[float, float]
    last_seen: int
    hit_count: int
    avg_confidence: float
    appearance_signature: tuple[float, ...] | None = None


class UniversalByteTracker:
    def __init__(self, frame_rate: float) -> None:
        self.frame_rate = int(max(frame_rate, 1))
        self.tracker = self._create_tracker()
        self.frame_index = 0
        self.next_fallback_id = 100000
        self.track_memory: dict[int, TrackState] = {}

    def _create_tracker(self) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=settings.tracker_activation_threshold,
            frame_rate=self.frame_rate,
        )

    def update(self, detections: list[DetectedObject]) -> list[TrackedObject]:
        self.frame_index += 1
        self._prune_stale_tracks()
        tracker_output = self._run_byte_track(detections)
        used_track_ids: set[int] = set()

        if tracker_output:
            return [
                self._finalize_tracked_item(item, used_track_ids)
                for item in tracker_output
            ]

        return [
            self._finalize_tracked_item(
                TrackedObject(
                    object_type=item.object_type,
                    class_id=item.class_id,
                    bbox=[float(value) for value in item.bbox],
                    confidence=float(item.confidence),
                    tracking_id=None,
                    is_predicted=False,
                    appearance_signature=item.appearance_signature,
                ),
                used_track_ids,
            )
            for item in detections
        ]

    def predict_only(self) -> list[TrackedObject]:
        self.frame_index += 1
        self._prune_stale_tracks()
        predicted_items: list[TrackedObject] = []

        for track_id in sorted(self.track_memory.keys()):
            metadata = self.track_memory.get(track_id)
            if metadata is None:
                continue

            frame_gap = max(self.frame_index - metadata.last_seen, 1)
            predicted_bbox = self._project_bbox(metadata.bbox, metadata.velocity, frame_gap)
            predicted_confidence = max(
                metadata.avg_confidence - min(frame_gap * 0.04, 0.2),
                0.0,
            )
            predicted_item = TrackedObject(
                object_type=metadata.object_type,
                class_id=metadata.class_id,
                bbox=predicted_bbox,
                confidence=predicted_confidence,
                tracking_id=track_id,
                is_predicted=True,
                appearance_signature=metadata.appearance_signature,
            )
            smoothed_bbox = self._smooth_bbox(track_id, predicted_item.bbox)
            finalized = TrackedObject(
                object_type=predicted_item.object_type,
                class_id=predicted_item.class_id,
                bbox=smoothed_bbox,
                confidence=predicted_item.confidence,
                tracking_id=track_id,
                is_predicted=True,
                appearance_signature=predicted_item.appearance_signature,
            )
            self._update_track_state(track_id, finalized)
            predicted_items.append(finalized)

        return predicted_items

    def has_active_tracks(self) -> bool:
        return bool(self.track_memory)

    def _run_byte_track(self, detections: list[DetectedObject]) -> list[TrackedObject]:
        if detections:
            xyxy = np.array([item.bbox for item in detections], dtype=np.float32)
            confidence = np.array([item.confidence for item in detections], dtype=np.float32)
            class_ids = np.array([item.class_id for item in detections], dtype=np.int32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.array([], dtype=np.float32)
            class_ids = np.array([], dtype=np.int32)

        tracker_input = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )
        try:
            tracked = self.tracker.update_with_detections(tracker_input)
        except Exception:
            logger.exception("ByteTrack update failed; using local fallback tracking")
            return []

        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=object)
        class_lookup = {item.class_id: item.object_type for item in detections}
        results: list[TrackedObject] = []

        for index, bbox in enumerate(tracked.xyxy):
            class_id = int(tracked.class_id[index]) if tracked.class_id is not None else -1
            tracker_id = int(tracker_ids[index]) if len(tracker_ids) > index and tracker_ids[index] is not None else None
            confidence_value = float(tracked.confidence[index]) if tracked.confidence is not None else 0.0
            bbox_values = [float(value) for value in bbox.tolist()]
            results.append(
                TrackedObject(
                    object_type=class_lookup.get(class_id, f"class_{class_id}"),
                    class_id=class_id,
                    bbox=bbox_values,
                    confidence=confidence_value,
                    tracking_id=tracker_id,
                    appearance_signature=self._match_detection_signature(class_id, bbox_values, detections),
                )
            )

        return results

    def _finalize_tracked_item(
        self,
        item: TrackedObject,
        used_track_ids: set[int],
    ) -> TrackedObject:
        tracking_id = self._resolve_tracking_id(item, used_track_ids)
        smoothed_bbox = self._smooth_bbox(tracking_id, item.bbox)
        finalized = TrackedObject(
            object_type=item.object_type,
            class_id=item.class_id,
            bbox=smoothed_bbox,
            confidence=item.confidence,
            tracking_id=tracking_id,
            is_predicted=item.is_predicted,
            appearance_signature=item.appearance_signature,
        )
        self._update_track_state(tracking_id, finalized)
        used_track_ids.add(tracking_id)
        return finalized

    def _resolve_tracking_id(
        self,
        item: TrackedObject,
        used_track_ids: set[int],
    ) -> int:
        if (
            item.tracking_id is not None
            and item.tracking_id not in used_track_ids
            and self._is_tracker_assignment_consistent(item.tracking_id, item)
        ):
            return item.tracking_id

        candidate_track_id = self._match_existing_track(item, used_track_ids)
        if candidate_track_id is not None:
            return candidate_track_id

        candidate_track_id = self.next_fallback_id
        self.next_fallback_id += 1
        return candidate_track_id

    def _match_existing_track(
        self,
        item: TrackedObject,
        used_track_ids: set[int],
    ) -> int | None:
        best_track_id: int | None = None
        best_score = float("-inf")
        center = self._bbox_center(item.bbox)
        item_area = self._bbox_area(item.bbox)

        for track_id, metadata in self.track_memory.items():
            if track_id in used_track_ids:
                continue
            if metadata.class_id != item.class_id:
                continue

            frame_gap = max(self.frame_index - metadata.last_seen, 1)
            predicted_center = (
                metadata.center[0] + metadata.velocity[0] * frame_gap,
                metadata.center[1] + metadata.velocity[1] * frame_gap,
            )
            distance = hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
            iou = self._bbox_iou(metadata.bbox, item.bbox)
            max_step = self._allowed_step(item.object_type, metadata.bbox, item.bbox, frame_gap)
            if distance > max_step and iou < settings.tracker_min_iou:
                continue

            previous_area = self._bbox_area(metadata.bbox)
            size_ratio = max(previous_area, item_area) / max(min(previous_area, item_area), 1.0)
            if size_ratio > settings.max_tracker_size_ratio_delta:
                continue

            motion_penalty = self._motion_inconsistency_penalty(metadata, center, frame_gap)
            appearance_similarity = self._appearance_similarity(
                metadata.appearance_signature,
                item.appearance_signature,
            )
            if (
                metadata.appearance_signature is not None
                and item.appearance_signature is not None
                and (
                    appearance_similarity < (settings.tracker_appearance_reject_threshold * 0.45)
                    or (
                        appearance_similarity < settings.tracker_appearance_reject_threshold
                        and iou < max(settings.tracker_min_iou, 0.16)
                    )
                )
            ):
                continue

            score = (
                iou * 5.0
                - (distance / max(max_step, 1.0))
                - (frame_gap - 1) * 0.08
                - motion_penalty
                + appearance_similarity * settings.tracker_appearance_similarity_bonus
                + min(metadata.hit_count, 10) * 0.07
                + metadata.avg_confidence * 0.35
            )
            if score > best_score:
                best_track_id = track_id
                best_score = score

        return best_track_id

    def _smooth_bbox(self, track_id: int, bbox: list[float]) -> list[float]:
        previous = self.track_memory.get(track_id)
        if previous is None:
            return [round(float(value), 2) for value in bbox]

        alpha = settings.tracker_box_smoothing_factor
        if self._is_small_object(previous.object_type):
            alpha *= 0.75
        if previous.hit_count <= 2:
            alpha = min(alpha + 0.08, 0.45)
        smoothed = [
            round((previous_value * (1.0 - alpha)) + (current_value * alpha), 2)
            for previous_value, current_value in zip(previous.bbox, bbox)
        ]
        return smoothed

    def _update_track_state(self, track_id: int, item: TrackedObject) -> None:
        center = self._bbox_center(item.bbox)
        existing = self.track_memory.get(track_id)
        if existing is None:
            velocity = (0.0, 0.0)
            hit_count = 1
            avg_confidence = item.confidence
        else:
            frame_gap = max(self.frame_index - existing.last_seen, 1)
            velocity = (
                (center[0] - existing.center[0]) / frame_gap,
                (center[1] - existing.center[1]) / frame_gap,
            )
            hit_count = existing.hit_count + 1
            avg_confidence = ((existing.avg_confidence * existing.hit_count) + item.confidence) / hit_count

        self.track_memory[track_id] = TrackState(
            object_type=item.object_type,
            class_id=item.class_id,
            bbox=[float(value) for value in item.bbox],
            center=center,
            velocity=velocity,
            last_seen=self.frame_index,
            hit_count=hit_count,
            avg_confidence=avg_confidence,
            appearance_signature=self._resolve_appearance_signature(existing, item),
        )

    def _resolve_appearance_signature(
        self,
        existing: TrackState | None,
        item: TrackedObject,
    ) -> tuple[float, ...] | None:
        if existing is None:
            return item.appearance_signature
        if item.appearance_signature is None:
            return existing.appearance_signature
        if existing.appearance_signature is None:
            return item.appearance_signature
        return self._blend_appearance_signature(existing.appearance_signature, item.appearance_signature)

    @staticmethod
    def _blend_appearance_signature(
        previous_signature: tuple[float, ...],
        current_signature: tuple[float, ...],
    ) -> tuple[float, ...]:
        if len(previous_signature) != len(current_signature):
            return current_signature

        alpha = settings.tracker_appearance_blend_alpha
        blended = np.array(
            [
                (left * (1.0 - alpha)) + (right * alpha)
                for left, right in zip(previous_signature, current_signature)
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(blended))
        if norm <= 0:
            return current_signature
        blended /= norm
        return tuple(float(value) for value in blended.tolist())

    @classmethod
    def _appearance_similarity(
        cls,
        left_signature: tuple[float, ...] | None,
        right_signature: tuple[float, ...] | None,
    ) -> float:
        if left_signature is None or right_signature is None:
            return 0.0
        if len(left_signature) != len(right_signature):
            return 0.0
        similarity = float(np.dot(np.array(left_signature), np.array(right_signature)))
        return max(min(similarity, 1.0), 0.0)

    @classmethod
    def _match_detection_signature(
        cls,
        class_id: int,
        bbox: list[float],
        detections: list[DetectedObject],
    ) -> tuple[float, ...] | None:
        best_match: DetectedObject | None = None
        best_iou = 0.0
        for detection in detections:
            if detection.class_id != class_id:
                continue
            iou = cls._bbox_iou(detection.bbox, bbox)
            if iou > best_iou:
                best_match = detection
                best_iou = iou
        if best_match is None:
            return None
        return best_match.appearance_signature

    def _prune_stale_tracks(self) -> None:
        stale_track_ids = [
            track_id
            for track_id, metadata in self.track_memory.items()
            if self.frame_index - metadata.last_seen > self._stale_frame_limit(metadata.object_type)
        ]
        for track_id in stale_track_ids:
            self.track_memory.pop(track_id, None)

    def _is_tracker_assignment_consistent(self, track_id: int, item: TrackedObject) -> bool:
        metadata = self.track_memory.get(track_id)
        if metadata is None:
            return True
        if metadata.class_id != item.class_id:
            return False

        frame_gap = max(self.frame_index - metadata.last_seen, 1)
        predicted_center = (
            metadata.center[0] + metadata.velocity[0] * frame_gap,
            metadata.center[1] + metadata.velocity[1] * frame_gap,
        )
        center = self._bbox_center(item.bbox)
        distance = hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
        iou = self._bbox_iou(metadata.bbox, item.bbox)
        max_step = self._allowed_step(item.object_type, metadata.bbox, item.bbox, frame_gap)
        if distance > (max_step * settings.tracker_hard_reject_distance_multiplier) and iou < settings.tracker_min_iou:
            return False

        previous_area = self._bbox_area(metadata.bbox)
        current_area = self._bbox_area(item.bbox)
        size_ratio = max(previous_area, current_area) / max(min(previous_area, current_area), 1.0)
        if size_ratio > settings.max_tracker_size_ratio_delta:
            return False

        appearance_similarity = self._appearance_similarity(
            metadata.appearance_signature,
            item.appearance_signature,
        )
        if (
            metadata.appearance_signature is not None
            and item.appearance_signature is not None
            and (
                appearance_similarity < (settings.tracker_appearance_reject_threshold * 0.45)
                or (
                    appearance_similarity < settings.tracker_appearance_reject_threshold
                    and iou < max(settings.tracker_min_iou, 0.14)
                )
            )
        ):
            return False

        return self._motion_inconsistency_penalty(metadata, center, frame_gap) < settings.tracker_motion_inconsistency_penalty

    def _allowed_step(
        self,
        object_type: str,
        previous_bbox: list[float],
        current_bbox: list[float],
        frame_gap: int,
    ) -> float:
        multiplier = settings.small_object_tracker_step_multiplier if self._is_small_object(object_type) else 1.0
        return max(
            settings.tracker_step_px * multiplier * frame_gap,
            max(self._bbox_scale(previous_bbox), self._bbox_scale(current_bbox)) * (2.8 if multiplier > 1.0 else 2.4),
        )

    def _motion_inconsistency_penalty(
        self,
        metadata: TrackState,
        current_center: tuple[float, float],
        frame_gap: int,
    ) -> float:
        velocity_x, velocity_y = metadata.velocity
        velocity_norm = hypot(velocity_x, velocity_y)
        if velocity_norm <= 1.0:
            return 0.0

        actual_vector = (
            (current_center[0] - metadata.center[0]) / frame_gap,
            (current_center[1] - metadata.center[1]) / frame_gap,
        )
        actual_norm = hypot(actual_vector[0], actual_vector[1])
        if actual_norm <= 1.0:
            return 0.0

        alignment = ((velocity_x * actual_vector[0]) + (velocity_y * actual_vector[1])) / max(velocity_norm * actual_norm, 1.0)
        if alignment >= 0.15:
            return 0.0

        return (0.15 - alignment) * settings.tracker_motion_inconsistency_penalty

    def _stale_frame_limit(self, object_type: str) -> int:
        if self._is_small_object(object_type):
            return int(round(settings.tracker_max_stale_frames * settings.small_object_stale_track_multiplier))
        return settings.tracker_max_stale_frames

    @staticmethod
    def _is_small_object(object_type: str) -> bool:
        return object_type.strip().lower() in settings.small_object_class_name_list

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _bbox_area(bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    @staticmethod
    def _bbox_scale(bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, y2 - y1, 1.0)

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

        union_area = UniversalByteTracker._bbox_area(left_bbox) + UniversalByteTracker._bbox_area(right_bbox) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _project_bbox(
        bbox: list[float],
        velocity: tuple[float, float],
        frame_gap: int,
    ) -> list[float]:
        shift_x = velocity[0] * frame_gap
        shift_y = velocity[1] * frame_gap
        x1, y1, x2, y2 = bbox
        return [
            round(x1 + shift_x, 2),
            round(y1 + shift_y, 2),
            round(x2 + shift_x, 2),
            round(y2 + shift_y, 2),
        ]

    def reset(self) -> None:
        self.tracker = self._create_tracker()
        self.track_memory.clear()


CricketByteTracker = UniversalByteTracker
