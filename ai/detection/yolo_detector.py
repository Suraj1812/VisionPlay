from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


MPLCONFIGDIR = Path("/tmp/visionplay-matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO

from backend.utils.config import settings


logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    object_type: str
    class_id: int
    bbox: list[float]
    confidence: float
    appearance_signature: tuple[float, ...] | None = None


class YOLODetector:
    def __init__(self, model_path: str | None = None, confidence: float | None = None) -> None:
        self.model_path = model_path or settings.resolved_yolo_model_path
        self.confidence = confidence if confidence is not None else settings.yolo_confidence
        self.device = self._resolve_device()
        self.model: YOLO | None = None
        self._model_lock = Lock()
        self.class_names_by_id: dict[int, str] = {}
        self.allowed_class_ids: list[int] | None = None
        self.small_object_class_ids: list[int] = []

    def get_model(self) -> YOLO:
        if self.model is not None:
            return self.model

        with self._model_lock:
            if self.model is not None:
                return self.model

            try:
                self.model = YOLO(self.model_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load YOLO model '{self.model_path}'. "
                    "Provide a valid local weights path or allow the model to download on first run."
                ) from exc

            try:
                self.model.to(self.device)
            except Exception:
                logger.warning("Unable to move YOLO model to %s; falling back to CPU", self.device)
                self.device = "cpu"
                self.model.to(self.device)

            self._refresh_class_configuration()
        return self.model

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[DetectedObject]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model = self.get_model()
        full_frame_results = self._predict(
            model,
            source=rgb_frame,
            classes=self.allowed_class_ids,
            conf=self._inference_confidence(self.allowed_class_ids),
            imgsz=settings.yolo_image_size,
        )

        detections = self._parse_detections(full_frame_results[0], rgb_frame)
        if self._should_run_recall_boost(rgb_frame, detections):
            detections.extend(
                self._detect_recall_boost(
                    rgb_frame,
                    classes=self.allowed_class_ids,
                )
            )
        if self._should_run_region_refinement(frame_id, rgb_frame, detections):
            detections.extend(self._detect_region_refinement(rgb_frame, detections))
        if self._should_run_cricket_focus_regions(frame_id, rgb_frame, detections):
            detections.extend(self._detect_cricket_focus_regions(rgb_frame))
        if self._should_run_actioncam_ball_tiles(frame_id, rgb_frame, detections):
            detections.extend(self._detect_actioncam_ball_tiles(rgb_frame))
        if self._should_run_small_object_tiles(frame_id, detections):
            detections.extend(self._detect_small_object_tiles(rgb_frame))
        return self._merge_detections(detections)

    def _refresh_class_configuration(self) -> None:
        model = self.model
        if model is None:
            return

        raw_names = model.names if hasattr(model, "names") else {}
        self.class_names_by_id = self._normalize_names(raw_names)
        tracked_names = set(settings.tracked_class_name_list)
        small_object_names = set(settings.small_object_class_name_list)

        if tracked_names:
            self.allowed_class_ids = [
                class_id
                for class_id, class_name in self.class_names_by_id.items()
                if class_name in tracked_names
            ]
            if not self.allowed_class_ids:
                logger.warning(
                    "No YOLO classes matched tracked_class_names=%s; falling back to all classes",
                    settings.tracked_class_name_list,
                )
                self.allowed_class_ids = None
        else:
            self.allowed_class_ids = None

        allowed_ids = set(self.allowed_class_ids or self.class_names_by_id.keys())
        self.small_object_class_ids = [
            class_id
            for class_id, class_name in self.class_names_by_id.items()
            if class_id in allowed_ids and class_name in small_object_names
        ]

    def _detect_small_object_tiles(self, rgb_frame: np.ndarray) -> list[DetectedObject]:
        if settings.small_object_tile_grid_size <= 1 or not self.small_object_class_ids:
            return []

        frame_height, frame_width = rgb_frame.shape[:2]
        tile_boxes = self._build_tile_boxes(frame_width, frame_height)
        tile_sources: list[np.ndarray] = []
        valid_tile_boxes: list[tuple[int, int, int, int]] = []
        detections: list[DetectedObject] = []

        for left, top, right, bottom in tile_boxes:
            tile = rgb_frame[top:bottom, left:right]
            if tile.size == 0:
                continue
            tile_sources.append(tile)
            valid_tile_boxes.append((left, top, right, bottom))

        if not tile_sources:
            return detections

        tile_results = self._predict(
            self.get_model(),
            source=tile_sources,
            classes=self.small_object_class_ids,
            conf=self._inference_confidence(self.small_object_class_ids, tiled=True),
            imgsz=settings.small_object_tile_image_size,
        )
        for (left, top, right, bottom), tile_result in zip(valid_tile_boxes, tile_results):
            detections.extend(
                self._parse_detections(
                    tile_result,
                    rgb_frame[top:bottom, left:right],
                    offset=(left, top),
                )
            )

        return detections

    def _detect_recall_boost(
        self,
        rgb_frame: np.ndarray,
        classes: list[int] | None,
    ) -> list[DetectedObject]:
        enhanced_frame = self._prepare_recall_frame(rgb_frame)
        recall_results = self._predict(
            self.get_model(),
            source=enhanced_frame,
            classes=classes,
            conf=max(self._inference_confidence(classes) - 0.05, 0.06),
            imgsz=max(settings.yolo_image_size, 960),
        )
        return self._parse_detections(recall_results[0], enhanced_frame)

    def _should_run_region_refinement(
        self,
        frame_id: int | None,
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> bool:
        if not settings.adaptive_roi_refinement_enabled:
            return False
        if (
            frame_id is not None
            and settings.adaptive_roi_refinement_frame_stride > 1
            and frame_id % settings.adaptive_roi_refinement_frame_stride != 0
            and len(detections) > settings.adaptive_roi_refinement_sparse_detection_count
        ):
            return False
        if len(detections) <= settings.adaptive_roi_refinement_sparse_detection_count:
            return True
        if any(
            detection.confidence
            < min(self._class_confidence_threshold(detection.object_type) + 0.18, 0.88)
            for detection in detections
        ):
            return True

        grayscale = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        return float(grayscale.std()) < 38.0 and len(detections) < 6

    def _detect_region_refinement(
        self,
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> list[DetectedObject]:
        proposal_boxes = self._build_region_proposals(rgb_frame, detections)
        if not proposal_boxes:
            return []

        proposal_sources = [
            rgb_frame[top:bottom, left:right]
            for left, top, right, bottom in proposal_boxes
            if bottom - top >= 2 and right - left >= 2
        ]
        if not proposal_sources:
            return []

        proposal_results = self._predict(
            self.get_model(),
            source=proposal_sources,
            classes=self.allowed_class_ids,
            conf=max(self._inference_confidence(self.allowed_class_ids) - 0.08, 0.05),
            imgsz=settings.adaptive_roi_refinement_image_size,
        )

        refined: list[DetectedObject] = []
        for (left, top, right, bottom), region_result in zip(proposal_boxes, proposal_results):
            refined.extend(
                self._parse_detections(
                    region_result,
                    rgb_frame[top:bottom, left:right],
                    offset=(left, top),
                )
            )

        return refined

    def _should_run_cricket_focus_regions(
        self,
        frame_id: int | None,
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> bool:
        if not settings.cricket_focus_roi_enabled:
            return False

        frame_height, frame_width = rgb_frame.shape[:2]
        if frame_width < 1280 or frame_height < 720:
            return False

        person_count = sum(1 for detection in detections if detection.object_type == "person")
        has_ball = any(detection.object_type == "sports ball" for detection in detections)
        has_bat = any("bat" in detection.object_type for detection in detections)
        needs_help = (
            person_count <= 1
            or (person_count <= 3 and not has_ball and not has_bat)
            or (len(detections) <= 2 and not has_ball)
            or self._is_actioncam_crease_cluster(detections, frame_width, frame_height) and not has_ball and not has_bat
        )
        if not needs_help:
            return False

        if settings.cricket_focus_roi_frame_stride <= 1 or frame_id is None:
            return True
        return frame_id % settings.cricket_focus_roi_frame_stride == 0

    def _detect_cricket_focus_regions(self, rgb_frame: np.ndarray) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        focus_boxes = self._build_cricket_focus_boxes(rgb_frame.shape[1], rgb_frame.shape[0])
        region_sources: list[np.ndarray] = []
        valid_focus_boxes: list[tuple[int, int, int, int]] = []

        for left, top, right, bottom in focus_boxes:
            region = rgb_frame[top:bottom, left:right]
            if region.size == 0:
                continue
            region_sources.append(region)
            valid_focus_boxes.append((left, top, right, bottom))

        if not region_sources:
            return detections

        region_results = self._predict(
            self.get_model(),
            source=region_sources,
            classes=self.allowed_class_ids,
            conf=max(self._inference_confidence(self.allowed_class_ids) - 0.08, 0.08),
            imgsz=settings.cricket_focus_roi_image_size,
        )
        for (left, top, right, bottom), region_result in zip(valid_focus_boxes, region_results):
            detections.extend(
                self._parse_detections(
                    region_result,
                    rgb_frame[top:bottom, left:right],
                    offset=(left, top),
                )
            )

        return detections

    def _detect_actioncam_ball_tiles(self, rgb_frame: np.ndarray) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        if not self.small_object_class_ids:
            return detections

        tile_boxes = self._build_actioncam_ball_boxes(rgb_frame.shape[1], rgb_frame.shape[0])
        region_sources: list[np.ndarray] = []
        valid_tile_boxes: list[tuple[int, int, int, int]] = []

        for left, top, right, bottom in tile_boxes:
            region = rgb_frame[top:bottom, left:right]
            if region.size == 0:
                continue
            region_sources.append(region)
            valid_tile_boxes.append((left, top, right, bottom))

        if not region_sources:
            return detections

        region_results = self._predict(
            self.get_model(),
            source=region_sources,
            classes=self.small_object_class_ids,
            conf=max(self._inference_confidence(self.small_object_class_ids, tiled=True) - 0.06, 0.08),
            imgsz=max(settings.small_object_tile_image_size, 960),
        )
        for (left, top, right, bottom), region_result in zip(valid_tile_boxes, region_results):
            detections.extend(
                self._parse_detections(
                    region_result,
                    rgb_frame[top:bottom, left:right],
                    offset=(left, top),
                )
            )

        return detections

    @staticmethod
    def _build_cricket_focus_boxes(frame_width: int, frame_height: int) -> list[tuple[int, int, int, int]]:
        candidate_boxes = [
            (
                int(frame_width * 0.2),
                int(frame_height * 0.08),
                int(frame_width * 0.8),
                int(frame_height * 0.92),
            ),
            (
                int(frame_width * 0.14),
                int(frame_height * 0.38),
                int(frame_width * 0.86),
                frame_height,
            ),
            (
                int(frame_width * 0.39),
                int(frame_height * 0.34),
                int(frame_width * 0.63),
                int(frame_height * 0.62),
            ),
        ]
        focus_boxes: list[tuple[int, int, int, int]] = []
        for candidate in candidate_boxes:
            focus_boxes = YOLODetector._append_unique_box(focus_boxes, candidate)
        return focus_boxes

    @staticmethod
    def _build_actioncam_ball_boxes(frame_width: int, frame_height: int) -> list[tuple[int, int, int, int]]:
        candidate_boxes = [
            (
                int(frame_width * 0.32),
                int(frame_height * 0.20),
                int(frame_width * 0.68),
                int(frame_height * 0.82),
            ),
            (
                int(frame_width * 0.40),
                int(frame_height * 0.32),
                int(frame_width * 0.62),
                int(frame_height * 0.64),
            ),
        ]
        tile_boxes: list[tuple[int, int, int, int]] = []
        for candidate in candidate_boxes:
            tile_boxes = YOLODetector._append_unique_box(tile_boxes, candidate)
        return tile_boxes

    @staticmethod
    def _is_actioncam_crease_cluster(
        detections: list[DetectedObject],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        central_people = 0
        for detection in detections:
            if detection.object_type != "person":
                continue
            x1, y1, x2, y2 = detection.bbox
            cx = ((x1 + x2) / 2.0) / max(frame_width, 1)
            cy = ((y1 + y2) / 2.0) / max(frame_height, 1)
            h = (y2 - y1) / max(frame_height, 1)
            if 0.43 <= cx <= 0.60 and 0.38 <= cy <= 0.58 and 0.035 <= h <= 0.16:
                central_people += 1
        return central_people >= 2

    def _build_region_proposals(
        self,
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> list[tuple[int, int, int, int]]:
        frame_height, frame_width = rgb_frame.shape[:2]
        max_regions = settings.adaptive_roi_refinement_max_regions
        proposals: list[tuple[int, int, int, int]] = []

        for detection in sorted(detections, key=lambda item: item.confidence):
            needs_refinement = (
                detection.confidence < min(self._class_confidence_threshold(detection.object_type) + 0.18, 0.88)
                or detection.class_id in self.small_object_class_ids
            )
            if not needs_refinement:
                continue
            proposal = self._expand_box(
                detection.bbox,
                frame_width,
                frame_height,
                settings.adaptive_roi_refinement_expand_ratio + (0.08 if detection.class_id in self.small_object_class_ids else 0.0),
            )
            proposals = self._append_unique_box(proposals, proposal)
            if len(proposals) >= max_regions:
                return proposals[:max_regions]

        for proposal in self._extract_saliency_proposals(rgb_frame):
            proposals = self._append_unique_box(proposals, proposal)
            if len(proposals) >= max_regions:
                return proposals[:max_regions]

        if not proposals:
            center_box = (
                int(frame_width * 0.12),
                int(frame_height * 0.1),
                int(frame_width * 0.88),
                int(frame_height * 0.9),
            )
            proposals = self._append_unique_box(proposals, center_box)

        return proposals[:max_regions]

    def _should_run_small_object_tiles(
        self,
        frame_id: int | None,
        full_frame_detections: list[DetectedObject],
    ) -> bool:
        if settings.small_object_tile_frame_stride <= 1:
            return True
        if frame_id is None or frame_id % settings.small_object_tile_frame_stride == 0:
            return True
        return not self._has_reliable_full_frame_small_object(full_frame_detections)

    def _should_run_actioncam_ball_tiles(
        self,
        frame_id: int | None,
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> bool:
        if not self.small_object_class_ids:
            return False
        frame_height, frame_width = rgb_frame.shape[:2]
        if frame_width < 1280 or frame_height < 720:
            return False
        if not self._is_actioncam_crease_cluster(detections, frame_width, frame_height):
            return False
        if self._has_reliable_full_frame_small_object(detections):
            return False
        if frame_id is None:
            return True
        return frame_id % max(settings.small_object_tile_frame_stride, 2) == 0

    def _has_reliable_full_frame_small_object(
        self,
        detections: list[DetectedObject],
    ) -> bool:
        for detection in detections:
            if detection.class_id not in self.small_object_class_ids:
                continue
            confidence_floor = min(self._class_confidence_threshold(detection.object_type) + 0.12, 0.92)
            if detection.confidence >= confidence_floor:
                return True
        return False

    @staticmethod
    def _should_run_recall_boost(
        rgb_frame: np.ndarray,
        detections: list[DetectedObject],
    ) -> bool:
        if len(detections) == 0:
            return True

        grayscale = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        brightness = float(grayscale.mean())
        contrast = float(grayscale.std())
        return brightness < 78.0 and contrast < 42.0 and len(detections) < 3

    @staticmethod
    def _prepare_recall_frame(rgb_frame: np.ndarray) -> np.ndarray:
        lab_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        luminance, channel_a, channel_b = cv2.split(lab_frame)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        enhanced_luminance = clahe.apply(luminance)
        merged_lab = cv2.merge((enhanced_luminance, channel_a, channel_b))
        enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        softened = cv2.GaussianBlur(enhanced_frame, (0, 0), 1.2)
        return cv2.addWeighted(enhanced_frame, 1.12, softened, -0.12, 0)

    @staticmethod
    def _build_tile_boxes(frame_width: int, frame_height: int) -> list[tuple[int, int, int, int]]:
        tile_boxes: list[tuple[int, int, int, int]] = []
        grid_size = settings.small_object_tile_grid_size
        overlap_ratio = settings.small_object_tile_overlap_ratio
        tile_width = frame_width / grid_size
        tile_height = frame_height / grid_size
        step_x = tile_width * (1.0 - overlap_ratio)
        step_y = tile_height * (1.0 - overlap_ratio)

        for row in range(grid_size):
            for column in range(grid_size):
                left = int(round(column * step_x))
                top = int(round(row * step_y))
                if column == grid_size - 1:
                    left = max(int(round(frame_width - tile_width)), 0)
                if row == grid_size - 1:
                    top = max(int(round(frame_height - tile_height)), 0)
                right = int(round(left + tile_width))
                bottom = int(round(top + tile_height))
                left = max(left, 0)
                top = max(top, 0)
                right = min(max(right, left + 1), frame_width)
                bottom = min(max(bottom, top + 1), frame_height)
                tile_boxes.append((left, top, right, bottom))

        return tile_boxes

    def _extract_saliency_proposals(
        self,
        rgb_frame: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        frame_height, frame_width = rgb_frame.shape[:2]
        frame_area = max(float(frame_height * frame_width), 1.0)
        grayscale = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_32F))
        edges = cv2.Canny(blurred, 70, 180)
        combined = cv2.addWeighted(laplacian, 0.72, edges, 0.28, 0.0)
        threshold_value = max(int(np.percentile(combined, 82)), 28)
        _, binary = cv2.threshold(combined, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < 42:
                continue

            area_ratio = area / frame_area
            if (
                area_ratio < settings.adaptive_roi_refinement_min_area_ratio
                or area_ratio > settings.adaptive_roi_refinement_max_area_ratio
            ):
                continue

            aspect_ratio = max(width, height) / max(min(width, height), 1.0)
            if aspect_ratio > 4.6:
                continue

            box = self._expand_box(
                [float(x), float(y), float(x + width), float(y + height)],
                frame_width,
                frame_height,
                settings.adaptive_roi_refinement_expand_ratio,
            )
            left, top, right, bottom = box
            roi = combined[top:bottom, left:right]
            if roi.size == 0:
                continue

            detail_score = float(roi.mean()) / 255.0
            fill_ratio = area / max(float(width * height), 1.0)
            center_x = left + ((right - left) / 2.0)
            center_y = top + ((bottom - top) / 2.0)
            center_bias = 1.0 - min(
                (((center_x - (frame_width / 2.0)) ** 2 + (center_y - (frame_height / 2.0)) ** 2) ** 0.5)
                / max(((frame_width / 2.0) ** 2 + (frame_height / 2.0) ** 2) ** 0.5, 1.0),
                1.0,
            )
            score = detail_score * 1.4 + fill_ratio * 0.8 + center_bias * 0.45 + area_ratio * 0.6
            candidates.append((score, box))

        proposals: list[tuple[int, int, int, int]] = []
        for _, proposal in sorted(candidates, key=lambda item: item[0], reverse=True):
            proposals = self._append_unique_box(proposals, proposal)
            if len(proposals) >= settings.adaptive_roi_refinement_max_regions:
                break
        return proposals

    @staticmethod
    def _expand_box(
        bbox: list[float],
        frame_width: int,
        frame_height: int,
        expand_ratio: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        pad_x = box_width * expand_ratio
        pad_y = box_height * expand_ratio
        left = max(int(round(x1 - pad_x)), 0)
        top = max(int(round(y1 - pad_y)), 0)
        right = min(int(round(x2 + pad_x)), frame_width)
        bottom = min(int(round(y2 + pad_y)), frame_height)
        return left, top, max(right, left + 1), max(bottom, top + 1)

    @classmethod
    def _append_unique_box(
        cls,
        proposals: list[tuple[int, int, int, int]],
        candidate: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        candidate_box = [float(value) for value in candidate]
        if any(
            cls._bbox_iou(candidate_box, [float(value) for value in existing]) >= 0.58
            for existing in proposals
        ):
            return proposals
        return [*proposals, candidate]

    def _parse_detections(
        self,
        result,
        source_frame: np.ndarray,
        offset: tuple[int, int] = (0, 0),
    ) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        if result.boxes is None:
            return detections

        allowed_ids = set(self.allowed_class_ids or self.class_names_by_id.keys())
        names = self._normalize_names(result.names)
        offset_x, offset_y = offset
        for box in result.boxes:
            class_index = int(box.cls.item())
            if class_index not in allowed_ids:
                continue

            class_name = names.get(class_index, f"class_{class_index}")
            xyxy_local = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf.item())
            if confidence < self._class_confidence_threshold(class_name):
                continue
            detections.append(
                DetectedObject(
                    object_type=class_name,
                    class_id=class_index,
                    bbox=[
                        xyxy_local[0] + offset_x,
                        xyxy_local[1] + offset_y,
                        xyxy_local[2] + offset_x,
                        xyxy_local[3] + offset_y,
                    ],
                    confidence=confidence,
                    appearance_signature=self._compute_appearance_signature(source_frame, xyxy_local),
                )
            )

        return detections

    @staticmethod
    def _merge_detections(detections: list[DetectedObject]) -> list[DetectedObject]:
        merged: list[DetectedObject] = []

        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            overlap_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if candidate.class_id == existing.class_id
                    and YOLODetector._bbox_iou(candidate.bbox, existing.bbox) >= settings.detection_nms_iou_threshold
                ),
                None,
            )
            if overlap_index is not None:
                merged[overlap_index] = YOLODetector._blend_detection(merged[overlap_index], candidate)
                continue
            merged.append(candidate)

        return merged

    @staticmethod
    def _compute_appearance_signature(
        source_frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[float, ...] | None:
        if source_frame.size == 0:
            return None

        frame_height, frame_width = source_frame.shape[:2]
        x1 = max(int(round(bbox[0])), 0)
        y1 = max(int(round(bbox[1])), 0)
        x2 = min(max(int(round(bbox[2])), x1 + 1), frame_width)
        y2 = min(max(int(round(bbox[3])), y1 + 1), frame_height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        region = source_frame[y1:y2, x1:x2]
        if region.size == 0:
            return None

        resized = cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA)
        hsv_region = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv_region], [0, 1], None, [6, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grayscale, 50, 140)
        extras = np.array(
            [
                float((edges > 0).mean()),
                float(grayscale.std() / 128.0),
            ],
            dtype=np.float32,
        )
        signature = np.concatenate((histogram, extras))
        norm = float(np.linalg.norm(signature))
        if norm <= 0:
            return None
        signature /= norm
        return tuple(float(value) for value in signature.tolist())

    def _class_confidence_threshold(self, class_name: str) -> float:
        threshold_map = settings.class_confidence_threshold_map
        threshold = threshold_map.get(class_name, settings.min_detection_confidence)
        if class_name in settings.small_object_class_name_list:
            threshold = max(threshold, settings.small_object_track_confidence_threshold)
        return min(max(threshold, 0.0), 1.0)

    def _inference_confidence(self, class_ids: list[int] | None, tiled: bool = False) -> float:
        if not self.class_names_by_id:
            return max(self.confidence, 0.05)

        available_class_ids = class_ids or list(self.class_names_by_id.keys())
        thresholds = [
            self._class_confidence_threshold(self.class_names_by_id[class_id])
            for class_id in available_class_ids
            if class_id in self.class_names_by_id
        ]
        if not thresholds:
            return max(self.confidence, 0.05)

        margin = 0.1 if tiled else 0.08
        return max(min(thresholds) - margin, 0.08)

    def _predict(self, model: YOLO, **kwargs):
        with self._model_lock:
            return model.predict(
                device=self.device,
                verbose=False,
                **kwargs,
            )

    @staticmethod
    def _blend_detection(existing: DetectedObject, candidate: DetectedObject) -> DetectedObject:
        existing_weight = max(existing.confidence, 0.05)
        candidate_weight = max(candidate.confidence, 0.05)
        total_weight = existing_weight + candidate_weight
        blended_bbox = [
            round(
                ((existing_value * existing_weight) + (candidate_value * candidate_weight)) / total_weight,
                2,
            )
            for existing_value, candidate_value in zip(existing.bbox, candidate.bbox)
        ]
        return DetectedObject(
            object_type=existing.object_type,
            class_id=existing.class_id,
            bbox=blended_bbox,
            confidence=max(existing.confidence, candidate.confidence),
            appearance_signature=YOLODetector._blend_signature(
                existing.appearance_signature,
                candidate.appearance_signature,
                existing_weight,
                candidate_weight,
            ),
        )

    @staticmethod
    def _blend_signature(
        existing_signature: tuple[float, ...] | None,
        candidate_signature: tuple[float, ...] | None,
        existing_weight: float,
        candidate_weight: float,
    ) -> tuple[float, ...] | None:
        if existing_signature is None:
            return candidate_signature
        if candidate_signature is None:
            return existing_signature
        if len(existing_signature) != len(candidate_signature):
            return candidate_signature

        total_weight = max(existing_weight + candidate_weight, 1e-6)
        blended = np.array(
            [
                ((left * existing_weight) + (right * candidate_weight)) / total_weight
                for left, right in zip(existing_signature, candidate_signature)
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(blended))
        if norm <= 0:
            return candidate_signature
        blended /= norm
        return tuple(float(value) for value in blended.tolist())

    @staticmethod
    def _normalize_names(raw_names: object) -> dict[int, str]:
        if isinstance(raw_names, dict):
            items = raw_names.items()
        elif isinstance(raw_names, list):
            items = enumerate(raw_names)
        else:
            return {}

        normalized: dict[int, str] = {}
        for class_id, class_name in items:
            try:
                normalized[int(class_id)] = str(class_name).strip().lower()
            except (TypeError, ValueError):
                continue
        return normalized

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
    def _resolve_device() -> str:
        configured_device = settings.yolo_device.strip().lower()
        if configured_device and configured_device != "auto":
            return configured_device

        if torch is None:
            return "cpu"

        try:
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass

        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
        except Exception:
            pass

        return "cpu"
