from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np

from backend.utils.config import settings


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional runtime dependency
    import mediapipe as mp
except Exception:  # pragma: no cover - keep local startup resilient if import fails
    mp = None


@dataclass
class AuxiliaryFace:
    bbox: list[float]
    confidence: float
    emotion: str
    smile_score: float
    eye_openness: float
    head_pose: str


@dataclass
class AuxiliaryHand:
    bbox: list[float]
    confidence: float
    handedness: str
    gesture: str | None = None
    gesture_confidence: float = 0.0


@dataclass
class AuxiliaryVisionResult:
    faces: list[AuxiliaryFace] = field(default_factory=list)
    hands: list[AuxiliaryHand] = field(default_factory=list)
    reactions: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class AuxiliaryVisionService:
    FACE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    GESTURE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._analysis_lock = Lock()
        self._initialized = False
        self._available = mp is not None
        self._face_landmarker = None
        self._gesture_recognizer = None
        self._last_timestamp_ms = -1

    @property
    def available(self) -> bool:
        return self._available

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> AuxiliaryVisionResult:
        if not self._available:
            return AuxiliaryVisionResult()

        self._ensure_initialized()
        if self._face_landmarker is None and self._gesture_recognizer is None:
            return AuxiliaryVisionResult()

        inference_frame, inference_scale = self._prepare_inference_frame(frame_bgr)
        rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = AuxiliaryVisionResult()

        with self._analysis_lock:
            monotonic_timestamp_ms = self._next_monotonic_timestamp_ms(timestamp_ms)

            try:
                if self._face_landmarker is not None:
                    face_result = self._face_landmarker.detect_for_video(mp_image, monotonic_timestamp_ms)
                    result.faces = self._parse_faces(
                        face_result,
                        inference_frame.shape[1],
                        inference_frame.shape[0],
                    )
            except Exception:
                logger.exception("Face landmarker inference failed")

            try:
                if self._gesture_recognizer is not None:
                    gesture_result = self._gesture_recognizer.recognize_for_video(mp_image, monotonic_timestamp_ms)
                    result.hands = self._parse_hands(
                        gesture_result,
                        inference_frame.shape[1],
                        inference_frame.shape[0],
                    )
            except Exception:
                logger.exception("Gesture recognizer inference failed")

        if inference_scale < 1.0:
            result.faces = [
                AuxiliaryFace(
                    bbox=self._rescale_bbox(face.bbox, inference_scale, frame_bgr.shape[1], frame_bgr.shape[0]),
                    confidence=face.confidence,
                    emotion=face.emotion,
                    smile_score=face.smile_score,
                    eye_openness=face.eye_openness,
                    head_pose=face.head_pose,
                )
                for face in result.faces
            ]
            result.hands = [
                AuxiliaryHand(
                    bbox=self._rescale_bbox(hand.bbox, inference_scale, frame_bgr.shape[1], frame_bgr.shape[0]),
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    gesture=hand.gesture,
                    gesture_confidence=hand.gesture_confidence,
                )
                for hand in result.hands
            ]

        result.metrics = self._build_metrics(result)
        result.reactions = self._build_reactions(result)
        return result

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if mp is None:
                self._available = False
                self._initialized = True
                return

            settings.ensure_directories()
            face_model_path = self._ensure_model_asset("face_landmarker.task", self.FACE_MODEL_URL)
            gesture_model_path = self._ensure_model_asset("gesture_recognizer.task", self.GESTURE_MODEL_URL)

            try:
                vision = mp.tasks.vision
                base_options = mp.tasks.BaseOptions
                running_mode = vision.RunningMode.VIDEO
                base_option_delegate = getattr(base_options, "Delegate", None)
                cpu_delegate = getattr(base_option_delegate, "CPU", None)

                if face_model_path is not None:
                    face_base_options: dict[str, Any] = {"model_asset_path": str(face_model_path)}
                    if cpu_delegate is not None:
                        face_base_options["delegate"] = cpu_delegate
                    self._face_landmarker = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=base_options(**face_base_options),
                            running_mode=running_mode,
                            num_faces=2,
                            output_face_blendshapes=True,
                            output_facial_transformation_matrixes=True,
                            min_face_detection_confidence=0.45,
                            min_face_presence_confidence=0.45,
                            min_tracking_confidence=0.45,
                        )
                    )

                if gesture_model_path is not None:
                    gesture_base_options: dict[str, Any] = {"model_asset_path": str(gesture_model_path)}
                    if cpu_delegate is not None:
                        gesture_base_options["delegate"] = cpu_delegate
                    self._gesture_recognizer = vision.GestureRecognizer.create_from_options(
                        vision.GestureRecognizerOptions(
                            base_options=base_options(**gesture_base_options),
                            running_mode=running_mode,
                            num_hands=2,
                            min_hand_detection_confidence=0.42,
                            min_hand_presence_confidence=0.42,
                            min_tracking_confidence=0.42,
                        )
                    )
            except Exception:
                logger.exception("Unable to initialize auxiliary MediaPipe tasks")
                self._available = False

            self._initialized = True

    def _ensure_model_asset(self, filename: str, url: str) -> Path | None:
        asset_path = settings.model_cache_path / filename
        if asset_path.exists():
            return asset_path
        if not settings.auto_download_auxiliary_models:
            logger.warning("Auxiliary model download disabled; missing %s", asset_path)
            return None

        try:
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading auxiliary model %s", filename)
            urlretrieve(url, asset_path)
            return asset_path
        except Exception:
            logger.exception("Unable to download auxiliary model from %s", url)
            return None

    @staticmethod
    def _prepare_inference_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        frame_height, frame_width = frame_bgr.shape[:2]
        max_dimension = max(frame_width, frame_height)
        if max_dimension <= settings.live_aux_max_dimension:
            return frame_bgr, 1.0

        scale = settings.live_aux_max_dimension / max(max_dimension, 1)
        resized_frame = cv2.resize(
            frame_bgr,
            (
                max(int(round(frame_width * scale)), 1),
                max(int(round(frame_height * scale)), 1),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return resized_frame, scale

    def _next_monotonic_timestamp_ms(self, requested_timestamp_ms: int) -> int:
        candidate = max(int(requested_timestamp_ms), 0)
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate

    @staticmethod
    def _rescale_bbox(
        bbox: list[float],
        inference_scale: float,
        frame_width: int,
        frame_height: int,
    ) -> list[float]:
        if inference_scale <= 0:
            return bbox
        scale_back = 1.0 / inference_scale
        return [
            round(min(max(value * scale_back, 0.0), float(frame_width if index % 2 == 0 else frame_height)), 2)
            for index, value in enumerate(bbox)
        ]

    def _parse_faces(self, result, frame_width: int, frame_height: int) -> list[AuxiliaryFace]:
        landmarks_by_face = list(getattr(result, "face_landmarks", []) or [])
        blendshapes_by_face = list(getattr(result, "face_blendshapes", []) or [])
        transforms = list(getattr(result, "facial_transformation_matrixes", []) or [])
        parsed_faces: list[AuxiliaryFace] = []

        for index, landmarks in enumerate(landmarks_by_face):
            bbox = self._bbox_from_normalized_landmarks(landmarks, frame_width, frame_height, padding_ratio=0.08)
            if bbox is None:
                continue
            blendshape_map = self._categories_to_map(blendshapes_by_face[index] if len(blendshapes_by_face) > index else [])
            smile_score = float(
                (blendshape_map.get("mouthSmileLeft", 0.0) + blendshape_map.get("mouthSmileRight", 0.0)) / 2.0
            )
            eye_openness = float(
                1.0 - ((blendshape_map.get("eyeBlinkLeft", 0.0) + blendshape_map.get("eyeBlinkRight", 0.0)) / 2.0)
            )
            emotion = self._classify_emotion(blendshape_map)
            head_pose = self._classify_head_pose(
                transforms[index] if len(transforms) > index else None,
                landmarks,
            )
            confidence = float(max(blendshape_map.get("facePresence", 0.65), 0.5))
            parsed_faces.append(
                AuxiliaryFace(
                    bbox=bbox,
                    confidence=min(confidence, 0.98),
                    emotion=emotion,
                    smile_score=round(smile_score, 4),
                    eye_openness=round(max(min(eye_openness, 1.0), 0.0), 4),
                    head_pose=head_pose,
                )
            )

        return parsed_faces

    def _parse_hands(self, result, frame_width: int, frame_height: int) -> list[AuxiliaryHand]:
        hand_landmarks = list(getattr(result, "hand_landmarks", []) or [])
        handedness_lists = list(getattr(result, "handedness", []) or [])
        gesture_lists = list(getattr(result, "gestures", []) or [])
        hands: list[AuxiliaryHand] = []

        for index, landmarks in enumerate(hand_landmarks):
            bbox = self._bbox_from_normalized_landmarks(landmarks, frame_width, frame_height, padding_ratio=0.12)
            if bbox is None:
                continue
            handedness = self._primary_category_name(handedness_lists[index] if len(handedness_lists) > index else [])
            gesture = self._primary_category_name(gesture_lists[index] if len(gesture_lists) > index else [])
            handedness_score = self._primary_category_score(
                handedness_lists[index] if len(handedness_lists) > index else []
            )
            gesture_confidence = self._primary_category_score(
                gesture_lists[index] if len(gesture_lists) > index else []
            )
            confidence = max(gesture_confidence, handedness_score * 0.92, 0.28)
            hands.append(
                AuxiliaryHand(
                    bbox=bbox,
                    confidence=round(confidence, 4),
                    handedness=handedness or "unknown",
                    gesture=gesture,
                    gesture_confidence=round(gesture_confidence, 4),
                )
            )

        return hands

    @staticmethod
    def _categories_to_map(categories) -> dict[str, float]:
        parsed: dict[str, float] = {}
        for category in categories or []:
            name = getattr(category, "category_name", None)
            score = getattr(category, "score", None)
            if not name or score is None:
                continue
            parsed[str(name)] = float(score)
        return parsed

    @staticmethod
    def _primary_category_name(categories) -> str | None:
        best_name = None
        best_score = float("-inf")
        for category in categories or []:
            score = float(getattr(category, "score", 0.0) or 0.0)
            if score > best_score:
                best_score = score
                best_name = getattr(category, "category_name", None)
        return str(best_name) if best_name else None

    @staticmethod
    def _primary_category_score(categories) -> float:
        best_score = 0.0
        for category in categories or []:
            best_score = max(best_score, float(getattr(category, "score", 0.0) or 0.0))
        return best_score

    @staticmethod
    def _bbox_from_normalized_landmarks(
        landmarks,
        frame_width: int,
        frame_height: int,
        padding_ratio: float,
    ) -> list[float] | None:
        if not landmarks:
            return None

        xs = [float(getattr(landmark, "x", 0.0)) for landmark in landmarks]
        ys = [float(getattr(landmark, "y", 0.0)) for landmark in landmarks]
        if not xs or not ys:
            return None

        left = min(xs) * frame_width
        right = max(xs) * frame_width
        top = min(ys) * frame_height
        bottom = max(ys) * frame_height
        width = max(right - left, 1.0)
        height = max(bottom - top, 1.0)
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio
        x1 = max(left - pad_x, 0.0)
        y1 = max(top - pad_y, 0.0)
        x2 = min(right + pad_x, float(frame_width))
        y2 = min(bottom + pad_y, float(frame_height))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

    @staticmethod
    def _classify_emotion(blendshape_map: dict[str, float]) -> str:
        smile = (blendshape_map.get("mouthSmileLeft", 0.0) + blendshape_map.get("mouthSmileRight", 0.0)) / 2.0
        jaw_open = blendshape_map.get("jawOpen", 0.0)
        blink = (blendshape_map.get("eyeBlinkLeft", 0.0) + blendshape_map.get("eyeBlinkRight", 0.0)) / 2.0
        eye_wide = (blendshape_map.get("eyeWideLeft", 0.0) + blendshape_map.get("eyeWideRight", 0.0)) / 2.0
        brow_down = (blendshape_map.get("browDownLeft", 0.0) + blendshape_map.get("browDownRight", 0.0)) / 2.0
        brow_inner_up = blendshape_map.get("browInnerUp", 0.0)
        frown = (blendshape_map.get("mouthFrownLeft", 0.0) + blendshape_map.get("mouthFrownRight", 0.0)) / 2.0

        if smile >= 0.46:
            return "smiling"
        if blink >= 0.56 and smile < 0.18 and jaw_open < 0.24:
            return "sleepy"
        if jaw_open >= 0.3 and (brow_inner_up >= 0.12 or eye_wide >= 0.12):
            return "surprised"
        if brow_down >= 0.24 and frown >= 0.08:
            return "angry"
        return "neutral"

    @staticmethod
    def _classify_head_pose(transform_matrix, landmarks) -> str:
        if transform_matrix is not None:
            matrix = np.array(getattr(transform_matrix, "rows", transform_matrix), dtype=np.float32).reshape(-1)
            if matrix.size >= 16:
                yaw_hint = float(matrix[8])
                pitch_hint = float(matrix[9])
                if yaw_hint >= 0.22:
                    return "turned_left"
                if yaw_hint <= -0.22:
                    return "turned_right"
                if pitch_hint >= 0.2:
                    return "looking_up"
                if pitch_hint <= -0.22:
                    return "looking_down"
        if not landmarks or len(landmarks) < 300:
            return "forward"

        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]
        horizontal_balance = float(getattr(nose, "x", 0.5)) - (
            float(getattr(left_face, "x", 0.25)) + float(getattr(right_face, "x", 0.75))
        ) / 2.0
        if horizontal_balance >= 0.035:
            return "turned_left"
        if horizontal_balance <= -0.035:
            return "turned_right"
        return "forward"

    @staticmethod
    def _build_metrics(result: AuxiliaryVisionResult) -> dict[str, Any]:
        primary_face = result.faces[0] if result.faces else None
        primary_hand = result.hands[0] if result.hands else None
        metrics: dict[str, Any] = {
            "face_count": len(result.faces),
            "hand_count": len(result.hands),
        }
        if primary_face is not None:
            metrics["emotion"] = primary_face.emotion
            metrics["smile_score"] = primary_face.smile_score
            metrics["eye_openness"] = primary_face.eye_openness
            metrics["head_pose"] = primary_face.head_pose
        if primary_hand is not None:
            normalized_gesture = (primary_hand.gesture or "").strip().lower()
            normalized_handedness = (primary_hand.handedness or "").strip().lower()
            if normalized_gesture not in {"", "none", "unknown"}:
                metrics["gesture"] = primary_hand.gesture
            if normalized_handedness not in {"", "unknown"}:
                metrics["handedness"] = primary_hand.handedness
        return metrics

    @staticmethod
    def _build_reactions(result: AuxiliaryVisionResult) -> list[str]:
        reactions: list[str] = []
        if result.faces:
            primary_face = result.faces[0]
            if primary_face.emotion != "neutral":
                reactions.append(f"Looks {primary_face.emotion}")
            if primary_face.head_pose != "forward":
                pose_label = primary_face.head_pose.replace("turned_", "").replace("_", " ")
                reactions.append(f"Looking {pose_label}")
        if result.hands:
            primary_hand = result.hands[0]
            normalized_handedness = (primary_hand.handedness or "").strip().lower()
            normalized_gesture = (primary_hand.gesture or "").strip().lower()
            if normalized_gesture not in {"", "none", "unknown"}:
                reactions.append(f"Gesture {primary_hand.gesture.replace('_', ' ')}")
            elif normalized_handedness not in {"", "unknown"}:
                reactions.append(f"{primary_hand.handedness.title()} hand visible")
        return reactions[:3]


auxiliary_vision_service = AuxiliaryVisionService()
