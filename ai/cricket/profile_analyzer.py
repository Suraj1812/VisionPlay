from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class CricketClipProfileReport:
    profile: str = "generic"
    confidence: float = 0.0
    specialized: bool = False
    overlay_present: bool = False
    fixed_camera_ratio: float = 0.0
    pitch_corridor_ratio: float = 0.0
    central_wicket_ratio: float = 0.0
    sample_count: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "confidence": round(self.confidence, 4),
            "specialized": self.specialized,
            "overlay_present": self.overlay_present,
            "fixed_camera_ratio": round(self.fixed_camera_ratio, 4),
            "pitch_corridor_ratio": round(self.pitch_corridor_ratio, 4),
            "central_wicket_ratio": round(self.central_wicket_ratio, 4),
            "sample_count": self.sample_count,
            "reasons": self.reasons,
        }


class CricketClipProfileAnalyzer:
    def __init__(self) -> None:
        self.reset()

    def observe(
        self,
        frame: np.ndarray,
        person_detections: list[dict[str, Any]],
        scene_cut: bool = False,
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return

        self._sample_count += 1
        self._overlay_votes += int(self._estimate_overlay_presence(frame))
        if not scene_cut:
            self._fixed_camera_votes += int(self._estimate_fixed_camera(frame))
        if self._estimate_pitch_corridor(frame):
            self._pitch_corridor_votes += 1
        if self._estimate_central_wicket_alignment(person_detections, frame_width, frame_height):
            self._central_wicket_votes += 1

    def build_report(self) -> CricketClipProfileReport:
        sample_count = max(self._sample_count, 1)
        fixed_camera_ratio = self._fixed_camera_votes / sample_count
        pitch_corridor_ratio = self._pitch_corridor_votes / sample_count
        central_wicket_ratio = self._central_wicket_votes / sample_count
        overlay_ratio = self._overlay_votes / sample_count
        confidence = min(
            fixed_camera_ratio * 0.34
            + pitch_corridor_ratio * 0.31
            + central_wicket_ratio * 0.35,
            1.0,
        )
        specialized = confidence >= 0.58
        reasons: list[str] = []
        if fixed_camera_ratio >= 0.55:
            reasons.append("camera remains stable from the bowler end")
        if pitch_corridor_ratio >= 0.52:
            reasons.append("central pitch corridor is consistently visible")
        if central_wicket_ratio >= 0.48:
            reasons.append("players repeatedly align around the far wicket corridor")
        if overlay_ratio >= 0.22:
            reasons.append("scoreboard-style overlay bands appear on the frame edges")
        return CricketClipProfileReport(
            profile="cricket_end_on_action_cam_v1" if specialized else "generic",
            confidence=confidence,
            specialized=specialized,
            overlay_present=overlay_ratio >= 0.22,
            fixed_camera_ratio=fixed_camera_ratio,
            pitch_corridor_ratio=pitch_corridor_ratio,
            central_wicket_ratio=central_wicket_ratio,
            sample_count=self._sample_count,
            reasons=reasons,
        )

    def reset(self) -> None:
        self._sample_count = 0
        self._fixed_camera_votes = 0
        self._pitch_corridor_votes = 0
        self._central_wicket_votes = 0
        self._overlay_votes = 0
        self._prev_gray: np.ndarray | None = None

    def _estimate_fixed_camera(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if self._prev_gray is None:
            self._prev_gray = small
            return True
        delta = cv2.absdiff(self._prev_gray, small)
        self._prev_gray = small
        return float(delta.mean()) <= 13.5

    @staticmethod
    def _estimate_pitch_corridor(frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        corridor = frame[int(height * 0.28):int(height * 0.96), int(width * 0.37):int(width * 0.63)]
        if corridor.size == 0:
            return False
        hsv = cv2.cvtColor(corridor, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean())
        value = float(hsv[:, :, 2].mean())
        brightness_std = float(hsv[:, :, 2].std())
        return saturation <= 72.0 and value >= 85.0 and brightness_std <= 48.0

    @staticmethod
    def _estimate_central_wicket_alignment(
        person_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        aligned = 0
        for detection in person_detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            cx = ((x1 + x2) / 2.0) / max(frame_width, 1)
            cy = ((y1 + y2) / 2.0) / max(frame_height, 1)
            height_ratio = max(y2 - y1, 1.0) / max(frame_height, 1)
            if 0.42 <= cx <= 0.61 and 0.34 <= cy <= 0.64 and 0.03 <= height_ratio <= 0.18:
                aligned += 1
        return aligned >= 2

    @staticmethod
    def _estimate_overlay_presence(frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        candidate_regions = [
            frame[0:int(height * 0.12), 0:width],
            frame[int(height * 0.84):height, 0:width],
        ]
        overlay_votes = 0
        for region in candidate_regions:
            if region.size == 0:
                continue
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float((edges > 0).mean())
            value_std = float(gray.std())
            if 0.016 <= edge_density <= 0.12 and value_std <= 58.0:
                overlay_votes += 1
        return overlay_votes >= 1
