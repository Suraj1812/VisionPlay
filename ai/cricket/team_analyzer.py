from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from backend.utils.config import settings


logger = logging.getLogger(__name__)


@dataclass
class PlayerInfo:
    tracking_id: int
    team_id: int = -1
    team_side: str = "unknown"
    role: str = "fielder"
    dominant_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: list[float] = field(default_factory=list)
    role_votes: Counter[str] = field(default_factory=Counter)
    side_votes: Counter[str] = field(default_factory=Counter)
    last_color_frame: int = -1
    last_seen_frame: int = -1
    confidence: float = 0.0


class TeamAnalyzer:
    def __init__(self) -> None:
        self._players: dict[int, PlayerInfo] = {}
        self._team_centers: list[np.ndarray] = []
        self._is_calibrated = False
        self._camera_profile = "default"
        self._color_samples: list[np.ndarray] = []
        self._sample_tids: list[int] = []
        self._min_samples = 12
        self._max_samples = 80
        self._color_eval_interval = 24
        self._role_eval_interval = 2
        self._ema_alpha = 0.08
        self._frame_roles: dict[int, dict[str, int]] = {}

    def analyze(
        self,
        frame: np.ndarray,
        player_detections: list[dict[str, Any]],
        frame_id: int,
    ) -> list[PlayerInfo]:
        frame_height, frame_width = frame.shape[:2]
        results: list[PlayerInfo] = []
        self._camera_profile = self._detect_camera_profile(player_detections, frame_width, frame_height)
        frame_role_assignments = self._assign_frame_roles(player_detections, frame_width, frame_height)

        for detection in player_detections:
            tracking_id = detection.get("tracking_id", -1)
            if tracking_id is None or tracking_id < 0:
                continue
            bbox = detection.get("bbox") or []
            if len(bbox) != 4:
                continue
            player = self._players.setdefault(tracking_id, PlayerInfo(tracking_id=tracking_id))
            player.bbox = [float(value) for value in bbox]
            player.last_seen_frame = frame_id
            player.confidence = max(float(detection.get("confidence", 0.0) or 0.0), player.confidence * 0.55)

            needs_color = (
                frame_id - player.last_color_frame >= self._color_eval_interval
                or player.team_id < 0
            )
            if needs_color:
                color = self._extract_dominant_color(frame, player.bbox)
                if color is not None:
                    player.dominant_color = tuple(float(value) for value in color.tolist())
                    player.last_color_frame = frame_id
                    if not self._is_calibrated and len(self._color_samples) < self._max_samples:
                        self._color_samples.append(color)
                        self._sample_tids.append(tracking_id)
                    if self._is_calibrated:
                        player.team_id = self._assign_team(color)
                        if 0 <= player.team_id < len(self._team_centers):
                            center = self._team_centers[player.team_id]
                            self._team_centers[player.team_id] = (
                                center * (1.0 - self._ema_alpha) + color * self._ema_alpha
                            )

            if frame_id % self._role_eval_interval == 0:
                role = frame_role_assignments.get(tracking_id, "fielder")
                player.role_votes[role] += 1
                player.role = player.role_votes.most_common(1)[0][0]
                side = self._role_to_side(player.role)
                player.side_votes[side] += 1
                player.team_side = player.side_votes.most_common(1)[0][0]

            results.append(player)

        if not self._is_calibrated and len(self._color_samples) >= self._min_samples:
            self._calibrate()

        self._frame_roles[frame_id] = frame_role_assignments
        self._prune_stale(frame_id)
        return results

    def get_team_summary(self) -> dict[str, Any]:
        teams: dict[int, list[dict[str, Any]]] = defaultdict(list)
        roles: dict[int, dict[str, Any]] = {}
        for player in self._players.values():
            teams[player.team_id].append(
                {
                    "id": player.tracking_id,
                    "role": player.role,
                    "team_side": player.team_side,
                    "color": [round(value, 2) for value in player.dominant_color],
                    "confidence": round(player.confidence, 4),
                }
            )
            roles[player.tracking_id] = {
                "role": player.role,
                "team_id": player.team_id,
                "team_side": player.team_side,
                "confidence": round(player.confidence, 4),
                "bbox": [round(value, 2) for value in player.bbox],
            }

        if not roles:
            return {}

        batting_team_id = self._resolve_team_id_for_roles({"striker", "non_striker"})
        fielding_team_id = self._resolve_team_id_for_roles({"bowler", "wicketkeeper", "fielder", "close_fielder"})

        team_payload: dict[str, Any] = {}
        for team_id, players in sorted(teams.items(), key=lambda item: item[0]):
            if team_id < 0:
                continue
            team_payload[f"team_{team_id}"] = {
                "players": players,
                "role_counts": dict(Counter(player["role"] for player in players)),
                "team_side": (
                    "batting"
                    if team_id == batting_team_id
                    else "fielding"
                    if team_id == fielding_team_id
                    else "unknown"
                ),
            }

        return {
            "camera_profile": self._camera_profile,
            "calibrated": self._is_calibrated,
            "teams": team_payload,
            "roles": roles,
            "batting_team_id": batting_team_id,
            "fielding_team_id": fielding_team_id,
            "role_method": "end-on action-cam heuristics",
        }

    def reset(self) -> None:
        self._players.clear()
        self._team_centers.clear()
        self._is_calibrated = False
        self._camera_profile = "default"
        self._color_samples.clear()
        self._sample_tids.clear()
        self._frame_roles.clear()

    def _extract_dominant_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        frame_height, frame_width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        box_height = y2 - y1
        box_width = x2 - x1
        if box_height < 26 or box_width < 12:
            return None

        torso = frame[
            y1 + int(box_height * 0.18): y1 + int(box_height * 0.62),
            x1 + int(box_width * 0.16): x2 - int(box_width * 0.16),
        ]
        if torso.size == 0 or torso.shape[0] < 6 or torso.shape[1] < 6:
            return None

        torso_small = cv2.resize(torso, (24, 32), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(torso_small, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        mask = (pixels[:, 0] > 38) & (pixels[:, 0] < 215)
        pixels = pixels[mask]
        if len(pixels) < 20:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            return None
        counts = np.bincount(labels.flatten())
        return centers[counts.argmax()]

    def _calibrate(self) -> None:
        colors = np.array(self._color_samples, dtype=np.float32)
        cluster_count = min(settings.cricket_team_color_clusters, len(colors))
        if cluster_count < 2:
            return
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        try:
            _, labels, centers = cv2.kmeans(
                colors,
                cluster_count,
                None,
                criteria,
                8,
                cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error:
            return

        self._team_centers = [centers[index] for index in range(cluster_count)]
        for sample_index, tracking_id in enumerate(self._sample_tids):
            if tracking_id in self._players:
                self._players[tracking_id].team_id = int(labels[sample_index][0])
        self._is_calibrated = True

    def _assign_team(self, color: np.ndarray) -> int:
        if not self._team_centers:
            return -1
        distances = [float(np.linalg.norm(color - center)) for center in self._team_centers]
        return int(np.argmin(distances))

    def _assign_frame_roles(
        self,
        player_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> dict[int, str]:
        candidates: list[dict[str, float]] = []
        for detection in player_detections:
            bbox = detection.get("bbox")
            tracking_id = detection.get("tracking_id")
            if tracking_id is None or tracking_id < 0 or not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            candidates.append(
                {
                    "tid": int(tracking_id),
                    "cx": ((x1 + x2) / 2.0) / max(frame_width, 1),
                    "cy": ((y1 + y2) / 2.0) / max(frame_height, 1),
                    "w": max(x2 - x1, 1.0) / max(frame_width, 1),
                    "h": max(y2 - y1, 1.0) / max(frame_height, 1),
                    "area": max(x2 - x1, 1.0) * max(y2 - y1, 1.0),
                }
            )

        if not candidates:
            return {}
        if self._camera_profile != "action-cam-end-on":
            return {candidate["tid"]: self._infer_default_role(candidate) for candidate in candidates}

        assignments: dict[int, str] = {candidate["tid"]: "fielder" for candidate in candidates}

        striker = self._pick_best(
            candidates,
            lambda candidate: 1.0
            - abs(candidate["cx"] - 0.5) * 3.6
            - abs(candidate["cy"] - 0.53) * 2.8
            - abs(candidate["h"] - 0.09) * 4.5,
        )
        if striker is not None:
            assignments[striker["tid"]] = "striker"

        wicketkeeper = self._pick_best(
            [candidate for candidate in candidates if assignments[candidate["tid"]] == "fielder"],
            lambda candidate: 1.0
            - abs(candidate["cx"] - 0.5) * 4.0
            - abs(candidate["cy"] - 0.43) * 3.6
            - abs(candidate["h"] - 0.08) * 4.0,
        )
        if wicketkeeper is not None:
            assignments[wicketkeeper["tid"]] = "wicketkeeper"

        bowler = self._pick_best(
            [candidate for candidate in candidates if assignments[candidate["tid"]] == "fielder"],
            lambda candidate: 1.0
            - abs(candidate["cx"] - 0.5) * 2.4
            + max(candidate["cy"] - 0.58, 0.0) * 2.8
            + max(candidate["h"] - 0.15, 0.0) * 2.4,
        )
        if bowler is not None:
            assignments[bowler["tid"]] = "bowler"

        non_striker = self._pick_best(
            [candidate for candidate in candidates if assignments[candidate["tid"]] == "fielder"],
            lambda candidate: 1.0
            - min(abs(candidate["cx"] - 0.37), abs(candidate["cx"] - 0.63)) * 3.2
            - abs(candidate["cy"] - 0.50) * 3.4
            - abs(candidate["h"] - 0.09) * 4.0,
        )
        if non_striker is not None:
            assignments[non_striker["tid"]] = "non_striker"

        umpire = self._pick_best(
            [candidate for candidate in candidates if assignments[candidate["tid"]] == "fielder"],
            lambda candidate: 1.0
            - abs(candidate["cx"] - 0.5) * 3.0
            - abs(candidate["cy"] - 0.31) * 3.2
            - abs(candidate["h"] - 0.11) * 3.6,
        )
        if umpire is not None:
            assignments[umpire["tid"]] = "umpire"

        for candidate in candidates:
            if assignments[candidate["tid"]] != "fielder":
                continue
            if 0.28 <= candidate["cx"] <= 0.72 and 0.34 <= candidate["cy"] <= 0.68:
                assignments[candidate["tid"]] = "close_fielder"
        return assignments

    @staticmethod
    def _pick_best(
        candidates: list[dict[str, float]],
        scorer,
    ) -> dict[str, float] | None:
        if not candidates:
            return None
        ranked = sorted(candidates, key=scorer, reverse=True)
        best = ranked[0]
        return best if scorer(best) > -0.8 else None

    @staticmethod
    def _detect_camera_profile(
        player_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> str:
        normalized_players: list[dict[str, float]] = []
        for detection in player_detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            normalized_players.append(
                {
                    "cx": ((x1 + x2) / 2.0) / max(frame_width, 1),
                    "cy": ((y1 + y2) / 2.0) / max(frame_height, 1),
                    "h": max(y2 - y1, 1.0) / max(frame_height, 1),
                }
            )

        if len(normalized_players) < 2:
            return "default"

        far_crease_players = [
            player
            for player in normalized_players
            if 0.42 <= player["cx"] <= 0.62 and 0.36 <= player["cy"] <= 0.60 and 0.03 <= player["h"] <= 0.18
        ]
        near_runner = any(
            0.24 <= player["cx"] <= 0.82 and player["cy"] >= 0.52 and player["h"] >= 0.16
            for player in normalized_players
        )
        wide_fielders = sum(
            1 for player in normalized_players if player["cx"] <= 0.18 or player["cx"] >= 0.82
        )
        if len(far_crease_players) >= 2 and (near_runner or wide_fielders >= 1):
            return "action-cam-end-on"
        return "default"

    def _resolve_team_id_for_roles(self, roles: set[str]) -> int:
        team_votes: Counter[int] = Counter()
        for player in self._players.values():
            if player.team_id < 0 or player.role not in roles:
                continue
            team_votes[player.team_id] += 1
        return team_votes.most_common(1)[0][0] if team_votes else -1

    @staticmethod
    def _role_to_side(role: str) -> str:
        if role in {"striker", "non_striker"}:
            return "batting"
        if role in {"bowler", "wicketkeeper", "fielder", "close_fielder"}:
            return "fielding"
        if role == "umpire":
            return "neutral"
        return "unknown"

    @staticmethod
    def _infer_default_role(candidate: dict[str, float]) -> str:
        cx = candidate["cx"]
        cy = candidate["cy"]
        if 0.35 < cx < 0.65 and cy > 0.72:
            return "striker"
        if 0.35 < cx < 0.65 and cy < 0.32:
            return "bowler"
        if 0.35 < cx < 0.65 and 0.82 < cy < 0.94:
            return "wicketkeeper"
        if (cx < 0.22 or cx > 0.78) and 0.50 < cy < 0.75:
            return "umpire"
        return "fielder"

    @staticmethod
    def _infer_role(
        cx: float,
        cy: float,
        height_ratio: float,
        camera_profile: str = "default",
    ) -> str:
        if camera_profile == "action-cam-end-on":
            if 0.44 < cx < 0.60 and 0.38 < cy < 0.50 and 0.03 <= height_ratio <= 0.14:
                return "wicketkeeper"
            if 0.42 < cx < 0.62 and 0.42 < cy < 0.62 and 0.03 <= height_ratio <= 0.18:
                return "striker"
            if 0.24 < cx < 0.82 and cy >= 0.52 and height_ratio >= 0.16:
                return "bowler"
            if (cx < 0.18 or cx > 0.82) and 0.30 < cy < 0.76:
                return "fielder"
            return "fielder"
        return TeamAnalyzer._infer_default_role({"cx": cx, "cy": cy, "h": height_ratio})

    def _prune_stale(self, frame_id: int) -> None:
        stale_players = [
            tracking_id
            for tracking_id, player in self._players.items()
            if frame_id - max(player.last_seen_frame, player.last_color_frame) > 300
        ]
        for tracking_id in stale_players:
            self._players.pop(tracking_id, None)
