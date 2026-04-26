
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class EventType(Enum):
    BALL_RELEASED = "ball_released"
    BALL_BOUNCED = "ball_bounced"
    BAT_IMPACT = "bat_impact"
    BALL_TO_BOUNDARY = "ball_to_boundary"
    FOUR = "four"
    SIX = "six"
    DOT_BALL = "dot_ball"
    WICKET = "wicket"
    WIDE = "wide"
    NO_BALL = "no_ball"
    CATCH_ATTEMPT = "catch_attempt"
    APPEAL = "appeal"
    RUN_SCORED = "run_scored"
    OVER_COMPLETE = "over_complete"
    CELEBRATION = "celebration"
    FIELD_CHANGE = "field_change"
    CAMERA_CUT = "camera_cut"
    REPLAY = "replay"

@dataclass
class CricketEvent:
    event_type: EventType
    frame_id: int
    timestamp_ms: int
    confidence: float
    details: dict = field(default_factory=dict)

@dataclass
class BallPoint:
    frame_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0

class DeliveryPhase(Enum):

    IDLE = "idle"           # Between deliveries
    BALL_IN_PLAY = "ball_in_play"   # Ball released, tracking
    POST_IMPACT = "post_impact"     # After bat/pad contact
    DEAD = "dead"           # Play stopped

class CricketEventClassifier:

    def __init__(self, fps: float = 25.0, frame_width: int = 1920, frame_height: int = 1080):
        self.fps = fps
        self.fw = frame_width
        self.fh = frame_height
        self.dt = 1.0 / max(fps, 1.0)

        self._ball_pts: deque[BallPoint] = deque(maxlen=150)
        self._ball_lost_frames = 0

        self._phase = DeliveryPhase.IDLE
        self._delivery_start_frame = -1
        self._total_deliveries = 0
        self._balls_this_over = 0

        self._bounce_evidence = 0
        self._impact_evidence = 0
        self._boundary_evidence = 0
        self._wicket_evidence = 0

        self._bounce_votes_needed = 2
        self._impact_votes_needed = 2
        self._boundary_votes_needed = 3
        self._wicket_votes_needed = 5

        self._cooldowns: dict[str, int] = {}
        self._cooldown_duration = int(fps * 1.0)

        self._player_count_buf: deque[int] = deque(maxlen=45)

        self.confirmed_events: list[CricketEvent] = []

    def process_frame(
        self,
        frame_id: int,
        ball_detection: dict | None,
        player_detections: list[dict],
        bat_detections: list[dict],
        fps: float | None = None,
        ball_velocity: tuple[float, float] | None = None,
        ball_acceleration: tuple[float, float] | None = None,
    ) -> list[CricketEvent]:
        if fps and fps != self.fps:
            self.fps = fps
            self.dt = 1.0 / max(fps, 1.0)

        ts = int((frame_id / max(self.fps, 1.0)) * 1000)
        new_events: list[CricketEvent] = []

        self._player_count_buf.append(len(player_detections))

        if ball_detection:
            bx = (ball_detection["bbox"][0] + ball_detection["bbox"][2]) / 2.0
            by = (ball_detection["bbox"][1] + ball_detection["bbox"][3]) / 2.0
            bp = BallPoint(frame_id=frame_id, x=bx, y=by)
            if ball_velocity is not None:
                bp.vx, bp.vy = ball_velocity
                bp.speed = math.hypot(bp.vx, bp.vy)
            elif self._ball_pts:
                prev = self._ball_pts[-1]
                df = max(frame_id - prev.frame_id, 1)
                bp.vx = (bx - prev.x) / df
                bp.vy = (by - prev.y) / df
                bp.speed = math.hypot(bp.vx, bp.vy)
            self._ball_pts.append(bp)
            self._ball_lost_frames = 0
        else:
            self._ball_lost_frames += 1

        if self._phase == DeliveryPhase.IDLE:
            ev = self._check_ball_release(frame_id, ts)
            if ev:
                new_events.append(ev)

        elif self._phase == DeliveryPhase.BALL_IN_PLAY:
            ev = self._check_bounce(frame_id, ts)
            if ev:
                new_events.append(ev)

            ev = self._check_impact(frame_id, ts, bat_detections)
            if ev:
                new_events.append(ev)

            ev = self._check_boundary(frame_id, ts)
            if ev:
                new_events.append(ev)

            if self._ball_lost_frames > int(self.fps * 1.5):
                self._phase = DeliveryPhase.DEAD
                scoring = self._determine_scoring(frame_id, ts)
                if scoring:
                    new_events.append(scoring)

        elif self._phase == DeliveryPhase.POST_IMPACT:
            ev = self._check_boundary(frame_id, ts)
            if ev:
                new_events.append(ev)

            if self._ball_lost_frames > int(self.fps * 2.0):
                self._phase = DeliveryPhase.DEAD
                scoring = self._determine_scoring(frame_id, ts)
                if scoring:
                    new_events.append(scoring)

        elif self._phase == DeliveryPhase.DEAD:
            if ball_detection and self._ball_lost_frames == 0:
                self._phase = DeliveryPhase.IDLE
                self._reset_evidence()

        ev = self._check_wicket(frame_id, ts)
        if ev:
            new_events.append(ev)

        if self._balls_this_over >= 6:
            new_events.append(CricketEvent(
                event_type=EventType.OVER_COMPLETE,
                frame_id=frame_id, timestamp_ms=ts,
                confidence=0.85,
                details={"balls": self._balls_this_over, "over_number": self._total_deliveries // 6}
            ))
            self._balls_this_over = 0

        filtered = self._apply_cooldowns(new_events, frame_id)
        self.confirmed_events.extend(filtered)
        return filtered

    def _check_ball_release(self, fid: int, ts: int) -> CricketEvent | None:
        if len(self._ball_pts) < 2:
            return None
        bp = self._ball_pts[-1]
        ny = bp.y / self.fh
        if ny < 0.50 and bp.vy > 1.5 and bp.speed > 2.0:
            self._phase = DeliveryPhase.BALL_IN_PLAY
            self._delivery_start_frame = fid
            self._total_deliveries += 1
            self._balls_this_over += 1
            self._reset_evidence()
            return CricketEvent(
                event_type=EventType.BALL_RELEASED,
                frame_id=fid, timestamp_ms=ts, confidence=0.75,
                details={"delivery": self._total_deliveries, "ball_in_over": self._balls_this_over}
            )
        return None

    def _check_bounce(self, fid: int, ts: int) -> CricketEvent | None:
        if len(self._ball_pts) < 4:
            return None
        pts = list(self._ball_pts)[-4:]
        for i in range(1, len(pts)):
            if pts[i-1].vy > 1.5 and pts[i].vy < -0.3:
                self._bounce_evidence += 1
                if self._bounce_evidence >= self._bounce_votes_needed:
                    bx, by = pts[i].x, pts[i].y
                    length = self._classify_length(by / self.fh)
                    line = self._classify_line(bx / self.fw)
                    conf = min(0.60 + self._bounce_evidence * 0.08, 0.95)
                    return CricketEvent(
                        event_type=EventType.BALL_BOUNCED,
                        frame_id=fid, timestamp_ms=ts, confidence=conf,
                        details={
                            "bounce_x": round(bx, 1), "bounce_y": round(by, 1),
                            "length": length, "line": line,
                            "norm_x": round(bx / self.fw, 3), "norm_y": round(by / self.fh, 3),
                        }
                    )
        return None

    def _check_impact(self, fid: int, ts: int, bat_dets: list[dict]) -> CricketEvent | None:
        if len(self._ball_pts) < 3:
            return None
        pts = list(self._ball_pts)[-3:]
        vx_diff = pts[-1].vx - pts[-2].vx
        vy_diff = pts[-1].vy - pts[-2].vy
        accel = math.hypot(vx_diff, vy_diff)
        if accel > 4.0:
            self._impact_evidence += 1
            if self._impact_evidence >= self._impact_votes_needed:
                bp = pts[-1]
                shot = self._classify_shot(bp.vx, bp.vy)
                power = "powerful" if bp.speed > 18 else ("medium" if bp.speed > 8 else "soft")
                exit_angle = math.degrees(math.atan2(-bp.vy, bp.vx)) if (bp.vx or bp.vy) else 0.0
                wagon_zone = self._wagon_wheel_zone(exit_angle)
                conf = min(0.55 + self._impact_evidence * 0.10, 0.95)
                self._phase = DeliveryPhase.POST_IMPACT
                return CricketEvent(
                    event_type=EventType.BAT_IMPACT,
                    frame_id=fid, timestamp_ms=ts, confidence=conf,
                    details={
                        "shot_type": shot, "power": power,
                        "exit_angle": round(exit_angle, 1),
                        "acceleration": round(accel, 2),
                        "speed_px_frame": round(bp.speed, 2),
                        "wagon_zone": wagon_zone,
                    }
                )
        return None

    def _check_boundary(self, fid: int, ts: int) -> CricketEvent | None:
        if not self._ball_pts:
            return None
        bp = self._ball_pts[-1]
        nx, ny = bp.x / self.fw, bp.y / self.fh
        margin = 0.06
        at_edge = nx < margin or nx > (1 - margin) or ny > (1 - margin)
        if not at_edge:
            self._boundary_evidence = 0
            return None
        self._boundary_evidence += 1
        if self._boundary_evidence < self._boundary_votes_needed:
            return None
        is_six = bp.vy < -4.0 and ny < 0.35
        etype = EventType.SIX if is_six else EventType.FOUR
        self._phase = DeliveryPhase.DEAD
        return CricketEvent(
            event_type=etype, frame_id=fid, timestamp_ms=ts,
            confidence=0.70,
            details={"exit_x": round(bp.x, 1), "exit_y": round(bp.y, 1),
                      "runs": 6 if is_six else 4}
        )

    def _check_wicket(self, fid: int, ts: int) -> CricketEvent | None:
        if len(self._player_count_buf) < 20:
            return None
        recent = list(self._player_count_buf)
        avg_recent = sum(recent[-8:]) / 8
        avg_baseline = sum(recent[:12]) / 12
        if avg_recent > avg_baseline * 1.6 and avg_recent > 5:
            self._wicket_evidence += 1
            if self._wicket_evidence >= self._wicket_votes_needed:
                self._phase = DeliveryPhase.DEAD
                return CricketEvent(
                    event_type=EventType.WICKET,
                    frame_id=fid, timestamp_ms=ts,
                    confidence=min(0.50 + self._wicket_evidence * 0.05, 0.85),
                    details={"clustering_ratio": round(avg_recent / max(avg_baseline, 1), 2)}
                )
        else:
            self._wicket_evidence = max(self._wicket_evidence - 1, 0)
        return None

    def _determine_scoring(self, fid: int, ts: int) -> CricketEvent | None:

        for ev in self.confirmed_events[-5:]:
            if ev.event_type in (EventType.FOUR, EventType.SIX, EventType.WICKET):
                if fid - ev.frame_id < int(self.fps * 3):
                    return None
        return CricketEvent(
            event_type=EventType.DOT_BALL,
            frame_id=fid, timestamp_ms=ts, confidence=0.45,
        )

    @staticmethod
    def _classify_length(norm_y: float) -> str:
        if norm_y > 0.78:
            return "yorker"
        elif norm_y > 0.68:
            return "full"
        elif norm_y > 0.55:
            return "good_length"
        elif norm_y > 0.42:
            return "short"
        return "bouncer"

    @staticmethod
    def _classify_line(norm_x: float) -> str:
        if norm_x < 0.40:
            return "outside_off"
        elif norm_x < 0.47:
            return "off_stump"
        elif norm_x < 0.53:
            return "middle_stump"
        elif norm_x < 0.60:
            return "leg_stump"
        return "outside_leg"

    @staticmethod
    def _classify_shot(vx: float, vy: float) -> str:
        if vx == 0 and vy == 0:
            return "defensive"
        angle = math.degrees(math.atan2(-vy, vx))
        speed = math.hypot(vx, vy)
        if speed < 3:
            return "defensive"
        if -20 <= angle <= 20:
            return "cover_drive" if vx > 0 else "flick"
        elif 20 < angle <= 55:
            return "lofted_drive"
        elif 55 < angle <= 90:
            return "upper_cut"
        elif -55 <= angle < -20:
            return "cut" if vx > 0 else "sweep"
        elif angle < -55:
            return "pull"
        elif 90 < angle <= 160:
            return "reverse_sweep"
        return "drive"

    @staticmethod
    def _wagon_wheel_zone(angle: float) -> int:

        a = (angle + 360) % 360
        return int((a + 22.5) // 45) % 8 + 1

    def _apply_cooldowns(self, events: list[CricketEvent], fid: int) -> list[CricketEvent]:
        filtered = []
        for ev in events:
            key = ev.event_type.value
            last = self._cooldowns.get(key, -9999)
            if fid - last >= self._cooldown_duration:
                filtered.append(ev)
                self._cooldowns[key] = fid
        return filtered

    def _reset_evidence(self):
        self._bounce_evidence = 0
        self._impact_evidence = 0
        self._boundary_evidence = 0

    def get_all_events(self) -> list[dict]:
        return [{
            "event_type": e.event_type.value,
            "frame_id": e.frame_id,
            "timestamp_ms": e.timestamp_ms,
            "confidence": round(e.confidence, 3),
            "details": e.details,
        } for e in self.confirmed_events]

    def get_delivery_summary(self) -> dict:
        fours = sum(1 for e in self.confirmed_events if e.event_type == EventType.FOUR)
        sixes = sum(1 for e in self.confirmed_events if e.event_type == EventType.SIX)
        wkts = sum(1 for e in self.confirmed_events if e.event_type == EventType.WICKET)
        dots = sum(1 for e in self.confirmed_events if e.event_type == EventType.DOT_BALL)
        impacts = sum(1 for e in self.confirmed_events if e.event_type == EventType.BAT_IMPACT)
        balls = self._total_deliveries
        return {
            "total_deliveries": balls,
            "estimated_overs": f"{balls // 6}.{balls % 6}",
            "fours": fours, "sixes": sixes, "wickets": wkts,
            "dot_balls": dots, "bat_impacts": impacts,
            "dot_pct": round(dots / max(balls, 1) * 100, 1),
            "boundary_pct": round((fours + sixes) / max(balls, 1) * 100, 1),
        }

    def reset(self):
        self._ball_pts.clear()
        self._ball_lost_frames = 0
        self._phase = DeliveryPhase.IDLE
        self._reset_evidence()
        self._player_count_buf.clear()
        self._cooldowns.clear()
        self._wicket_evidence = 0
