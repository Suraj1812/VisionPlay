
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class BallRecord:
    ball_number: int
    over: str
    frame_start: int
    frame_end: int
    ts_start: int
    ts_end: int
    length: str = "unknown"
    line: str = "unknown"
    shot_type: str = "unknown"
    power: str = "unknown"
    runs: int = 0
    is_boundary: bool = False
    is_six: bool = False
    is_wicket: bool = False
    is_wide: bool = False
    is_no_ball: bool = False
    is_dot: bool = False
    wagon_zone: int = 0
    exit_angle: float = 0.0
    bounce_x: float = 0.0
    bounce_y: float = 0.0
    commentary: str = ""

class CricketAnalyticsAggregator:

    def __init__(self):
        self._balls: list[BallRecord] = []
        self._current: BallRecord | None = None
        self._count = 0

        self._partnership_runs = 0
        self._partnership_balls = 0

    def on_event(self, event: dict):
        etype = event.get("event_type", "")
        details = event.get("details", {})
        fid = event.get("frame_id", 0)
        ts = event.get("timestamp_ms", 0)

        if etype == "ball_released":
            self._count += 1
            ov = (self._count - 1) // 6
            b = ((self._count - 1) % 6) + 1
            self._current = BallRecord(
                ball_number=self._count, over=f"{ov}.{b}",
                frame_start=fid, frame_end=fid, ts_start=ts, ts_end=ts,
            )

        elif self._current:
            if etype == "ball_bounced":
                self._current.length = details.get("length", "unknown")
                self._current.line = details.get("line", "unknown")
                self._current.bounce_x = details.get("bounce_x", 0.0)
                self._current.bounce_y = details.get("bounce_y", 0.0)

            elif etype == "bat_impact":
                self._current.shot_type = details.get("shot_type", "unknown")
                self._current.power = details.get("power", "unknown")
                self._current.exit_angle = details.get("exit_angle", 0.0)
                self._current.wagon_zone = details.get("wagon_zone", 0)

            elif etype == "four":
                self._current.runs = 4
                self._current.is_boundary = True

            elif etype == "six":
                self._current.runs = 6
                self._current.is_six = True
                self._current.is_boundary = True

            elif etype == "dot_ball":
                self._current.is_dot = True
                self._current.runs = 0

            elif etype == "run_scored":
                self._current.runs = details.get("estimated_runs", 1)

            elif etype == "wicket":
                self._current.is_wicket = True
                self._partnership_runs = 0
                self._partnership_balls = 0

            elif etype in ("camera_cut", "over_complete"):
                self._finalize(fid, ts)

    def _finalize(self, fid: int, ts: int):
        if self._current is None:
            return
        self._current.frame_end = fid
        self._current.ts_end = ts
        self._current.commentary = self._gen_commentary(self._current)
        self._partnership_runs += self._current.runs
        self._partnership_balls += 1
        self._balls.append(self._current)
        self._current = None

    def _gen_commentary(self, b: BallRecord) -> str:
        import random

        shot_name = b.shot_type.replace("_", " ") if b.shot_type != "unknown" else ""
        length_name = b.length.replace("_", " ") if b.length != "unknown" else ""
        line_name = b.line.replace("_", " ") if b.line != "unknown" else ""

        delivery_desc = b.over
        if length_name:
            delivery_desc += f", {length_name} delivery"
        if line_name:
            delivery_desc += f" on the {line_name}"

        if b.is_wicket:
            wicket_phrases = [
                "— OUT! That's a big wicket! The fielders converge in celebration",
                "— GONE! Edged and taken! The bowler is thrilled",
                "— WICKET! That's the breakthrough they were looking for",
                "— OUT! What a delivery, the batter had no answer to that",
                "— DISMISSED! The crowd erupts, huge moment in this match",
            ]
            return f"{delivery_desc} {random.choice(wicket_phrases)}"

        if b.is_six:
            six_phrases = [
                f"— MASSIVE SIX! {shot_name}, {b.power} hit sailing over the ropes! That's gone miles into the stands",
                f"— SIX! Absolutely creamed! {shot_name} with pure timing, dispatched into the crowd",
                f"— HUGE! That's out of the ground! {shot_name}, {b.power} connection sends it into orbit",
                f"— SIX RUNS! What a strike! {shot_name} with devastating power, no chance for the fielder",
                f"— Maximum! {shot_name}, {b.power} blow! Picked the length early and punished it",
            ]
            return f"{delivery_desc} {random.choice(six_phrases)}"

        if b.is_boundary:
            four_phrases = [
                f"— FOUR! Exquisite {shot_name}, that races away to the boundary rope",
                f"— Boundary! Beautiful {shot_name}, perfectly placed through the gap",
                f"— FOUR RUNS! Superb {shot_name}, the fielder gives chase but it's in vain",
                f"— That's four! Elegant {shot_name}, timing and placement in perfect harmony",
                f"— Boundary ball! Crisp {shot_name}, too quick for the fielding side",
            ]
            return f"{delivery_desc} {random.choice(four_phrases)}"

        if b.is_dot:
            if shot_name:
                dot_phrases = [
                    f"— {shot_name}, beaten! Dot ball, excellent bowling",
                    f"— played and missed, {shot_name} attempt finds nothing but air",
                    f"— {shot_name}, straight to the fielder. No run, building pressure",
                    f"— {shot_name}, well fielded! Can't get it away, dot ball",
                    f"— defended solidly, {shot_name} keeps out a good delivery",
                ]
            else:
                dot_phrases = [
                    "— dot ball. No run, tidy bowling keeping the batter honest",
                    "— left alone outside off, good judgement from the batter",
                    "— beaten on the outside edge! No run, the bowler is on top here",
                    "— blocked back down the pitch, solid defence, no run",
                    "— dot ball, the pressure mounts. Good tight line and length",
                ]
            return f"{delivery_desc} {random.choice(dot_phrases)}"

        if b.runs == 1:
            single_phrases = [
                "— nudged away for a single, good rotation of strike",
                "— pushed into the gap, quick single taken with sharp running",
                "— worked off the pads for one, ticking the scoreboard over",
                "— dabbed to third man, easy single. Smart cricket",
                "— soft hands, placed into the gap for a comfortable single",
            ]
            return f"{delivery_desc} {random.choice(single_phrases)}"

        if b.runs == 2:
            double_phrases = [
                "— driven into the gap, excellent running between the wickets, two taken",
                "— turned for two! Good placement and quick running makes the difference",
                "— tucked away, hustled back for the second run. Great intent shown",
                "— punched off the back foot, two runs with energetic running",
            ]
            return f"{delivery_desc} {random.choice(double_phrases)}"

        if b.runs == 3:
            triple_phrases = [
                "— driven hard, outstanding running! Three taken, the fielder fumbles at the boundary",
                "— excellent running between the wickets, three runs! That's athletic cricket",
                "— misfield at the boundary! Three runs taken, the batting side capitalizes",
            ]
            return f"{delivery_desc} {random.choice(triple_phrases)}"

        if b.runs > 0:
            return f"{delivery_desc} — {b.runs} runs, good work from the batting side"

        return delivery_desc

    def build_scorecard(self) -> dict:
        total_runs = sum(b.runs for b in self._balls)
        wkts = sum(1 for b in self._balls if b.is_wicket)
        n = len(self._balls)
        fours = sum(1 for b in self._balls if b.is_boundary and not b.is_six)
        sixes = sum(1 for b in self._balls if b.is_six)
        dots = sum(1 for b in self._balls if b.is_dot)
        ov_c = n // 6
        ov_r = n % 6
        rr = round(total_runs / max(n / 6, 0.1), 2) if n > 0 else 0.0
        return {
            "total_runs": total_runs, "total_wickets": wkts,
            "score": f"{total_runs}/{wkts}", "overs": f"{ov_c}.{ov_r}",
            "total_balls": n, "run_rate": rr,
            "fours": fours, "sixes": sixes, "dot_balls": dots,
            "dot_pct": round(dots / max(n, 1) * 100, 1),
            "boundary_pct": round((fours + sixes) / max(n, 1) * 100, 1),
            "partnership": {"runs": self._partnership_runs, "balls": self._partnership_balls},
        }

    def build_pitch_map(self) -> list[dict]:
        return [{
            "ball": b.ball_number, "over": b.over,
            "bx": b.bounce_x, "by": b.bounce_y,
            "length": b.length, "line": b.line,
            "runs": b.runs, "wicket": b.is_wicket,
        } for b in self._balls if b.bounce_x > 0 or b.bounce_y > 0]

    def build_wagon_wheel(self) -> list[dict]:
        return [{
            "ball": b.ball_number, "over": b.over,
            "zone": b.wagon_zone, "angle": b.exit_angle,
            "shot": b.shot_type, "runs": b.runs,
            "boundary": b.is_boundary, "six": b.is_six,
            "power": b.power,
        } for b in self._balls if b.shot_type != "unknown"]

    def build_timeline(self) -> list[dict]:
        return [{
            "ball": b.ball_number, "over": b.over,
            "runs": b.runs, "boundary": b.is_boundary,
            "six": b.is_six, "wicket": b.is_wicket,
            "dot": b.is_dot, "shot": b.shot_type,
            "length": b.length, "line": b.line,
            "zone": b.wagon_zone,
            "commentary": b.commentary,
            "ts_start": b.ts_start, "ts_end": b.ts_end,
        } for b in self._balls]

    def build_over_breakdown(self) -> list[dict]:
        overs: dict[int, list[BallRecord]] = {}
        for b in self._balls:
            ov_num = (b.ball_number - 1) // 6
            overs.setdefault(ov_num, []).append(b)
        result = []
        for ov_num in sorted(overs):
            balls = overs[ov_num]
            r = sum(b.runs for b in balls)
            w = sum(1 for b in balls if b.is_wicket)
            d = sum(1 for b in balls if b.is_dot)
            result.append({
                "over": ov_num + 1, "runs": r, "wickets": w,
                "dots": d, "balls": len(balls),
            })
        return result

    def build_momentum(self) -> list[dict]:
        cum = 0
        out = []
        for b in self._balls:
            cum += b.runs
            out.append({
                "ball": b.ball_number, "over": b.over,
                "cum_runs": cum, "this_ball": b.runs,
            })
        return out

    def build_pressure_index(self) -> list[dict]:

        if len(self._balls) < 6:
            return []
        result = []
        for i in range(5, len(self._balls)):
            window = self._balls[i-5:i+1]
            dots = sum(1 for b in window if b.is_dot)
            wkts = sum(1 for b in window if b.is_wicket)
            pressure = round((dots / 6) * 100 + wkts * 30, 1)
            result.append({
                "ball": self._balls[i].ball_number,
                "over": self._balls[i].over,
                "pressure_index": min(pressure, 100),
            })
        return result

    def build_delivery_summary(self) -> dict:
        n = len(self._balls)
        if n == 0:
            return {}
        fours = sum(1 for b in self._balls if b.is_boundary and not b.is_six)
        sixes = sum(1 for b in self._balls if b.is_six)
        dots = sum(1 for b in self._balls if b.is_dot)
        ov_c = n // 6
        ov_r = n % 6
        return {
            "total_deliveries": n,
            "estimated_overs": f"{ov_c}.{ov_r}",
            "boundary_pct": round((fours + sixes) / max(n, 1) * 100, 1),
            "dot_pct": round(dots / max(n, 1) * 100, 1),
            "bat_impacts": sum(1 for b in self._balls if b.shot_type != "unknown"),
            "fours": fours,
            "sixes": sixes,
        }

    def build_full_analytics(self) -> dict:
        return {
            "scorecard": self.build_scorecard(),
            "timeline": self.build_timeline(),
            "pitch_map": self.build_pitch_map(),
            "wagon_wheel": self.build_wagon_wheel(),
            "momentum": self.build_momentum(),
            "over_breakdown": self.build_over_breakdown(),
            "pressure_index": self.build_pressure_index(),
            "delivery_summary": self.build_delivery_summary(),
            "commentary": [b.commentary for b in self._balls if b.commentary],
        }

    def reset(self):
        self._balls.clear()
        self._current = None
        self._count = 0
        self._partnership_runs = 0
        self._partnership_balls = 0
