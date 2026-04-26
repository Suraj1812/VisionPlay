from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_FEATURES = [
    "clip profile routing",
    "cricket role inference",
    "delivery timeline",
    "ball path extraction",
    "subtitle track generation",
]

UNAVAILABLE_FEATURES = [
    "named player identification",
    "named bowler identification",
    "fully trusted scoreboard OCR for action-cam clips",
]


@dataclass
class ScoreState:
    score: str
    runs: int
    wickets: int
    overs: str
    overs_float: float
    run_rate: float
    batting_team: str
    confidence: float
    frame_id: int
    timestamp_ms: int


def build_cricket_package(
    *,
    heuristic_analytics: dict[str, Any],
    score_timeline: list[dict[str, Any]],
    team_summary: dict[str, Any],
    delivery_summary: dict[str, Any],
    visual_events: list[dict[str, Any]],
    last_score: Any,
    camera_cuts: int,
    ball_trajectory: list[tuple[float, float]] | list[list[float]] | None,
    transcript: dict[str, Any] | None = None,
    profile_report: dict[str, Any] | None = None,
    fps: float = 25.0,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> dict[str, Any]:
    normalized_profile = _normalize_profile_report(profile_report or {})
    normalized_timeline = _normalize_score_timeline(score_timeline)
    scorecard = _build_scorecard(
        normalized_timeline,
        heuristic_analytics.get("scorecard", {}),
        last_score,
        allow_scoreboard=normalized_profile["overlay_present"],
    )
    standardized_events = _standardize_visual_events(visual_events)
    team_analysis = _summarize_team_analysis(team_summary)
    deliveries = _build_deliveries(
        heuristic_timeline=heuristic_analytics.get("timeline", []),
        delivery_summary=delivery_summary,
        standardized_events=standardized_events,
        team_analysis=team_analysis,
    )
    feed_entries = _build_feed_entries(
        deliveries=deliveries,
        scorecard=scorecard,
        standardized_events=standardized_events,
        allow_scoreboard=normalized_profile["overlay_present"],
        scoreboard_timeline=normalized_timeline,
    )
    ball_path = _build_ball_path(
        ball_trajectory=ball_trajectory or [],
        events=standardized_events,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    speech = _build_speech(transcript or {})
    subtitles = _build_subtitles(
        deliveries=deliveries,
        speech=speech,
        scorecard=scorecard,
    )
    quality = _build_quality(
        deliveries=deliveries,
        ball_path=ball_path,
        roles=team_analysis.get("roles", {}),
        speech=speech,
        events=standardized_events,
    )
    warnings = _build_warnings(
        normalized_profile=normalized_profile,
        scorecard=scorecard,
        speech=speech,
        team_analysis=team_analysis,
    )
    capabilities = {
        "mode": "specialized" if normalized_profile["specialized"] else "fallback",
        "profile": normalized_profile["profile"],
        "score_source": scorecard.get("source", "visual-heuristic"),
        "feed_source": "delivery-timeline" if deliveries else "event-feed" if standardized_events else "unavailable",
        "supported_features": SUPPORTED_FEATURES,
        "unavailable_features": UNAVAILABLE_FEATURES,
        "overlay_present": normalized_profile["overlay_present"],
        "scoreboard_samples": len(normalized_timeline),
        "subtitle_mode": subtitles.get("mode", "none"),
    }

    return {
        "profile": normalized_profile["profile"],
        "mode": capabilities["mode"],
        "scorecard": scorecard,
        "timeline": feed_entries,
        "pitch_map": heuristic_analytics.get("pitch_map", []),
        "wagon_wheel": heuristic_analytics.get("wagon_wheel", []),
        "momentum": heuristic_analytics.get("momentum", []),
        "over_breakdown": heuristic_analytics.get("over_breakdown", []),
        "pressure_index": heuristic_analytics.get("pressure_index", []),
        "commentary": [entry["commentary"] for entry in feed_entries if entry.get("commentary")],
        "events": standardized_events,
        "deliveries": deliveries,
        "delivery_summary": _build_delivery_summary(deliveries, scorecard, delivery_summary),
        "teams": team_analysis.get("teams", {}),
        "roles": team_analysis.get("roles", {}),
        "team_analysis": team_analysis,
        "ball_path": ball_path,
        "ball_trajectory": ball_path.get("points", []),
        "camera_cuts": int(camera_cuts),
        "score_timeline": [state.__dict__ for state in normalized_timeline],
        "scoreboard_last": {
            "score": scorecard.get("score", ""),
            "overs": scorecard.get("overs", ""),
            "run_rate": scorecard.get("run_rate", 0.0),
            "batting_team": scorecard.get("batting_team", ""),
        },
        "speech": speech,
        "transcript": speech,
        "subtitles": subtitles,
        "quality": quality,
        "capabilities": capabilities,
        "warnings": warnings,
        "profile_report": normalized_profile,
        "fps": fps,
        "frame_size": {"width": frame_width, "height": frame_height},
    }


def _normalize_profile_report(profile_report: dict[str, Any]) -> dict[str, Any]:
    profile = str(profile_report.get("profile") or "generic").strip() or "generic"
    confidence = _coerce_float(profile_report.get("confidence"), default=0.0)
    overlay_present = bool(profile_report.get("overlay_present", False))
    specialized = bool(profile_report.get("specialized", False)) and confidence >= 0.4
    return {
        "profile": profile,
        "confidence": confidence,
        "specialized": specialized,
        "overlay_present": overlay_present,
        "reasons": list(profile_report.get("reasons") or []),
        "sample_count": _coerce_int(profile_report.get("sample_count"), default=0, minimum=0),
    }


def _normalize_score_timeline(score_timeline: list[dict[str, Any]]) -> list[ScoreState]:
    normalized: list[ScoreState] = []
    seen_states: set[tuple[int, int, str]] = set()
    for item in score_timeline:
        if not isinstance(item, dict):
            continue
        runs = _coerce_int(item.get("runs"), default=-1)
        wickets = _coerce_int(item.get("wickets"), default=-1)
        overs = str(item.get("overs", "")).strip()
        overs_float = _coerce_float(item.get("overs_float"), default=-1.0)
        if runs < 0 or wickets < 0 or not overs or overs_float < 0:
            continue
        key = (runs, wickets, overs)
        if key in seen_states:
            continue
        seen_states.add(key)
        normalized.append(
            ScoreState(
                score=str(item.get("score", f"{runs}/{wickets}")).strip() or f"{runs}/{wickets}",
                runs=runs,
                wickets=wickets,
                overs=overs,
                overs_float=overs_float,
                run_rate=_coerce_float(item.get("run_rate"), default=0.0),
                batting_team=str(item.get("batting_team", "")).strip(),
                confidence=_coerce_float(item.get("confidence"), default=0.0),
                frame_id=_coerce_int(item.get("frame_id"), default=0, minimum=0),
                timestamp_ms=_coerce_int(item.get("timestamp_ms"), default=0, minimum=0),
            )
        )
    normalized.sort(key=lambda state: (state.frame_id, state.timestamp_ms, state.overs_float))
    return normalized


def _build_scorecard(
    timeline: list[ScoreState],
    heuristic_scorecard: dict[str, Any],
    last_score: Any,
    allow_scoreboard: bool,
) -> dict[str, Any]:
    if allow_scoreboard and timeline:
        last_state = timeline[-1]
        total_balls = _overs_to_balls(last_state.overs_float)
        return {
            "total_runs": last_state.runs,
            "total_wickets": last_state.wickets,
            "score": last_state.score,
            "overs": last_state.overs,
            "total_balls": total_balls,
            "run_rate": round(last_state.run_rate or 0.0, 2),
            "fours": _coerce_int(heuristic_scorecard.get("fours"), default=0, minimum=0),
            "sixes": _coerce_int(heuristic_scorecard.get("sixes"), default=0, minimum=0),
            "dot_balls": _coerce_int(heuristic_scorecard.get("dot_balls"), default=0, minimum=0),
            "dot_pct": _coerce_float(heuristic_scorecard.get("dot_pct"), default=0.0),
            "boundary_pct": _coerce_float(heuristic_scorecard.get("boundary_pct"), default=0.0),
            "partnership": heuristic_scorecard.get("partnership", {"runs": 0, "balls": 0}),
            "batting_team": last_state.batting_team or getattr(last_score, "batting_team", ""),
            "source": "scoreboard",
            "scoreboard_samples": len(timeline),
        }

    fallback = dict(heuristic_scorecard or {})
    fallback.setdefault("total_runs", 0)
    fallback.setdefault("total_wickets", 0)
    fallback.setdefault("score", "0/0")
    fallback.setdefault("overs", "0.0")
    fallback.setdefault("total_balls", 0)
    fallback.setdefault("run_rate", 0.0)
    fallback.setdefault("fours", 0)
    fallback.setdefault("sixes", 0)
    fallback.setdefault("dot_balls", 0)
    fallback.setdefault("dot_pct", 0.0)
    fallback.setdefault("boundary_pct", 0.0)
    fallback.setdefault("partnership", {"runs": 0, "balls": 0})
    fallback.setdefault("batting_team", getattr(last_score, "batting_team", ""))
    fallback.setdefault("source", "visual-heuristic")
    fallback.setdefault("scoreboard_samples", 0)
    return fallback


def _standardize_visual_events(visual_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standardized: list[dict[str, Any]] = []
    delivery_index = 0
    for raw_event in visual_events:
        if not isinstance(raw_event, dict):
            continue
        raw_type = str(raw_event.get("event_type", "")).strip()
        details = raw_event.get("details") if isinstance(raw_event.get("details"), dict) else {}
        timestamp_ms = _coerce_int(raw_event.get("timestamp_ms"), default=0, minimum=0)
        confidence = _coerce_float(raw_event.get("confidence"), default=0.0)
        event_type = raw_type
        if raw_type == "wicket":
            event_type = "wicket_event"
        elif raw_type == "four":
            event_type = "boundary_likely"
            details = {**details, "runs": 4, "result": "four"}
        elif raw_type == "six":
            event_type = "boundary_likely"
            details = {**details, "runs": 6, "result": "six"}
        elif raw_type == "dot_ball":
            event_type = "dead_ball"
        elif raw_type == "ball_released":
            delivery_index += 1
            standardized.append(
                {
                    "event_type": "delivery_start",
                    "timestamp_ms": timestamp_ms,
                    "confidence": confidence,
                    "delivery_index": delivery_index,
                    "details": {"delivery_index": delivery_index},
                }
            )
        if raw_type == "bat_impact":
            shot_type = str(details.get("shot_type", "unknown")).strip().lower()
            standardized.append(
                {
                    "event_type": "attacking_shot" if shot_type not in {"defensive", "leave", "unknown"} else "defensive_shot",
                    "timestamp_ms": timestamp_ms,
                    "confidence": confidence,
                    "delivery_index": delivery_index,
                    "details": details,
                }
            )
        standardized.append(
            {
                "event_type": event_type,
                "timestamp_ms": timestamp_ms,
                "confidence": confidence,
                "delivery_index": delivery_index,
                "details": details,
            }
        )
        if raw_type in {"wicket", "four", "six", "dot_ball", "over_complete"}:
            standardized.append(
                {
                    "event_type": "delivery_end",
                    "timestamp_ms": timestamp_ms,
                    "confidence": confidence,
                    "delivery_index": delivery_index,
                    "details": {"result": raw_type},
                }
            )
    return standardized


def _build_deliveries(
    heuristic_timeline: list[dict[str, Any]],
    delivery_summary: dict[str, Any],
    standardized_events: list[dict[str, Any]],
    team_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    deliveries: list[dict[str, Any]] = []
    if heuristic_timeline:
        for index, entry in enumerate(heuristic_timeline, start=1):
            if not isinstance(entry, dict):
                continue
            deliveries.append(
                {
                    "delivery_id": f"d{index}",
                    "ball": _coerce_int(entry.get("ball"), default=index, minimum=1),
                    "over": str(entry.get("over", f"0.{index}")),
                    "ts_start": _coerce_int(entry.get("ts_start"), default=max((index - 1) * 4000, 0), minimum=0),
                    "ts_end": _coerce_int(entry.get("ts_end"), default=index * 4000, minimum=0),
                    "runs": _coerce_int(entry.get("runs"), default=0, minimum=0),
                    "result": _delivery_result_from_entry(entry),
                    "shot": str(entry.get("shot", "unknown")),
                    "length": str(entry.get("length", "unknown")),
                    "line": str(entry.get("line", "unknown")),
                    "commentary": str(entry.get("commentary") or entry.get("phase_summary") or f"Delivery {index}"),
                    "confidence": _coerce_float(entry.get("confidence"), default=0.55),
                    "events": _events_for_window(
                        standardized_events,
                        start_ms=_coerce_int(entry.get("ts_start"), default=max((index - 1) * 4000, 0), minimum=0),
                        end_ms=_coerce_int(entry.get("ts_end"), default=index * 4000, minimum=0),
                    ),
                    "role_snapshot": _role_snapshot(team_analysis),
                }
            )
    elif standardized_events:
        grouped: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for event in standardized_events:
            if event["event_type"] == "delivery_start" and current:
                grouped.append(current)
                current = [event]
            else:
                current.append(event)
                if event["event_type"] == "delivery_end":
                    grouped.append(current)
                    current = []
        if current:
            grouped.append(current)

        for index, events in enumerate(grouped, start=1):
            start_ms = min((_coerce_int(event.get("timestamp_ms"), default=0, minimum=0) for event in events), default=0)
            end_ms = max((_coerce_int(event.get("timestamp_ms"), default=0, minimum=0) for event in events), default=start_ms) + 1800
            result = "delivery"
            runs = 0
            for event in events:
                details = event.get("details") if isinstance(event.get("details"), dict) else {}
                if event["event_type"] == "wicket_event":
                    result = "wicket"
                if event["event_type"] == "boundary_likely":
                    runs = max(runs, _coerce_int(details.get("runs"), default=4, minimum=0))
                    result = details.get("result", "boundary")
            deliveries.append(
                {
                    "delivery_id": f"d{index}",
                    "ball": index,
                    "over": f"{(index - 1) // 6}.{((index - 1) % 6) + 1}",
                    "ts_start": start_ms,
                    "ts_end": end_ms,
                    "runs": runs,
                    "result": result,
                    "shot": "unknown",
                    "length": "unknown",
                    "line": "unknown",
                    "commentary": _delivery_commentary_from_events(events, index),
                    "confidence": round(sum(_coerce_float(event.get("confidence"), default=0.0) for event in events) / max(len(events), 1), 4),
                    "events": events,
                    "role_snapshot": _role_snapshot(team_analysis),
                }
            )

    if not deliveries:
        total_deliveries = _coerce_int(delivery_summary.get("total_deliveries"), default=0, minimum=0)
        for index in range(1, total_deliveries + 1):
            deliveries.append(
                {
                    "delivery_id": f"d{index}",
                    "ball": index,
                    "over": f"{(index - 1) // 6}.{((index - 1) % 6) + 1}",
                    "ts_start": (index - 1) * 4000,
                    "ts_end": index * 4000,
                    "runs": 0,
                    "result": "delivery",
                    "shot": "unknown",
                    "length": "unknown",
                    "line": "unknown",
                    "commentary": f"Delivery {index}",
                    "confidence": 0.4,
                    "events": [],
                    "role_snapshot": _role_snapshot(team_analysis),
                }
            )
    return deliveries


def _build_feed_entries(
    deliveries: list[dict[str, Any]],
    scorecard: dict[str, Any],
    standardized_events: list[dict[str, Any]],
    allow_scoreboard: bool,
    scoreboard_timeline: list[ScoreState],
) -> list[dict[str, Any]]:
    if deliveries:
        return [
            {
                "ball": delivery["ball"],
                "over": delivery["over"],
                "runs": delivery["runs"],
                "boundary": delivery["result"] in {"four", "boundary"},
                "six": delivery["result"] == "six",
                "wicket": delivery["result"] == "wicket",
                "dot": delivery["runs"] == 0 and delivery["result"] not in {"wicket"},
                "shot": delivery["shot"],
                "length": delivery["length"],
                "line": delivery["line"],
                "zone": 0,
                "commentary": delivery["commentary"],
                "ts_start": delivery["ts_start"],
                "ts_end": delivery["ts_end"],
                "score": scorecard.get("score", ""),
                "detail": _delivery_detail(delivery),
                "source": "delivery-timeline",
                "confidence": delivery["confidence"],
                "events": delivery["events"],
            }
            for delivery in deliveries
        ]

    if allow_scoreboard and scoreboard_timeline:
        entries: list[dict[str, Any]] = []
        for index, state in enumerate(scoreboard_timeline):
            previous = scoreboard_timeline[index - 1] if index > 0 else None
            delta_runs = state.runs - previous.runs if previous else 0
            entries.append(
                {
                    "ball": max(index + 1, _overs_to_balls(state.overs_float)),
                    "over": state.overs,
                    "runs": max(delta_runs, 0),
                    "boundary": delta_runs == 4,
                    "six": delta_runs >= 6,
                    "wicket": previous is not None and state.wickets > previous.wickets,
                    "dot": previous is not None and delta_runs == 0,
                    "shot": "unknown",
                    "length": "unknown",
                    "line": "unknown",
                    "zone": 0,
                    "commentary": f"Score moved to {state.score} at {state.overs} overs.",
                    "ts_start": state.timestamp_ms,
                    "ts_end": state.timestamp_ms + 3500,
                    "score": state.score,
                    "detail": f"RR {state.run_rate:.2f}",
                    "source": "scoreboard",
                    "confidence": state.confidence,
                }
            )
        return entries

    return [
        {
            "ball": index + 1,
            "over": f"{index // 6}.{(index % 6) + 1}",
            "runs": _coerce_int(event.get("details", {}).get("runs"), default=0, minimum=0),
            "boundary": event["event_type"] == "boundary_likely",
            "six": event["event_type"] == "boundary_likely" and _coerce_int(event.get("details", {}).get("runs"), default=0, minimum=0) >= 6,
            "wicket": event["event_type"] == "wicket_event",
            "dot": event["event_type"] == "dead_ball",
            "shot": str(event.get("details", {}).get("shot_type", "unknown")),
            "length": str(event.get("details", {}).get("length", "unknown")),
            "line": str(event.get("details", {}).get("line", "unknown")),
            "zone": _coerce_int(event.get("details", {}).get("wagon_zone"), default=0, minimum=0),
            "commentary": _event_label(event),
            "ts_start": _coerce_int(event.get("timestamp_ms"), default=0, minimum=0),
            "ts_end": _coerce_int(event.get("timestamp_ms"), default=0, minimum=0) + 2200,
            "score": scorecard.get("score", ""),
            "detail": "",
            "source": "event-feed",
            "confidence": _coerce_float(event.get("confidence"), default=0.0),
        }
        for index, event in enumerate(standardized_events)
    ]


def _summarize_team_analysis(team_summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(team_summary, dict) or not team_summary:
        return {
            "camera_profile": "default",
            "calibrated": False,
            "teams": {},
            "roles": {},
            "batting_team_id": -1,
            "fielding_team_id": -1,
            "role_method": "unavailable",
        }
    return {
        "camera_profile": str(team_summary.get("camera_profile", "default")),
        "calibrated": bool(team_summary.get("calibrated", False)),
        "teams": team_summary.get("teams", {}) if isinstance(team_summary.get("teams"), dict) else {},
        "roles": team_summary.get("roles", {}) if isinstance(team_summary.get("roles"), dict) else {},
        "batting_team_id": _coerce_int(team_summary.get("batting_team_id"), default=-1),
        "fielding_team_id": _coerce_int(team_summary.get("fielding_team_id"), default=-1),
        "role_method": str(team_summary.get("role_method", "heuristic")),
    }


def _build_ball_path(
    ball_trajectory: list[tuple[float, float]] | list[list[float]],
    events: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    points = [
        {
            "index": index,
            "x": round(float(point[0]), 2),
            "y": round(float(point[1]), 2),
            "x_ratio": round(float(point[0]) / max(frame_width, 1), 4),
            "y_ratio": round(float(point[1]) / max(frame_height, 1), 4),
        }
        for index, point in enumerate(ball_trajectory)
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    release_event = _first_event(events, "ball_released")
    bounce_event = _first_event(events, "ball_bounced")
    contact_event = _first_event(events, "bat_contact") or _first_event(events, "attacking_shot") or _first_event(events, "defensive_shot")
    wicket_event = _first_event(events, "wicket_event")
    anchors = {
        "release": _anchor_from_event_or_point(release_event, points, fallback_index=0),
        "bounce": _anchor_from_event_or_point(bounce_event, points, fallback_index=max(len(points) // 3, 0)),
        "contact": _anchor_from_event_or_point(contact_event, points, fallback_index=max(len(points) // 2, 0)),
        "wicket_line": _anchor_from_event_or_point(wicket_event, points, fallback_index=len(points) - 1),
    }
    coverage = min(len(points) / 18.0, 1.0) if points else 0.0
    confidence = min(
        coverage * 0.55
        + (0.15 if anchors["release"] else 0.0)
        + (0.15 if anchors["bounce"] else 0.0)
        + (0.15 if anchors["contact"] else 0.0),
        1.0,
    )
    return {
        "points": points,
        "anchors": anchors,
        "coverage": round(coverage, 4),
        "confidence": round(confidence, 4),
    }


def _build_speech(transcript: dict[str, Any]) -> dict[str, Any]:
    segments = transcript.get("segments") if isinstance(transcript.get("segments"), list) else []
    confidence = _coerce_float(transcript.get("confidence"), default=0.0)
    speech_present = bool(transcript.get("speech_present", False)) and confidence >= 0.25
    mode = "speech-led" if speech_present else "event-led"
    return {
        "status": str(transcript.get("status", "unavailable")),
        "source": str(transcript.get("source", "none")),
        "language": str(transcript.get("language", "")),
        "confidence": confidence,
        "speech_present": speech_present,
        "segments": [
            {
                "start_ms": _coerce_int(segment.get("start_ms"), default=0, minimum=0),
                "end_ms": _coerce_int(segment.get("end_ms"), default=0, minimum=0),
                "text": str(segment.get("text", "")).strip(),
                "confidence": _coerce_float(segment.get("confidence"), default=0.0),
            }
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("text", "")).strip()
        ],
        "mode": mode,
        "reason": str(transcript.get("reason", "")),
    }


def _build_subtitles(
    deliveries: list[dict[str, Any]],
    speech: dict[str, Any],
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    cues: list[dict[str, Any]] = []
    if speech.get("speech_present") and speech.get("segments"):
        for segment in speech["segments"]:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            delivery = _find_delivery_for_time(deliveries, _coerce_int(segment.get("start_ms"), default=0, minimum=0))
            prefix = f"{delivery['over']} · " if delivery else ""
            cues.append(
                {
                    "start_ms": _coerce_int(segment.get("start_ms"), default=0, minimum=0),
                    "end_ms": max(
                        _coerce_int(segment.get("end_ms"), default=0, minimum=0),
                        _coerce_int(segment.get("start_ms"), default=0, minimum=0) + 900,
                    ),
                    "text": f"{prefix}{text}",
                    "source": "speech",
                    "confidence": _coerce_float(segment.get("confidence"), default=0.0),
                }
            )

    if not cues:
        for delivery in deliveries:
            cues.append(
                {
                    "start_ms": delivery["ts_start"],
                    "end_ms": max(delivery["ts_end"], delivery["ts_start"] + 1200),
                    "text": _subtitle_text_for_delivery(delivery, scorecard),
                    "source": "event",
                    "confidence": delivery["confidence"],
                }
            )

    return {
        "mode": "speech-led" if speech.get("speech_present") and speech.get("segments") else "event-led",
        "cues": cues,
    }


def _build_quality(
    deliveries: list[dict[str, Any]],
    ball_path: dict[str, Any],
    roles: dict[str, Any],
    speech: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    role_confidences = [
        _coerce_float(role.get("confidence"), default=0.0)
        for role in roles.values()
        if isinstance(role, dict)
    ]
    event_confidences = [_coerce_float(event.get("confidence"), default=0.0) for event in events]
    ball_visibility = min(len(ball_path.get("points", [])) / max(len(deliveries) * 6, 1), 1.0) if deliveries else min(len(ball_path.get("points", [])) / 12.0, 1.0)
    return {
        "ball_visibility": round(ball_visibility, 4),
        "ball_path_confidence": _coerce_float(ball_path.get("confidence"), default=0.0),
        "role_confidence": round(sum(role_confidences) / len(role_confidences), 4) if role_confidences else 0.0,
        "event_confidence": round(sum(event_confidences) / len(event_confidences), 4) if event_confidences else 0.0,
        "subtitle_mode": str(speech.get("mode", "event-led")),
        "transcript_confidence": _coerce_float(speech.get("confidence"), default=0.0),
    }


def _build_warnings(
    normalized_profile: dict[str, Any],
    scorecard: dict[str, Any],
    speech: dict[str, Any],
    team_analysis: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if normalized_profile["profile"] == "cricket_end_on_action_cam_v1" and not normalized_profile["overlay_present"]:
        warnings.append("No scoreboard overlay was detected, so match state is driven by visual cricket analysis.")
    if scorecard.get("source") != "scoreboard":
        warnings.append("Score and over figures are heuristic unless a real overlay is detected.")
    if not speech.get("speech_present"):
        warnings.append("Audio captions are event-led because no confident speech transcript was available.")
    if not team_analysis.get("calibrated"):
        warnings.append("Team sides are approximate until jersey colors stay stable for long enough.")
    return warnings


def _build_delivery_summary(
    deliveries: list[dict[str, Any]],
    scorecard: dict[str, Any],
    fallback_summary: dict[str, Any],
) -> dict[str, Any]:
    if not deliveries:
        summary = dict(fallback_summary or {})
        summary.setdefault("total_deliveries", 0)
        summary.setdefault("estimated_overs", "0.0")
        summary.setdefault("source", "none")
        return summary
    wickets = sum(1 for delivery in deliveries if delivery["result"] == "wicket")
    boundaries = sum(1 for delivery in deliveries if delivery["result"] in {"four", "six", "boundary"})
    dots = sum(1 for delivery in deliveries if delivery["runs"] == 0 and delivery["result"] != "wicket")
    return {
        "total_deliveries": len(deliveries),
        "estimated_overs": f"{len(deliveries) // 6}.{len(deliveries) % 6}",
        "fours": sum(1 for delivery in deliveries if delivery["result"] == "four"),
        "sixes": sum(1 for delivery in deliveries if delivery["result"] == "six"),
        "wickets": wickets,
        "dot_balls": dots,
        "boundary_pct": round((boundaries / max(len(deliveries), 1)) * 100, 1),
        "dot_pct": round((dots / max(len(deliveries), 1)) * 100, 1),
        "source": "specialized",
        "score": scorecard.get("score", "0/0"),
    }


def _role_snapshot(team_analysis: dict[str, Any]) -> dict[str, Any]:
    roles = team_analysis.get("roles", {})
    if not isinstance(roles, dict):
        return {}
    snapshot: dict[str, Any] = {}
    for tracking_id, role_data in roles.items():
        if not isinstance(role_data, dict):
            continue
        role_name = str(role_data.get("role", "")).strip()
        if role_name in {"striker", "non_striker", "bowler", "wicketkeeper", "umpire"}:
            snapshot[role_name] = {
                "tracking_id": _coerce_int(tracking_id, default=-1),
                "team_side": str(role_data.get("team_side", "unknown")),
                "confidence": _coerce_float(role_data.get("confidence"), default=0.0),
            }
    return snapshot


def _events_for_window(events: list[dict[str, Any]], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if start_ms - 200 <= _coerce_int(event.get("timestamp_ms"), default=0, minimum=0) <= end_ms + 400
    ]


def _delivery_result_from_entry(entry: dict[str, Any]) -> str:
    if entry.get("wicket"):
        return "wicket"
    if entry.get("six"):
        return "six"
    if entry.get("boundary"):
        return "four"
    if _coerce_int(entry.get("runs"), default=0, minimum=0) == 0:
        return "dot"
    return "runs"


def _delivery_commentary_from_events(events: list[dict[str, Any]], index: int) -> str:
    labels = [_event_label(event) for event in events if _event_label(event)]
    if labels:
        return " · ".join(labels[:2])
    return f"Delivery {index}"


def _delivery_detail(delivery: dict[str, Any]) -> str:
    parts = []
    if delivery["shot"] != "unknown":
        parts.append(str(delivery["shot"]).replace("_", " "))
    if delivery["length"] != "unknown":
        parts.append(str(delivery["length"]).replace("_", " "))
    if delivery["line"] != "unknown":
        parts.append(str(delivery["line"]).replace("_", " "))
    return " · ".join(parts)


def _subtitle_text_for_delivery(delivery: dict[str, Any], scorecard: dict[str, Any]) -> str:
    prefix = f"{delivery['over']} · "
    if delivery["result"] == "wicket":
        return f"{prefix}Wicket event"
    if delivery["result"] == "six":
        return f"{prefix}Six over {delivery['line'].replace('_', ' ') if delivery['line'] != 'unknown' else 'the ropes'}"
    if delivery["result"] == "four":
        return f"{prefix}Boundary"
    detail = _delivery_detail(delivery)
    if detail:
        return f"{prefix}{detail}"
    score = scorecard.get("score")
    return f"{prefix}{score}" if score else prefix.strip()


def _find_delivery_for_time(deliveries: list[dict[str, Any]], timestamp_ms: int) -> dict[str, Any] | None:
    for delivery in deliveries:
        if delivery["ts_start"] <= timestamp_ms <= delivery["ts_end"] + 500:
            return delivery
    return None


def _event_label(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", "")).strip()
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    labels = {
        "delivery_start": "Delivery starts",
        "ball_released": "Ball released",
        "ball_bounced": "Ball pitched",
        "bat_contact": "Bat contact",
        "attacking_shot": "Attacking shot",
        "defensive_shot": "Defensive shot",
        "leave": "Left alone",
        "run_attempt": "Run attempt",
        "boundary_likely": f"{_coerce_int(details.get('runs'), default=4, minimum=0) or 4} likely",
        "wicket_event": "Wicket event",
        "delivery_end": "Delivery ends",
        "dead_ball": "Dead ball",
    }
    return labels.get(event_type, event_type.replace("_", " ").strip().title())


def _first_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in events:
        if str(event.get("event_type", "")).strip() == event_type:
            return event
    return None


def _anchor_from_event_or_point(
    event: dict[str, Any] | None,
    points: list[dict[str, Any]],
    fallback_index: int,
) -> dict[str, Any] | None:
    if event:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if "bounce_x" in details and "bounce_y" in details:
            return {
                "timestamp_ms": _coerce_int(event.get("timestamp_ms"), default=0, minimum=0),
                "x": _coerce_float(details.get("bounce_x"), default=0.0),
                "y": _coerce_float(details.get("bounce_y"), default=0.0),
            }
        if points:
            point = points[min(max(fallback_index, 0), len(points) - 1)]
            return {
                "timestamp_ms": _coerce_int(event.get("timestamp_ms"), default=0, minimum=0),
                "x": point["x"],
                "y": point["y"],
            }
    if not points:
        return None
    point = points[min(max(fallback_index, 0), len(points) - 1)]
    return {"timestamp_ms": 0, "x": point["x"], "y": point["y"]}


def _overs_to_balls(overs_float: float) -> int:
    whole = int(max(overs_float, 0))
    partial = int(round((overs_float - whole) * 10))
    return whole * 6 + max(partial, 0)


def _coerce_int(value: Any, default: int = 0, minimum: int | None = None) -> int:
    try:
        coerced = int(value)
    except Exception:
        coerced = default
    if minimum is not None:
        return max(coerced, minimum)
    return coerced


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        coerced = float(value)
    except Exception:
        coerced = default
    if not coerced == coerced:
        return default
    return coerced
