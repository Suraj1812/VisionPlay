from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import hypot, isfinite
from typing import Any

from backend.utils.config import settings


def build_track_analytics(
    detections: list[dict[str, Any]],
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_detections = [
        normalized
        for detection in detections
        if (normalized := _normalize_detection(detection)) is not None
    ]
    track_groups = _group_detections_by_track(normalized_detections)
    tracks: list[dict[str, Any]] = []

    for (tracking_id, object_type), points_by_frame in track_groups.items():
        ordered_points = [point for _, point in sorted(points_by_frame.items())]
        if not ordered_points:
            continue

        qualified_points = _select_best_segment(ordered_points, object_type, fps)
        track = _build_track_payload(
            tracking_id=tracking_id,
            object_type=object_type,
            class_id=_resolve_track_class_id(qualified_points),
            ordered_points=qualified_points,
            fps=fps,
        )
        if track is None:
            continue
        tracks.append(track)

    tracks.sort(key=lambda item: (-item["primary_score"], item["object_type"], item["tracking_id"]))
    primary_track_keys = {
        (track["object_type"], track["tracking_id"])
        for track in tracks[: settings.primary_track_count]
    }
    for track in tracks:
        track["is_primary"] = (track["object_type"], track["tracking_id"]) in primary_track_keys

    events = infer_track_events(tracks)
    summary = summarize_tracks_and_detections(
        detections=normalized_detections,
        tracks=tracks,
        fps=fps,
        events=events,
    )
    return tracks, summary


def summarize_tracks_and_detections(
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    fps: float,
    frame_errors: int = 0,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_detections = [
        normalized
        for detection in detections
        if (normalized := _normalize_detection(detection)) is not None
    ]
    normalized_tracks = [
        normalized
        for track in tracks
        if (normalized := _normalize_track_payload(track, fps)) is not None
    ]

    detections_by_type = Counter(item["object_type"] for item in normalized_detections)
    tracks_by_type = Counter(item["object_type"] for item in normalized_tracks)
    object_types = sorted(set(detections_by_type) | set(tracks_by_type))
    events = _normalize_events(events or infer_track_events(normalized_tracks))

    avg_track_duration = (
        sum(track["duration_ms"] for track in normalized_tracks) / len(normalized_tracks)
        if normalized_tracks
        else 0.0
    )
    avg_track_confidence = (
        sum(track["avg_confidence"] for track in normalized_tracks) / len(normalized_tracks)
        if normalized_tracks
        else 0.0
    )
    peak_speed_track = max(normalized_tracks, key=lambda item: item["max_speed_px_s"], default=None)
    interaction_count = sum(1 for event in events if event["event_type"] == "interaction")
    primary_tracks_source = [
        track for track in normalized_tracks if track.get("is_primary")
    ] or sorted(
        normalized_tracks,
        key=lambda item: item.get("primary_score", 0.0),
        reverse=True,
    )[: settings.primary_track_count]

    return {
        "total_detections": len(normalized_detections),
        "tracked_objects": len(normalized_tracks),
        "object_types": object_types,
        "detections_by_type": dict(sorted(detections_by_type.items())),
        "tracks_by_type": dict(sorted(tracks_by_type.items())),
        "primary_tracks": [
            {
                "tracking_id": track["tracking_id"],
                "object_type": track["object_type"],
                "duration_ms": track["duration_ms"],
                "distance_px": track["distance_px"],
                "max_speed_px_s": track["max_speed_px_s"],
                "avg_confidence": track["avg_confidence"],
            }
            for track in primary_tracks_source[: settings.primary_track_count]
        ],
        "avg_track_duration_ms": round(avg_track_duration, 2),
        "longest_track_duration_ms": round(
            max((track["duration_ms"] for track in normalized_tracks), default=0.0),
            2,
        ),
        "avg_track_confidence": round(avg_track_confidence, 4),
        "peak_speed_px_s": round(
            max((track["max_speed_px_s"] for track in normalized_tracks), default=0.0),
            2,
        ),
        "peak_speed_track_id": peak_speed_track["tracking_id"] if peak_speed_track is not None else None,
        "peak_speed_object_type": peak_speed_track["object_type"] if peak_speed_track is not None else None,
        "events_count": len(events),
        "interaction_count": interaction_count,
        "events": events,
        "frame_errors": max(int(frame_errors), 0),
        "fps": round(float(fps), 2) if fps > 0 else 0.0,
    }


def infer_track_events(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for track in tracks:
        events.append(
            {
                "event_type": "entry",
                "start_frame": track["frame_start"],
                "end_frame": track["frame_start"],
                "duration_frames": 1,
                "object_type": track["object_type"],
                "tracking_ids": [track["tracking_id"]],
                "details": {
                    "duration_ms": track["duration_ms"],
                    "confidence": track["avg_confidence"],
                },
            }
        )
        events.append(
            {
                "event_type": "exit",
                "start_frame": track["frame_end"],
                "end_frame": track["frame_end"],
                "duration_frames": 1,
                "object_type": track["object_type"],
                "tracking_ids": [track["tracking_id"]],
                "details": {
                    "duration_ms": track["duration_ms"],
                    "confidence": track["avg_confidence"],
                },
            }
        )

    interaction_candidates = [
        track
        for track in tracks
        if track.get("is_primary") or track["detection_count"] >= max(settings.min_track_points + 1, 3)
    ]
    interaction_candidates = interaction_candidates[: max(settings.primary_track_count * 2, 8)]

    for left_track, right_track in combinations(interaction_candidates, 2):
        left_lookup = {point["frame_id"]: point for point in left_track["path"]}
        right_lookup = {point["frame_id"]: point for point in right_track["path"]}
        shared_frames = sorted(set(left_lookup) & set(right_lookup))
        if not shared_frames:
            continue

        current_window: list[dict[str, Any]] = []
        interaction_windows: list[list[dict[str, Any]]] = []
        for frame_id in shared_frames:
            left_point = left_lookup[frame_id]
            right_point = right_lookup[frame_id]
            distance = hypot(left_point["x"] - right_point["x"], left_point["y"] - right_point["y"])
            distance_limit = max(
                24.0,
                max(left_point.get("scale", 0.0), right_point.get("scale", 0.0)) * 1.5,
                settings.tracker_step_px * settings.interaction_distance_ratio * 4.0,
            )
            if distance <= distance_limit:
                current_window.append(
                    {
                        "frame_id": frame_id,
                        "distance_px": distance,
                    }
                )
                continue

            if len(current_window) >= settings.interaction_min_consecutive_frames:
                interaction_windows.append(current_window)
            current_window = []

        if len(current_window) >= settings.interaction_min_consecutive_frames:
            interaction_windows.append(current_window)

        for window in interaction_windows:
            first_frame = window[0]["frame_id"]
            last_frame = window[-1]["frame_id"]
            avg_distance = sum(item["distance_px"] for item in window) / len(window)
            events.append(
                {
                    "event_type": "interaction",
                    "start_frame": first_frame,
                    "end_frame": last_frame,
                    "duration_frames": last_frame - first_frame + 1,
                    "object_type": "multiple",
                    "tracking_ids": [left_track["tracking_id"], right_track["tracking_id"]],
                    "details": {
                        "participants": [left_track["object_type"], right_track["object_type"]],
                        "avg_distance_px": round(avg_distance, 2),
                    },
                }
            )

    return _normalize_events(events)


def _group_detections_by_track(
    detections: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[int, dict[str, Any]]]:
    track_groups: dict[tuple[int, str], dict[int, dict[str, Any]]] = defaultdict(dict)

    for detection in detections:
        tracking_id = detection.get("tracking_id")
        if tracking_id is None:
            continue

        x1, y1, x2, y2 = detection["bbox"]
        point = {
            "frame_id": detection["frame_id"],
            "timestamp_ms": detection["timestamp_ms"],
            "x": round((x1 + x2) / 2, 2),
            "y": round((y1 + y2) / 2, 2),
            "confidence": detection["confidence"],
            "scale": round(max(x2 - x1, y2 - y1), 2),
            "class_id": detection["class_id"],
        }
        key = (tracking_id, detection["object_type"])
        existing = track_groups[key].get(detection["frame_id"])
        if existing is None or point["confidence"] >= existing["confidence"]:
            track_groups[key][detection["frame_id"]] = point

    return track_groups


def _build_track_payload(
    tracking_id: int,
    object_type: str,
    class_id: int,
    ordered_points: list[dict[str, Any]],
    fps: float,
) -> dict[str, Any] | None:
    if not ordered_points:
        return None

    stabilized_points = _stabilize_track_path_timestamps(ordered_points, fps)
    avg_confidence = sum(point["confidence"] for point in stabilized_points) / len(stabilized_points)
    max_confidence = max((point["confidence"] for point in stabilized_points), default=0.0)
    smoothed_points = _smooth_track_points(stabilized_points, object_type)
    distance_px, speed_samples = _compute_track_motion(smoothed_points, object_type, fps)
    duration_ms = _compute_duration_ms(smoothed_points, fps)
    if not _is_track_qualified(smoothed_points, distance_px, duration_ms, avg_confidence, object_type, fps):
        return None

    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
    max_speed = max(speed_samples, default=0.0)
    primary_score = (
        (distance_px * 0.03)
        + (duration_ms / 500.0)
        + (avg_confidence * 10.0)
        + (len(smoothed_points) * 0.5)
    )

    return {
        "tracking_id": tracking_id,
        "object_type": object_type,
        "class_id": class_id,
        "frame_start": smoothed_points[0]["frame_id"],
        "frame_end": smoothed_points[-1]["frame_id"],
        "duration_frames": smoothed_points[-1]["frame_id"] - smoothed_points[0]["frame_id"] + 1,
        "duration_ms": round(max(duration_ms, 0.0), 2),
        "time_in_frame_ms": round(max(duration_ms, 0.0), 2),
        "detection_count": len(smoothed_points),
        "avg_confidence": round(avg_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "distance_px": round(max(distance_px, 0.0), 2),
        "avg_speed_px_s": round(max(avg_speed, 0.0), 2),
        "max_speed_px_s": round(max(max_speed, 0.0), 2),
        "primary_score": round(primary_score, 2),
        "is_primary": False,
        "path": [
            {
                "frame_id": point["frame_id"],
                "timestamp_ms": point["timestamp_ms"],
                "x": point["x"],
                "y": point["y"],
                "confidence": point["confidence"],
                "scale": point["scale"],
            }
            for point in smoothed_points
        ],
    }


def _normalize_track_payload(track: Any, fps: float) -> dict[str, Any] | None:
    if not isinstance(track, dict):
        return None

    object_type = str(track.get("object_type", "")).strip().lower()
    if not object_type:
        return None

    try:
        tracking_id = int(track["tracking_id"])
    except (KeyError, TypeError, ValueError):
        return None

    path = _normalize_track_path(track.get("path"), fps)
    if not path:
        return None

    class_id = _coerce_int(track.get("class_id"), default=-1)
    avg_confidence = (
        sum(point.get("confidence", 0.0) for point in path) / len(path)
        if path
        else 0.0
    )
    max_confidence = max((point.get("confidence", 0.0) for point in path), default=0.0)
    smoothed_path = _smooth_track_points(path, object_type)
    distance_px, speed_samples = _compute_track_motion(smoothed_path, object_type, fps)
    duration_ms = _compute_duration_ms(smoothed_path, fps)
    if not _is_track_qualified(smoothed_path, distance_px, duration_ms, avg_confidence, object_type, fps):
        return None

    avg_confidence = (
        sum(point.get("confidence", 0.0) for point in smoothed_path) / len(smoothed_path)
        if smoothed_path
        else 0.0
    )
    max_confidence = max((point.get("confidence", 0.0) for point in smoothed_path), default=0.0)
    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
    max_speed = max(speed_samples, default=0.0)
    primary_score = _coerce_float(track.get("primary_score"), default=0.0)
    if primary_score <= 0:
        primary_score = (
            (distance_px * 0.03)
            + (duration_ms / 500.0)
            + (avg_confidence * 10.0)
            + (len(path) * 0.5)
        )

    return {
        "tracking_id": tracking_id,
        "object_type": object_type,
        "class_id": class_id,
        "frame_start": smoothed_path[0]["frame_id"],
        "frame_end": smoothed_path[-1]["frame_id"],
        "duration_frames": smoothed_path[-1]["frame_id"] - smoothed_path[0]["frame_id"] + 1,
        "duration_ms": round(max(duration_ms, 0.0), 2),
        "time_in_frame_ms": round(max(duration_ms, 0.0), 2),
        "detection_count": len(smoothed_path),
        "avg_confidence": round(avg_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "distance_px": round(max(distance_px, 0.0), 2),
        "avg_speed_px_s": round(max(avg_speed, 0.0), 2),
        "max_speed_px_s": round(max(max_speed, 0.0), 2),
        "primary_score": round(primary_score, 2),
        "is_primary": bool(track.get("is_primary", False)),
        "path": smoothed_path,
    }


def _select_best_segment(
    ordered_points: list[dict[str, Any]],
    object_type: str,
    fps: float,
) -> list[dict[str, Any]]:
    if len(ordered_points) <= 1:
        return ordered_points

    segments: list[list[dict[str, Any]]] = []
    current_segment = [ordered_points[0]]

    for point in ordered_points[1:]:
        previous = current_segment[-1]
        if _is_plausible_segment(previous, point, object_type, fps):
            current_segment.append(point)
            continue

        segments.append(current_segment)
        current_segment = [point]

    segments.append(current_segment)
    return max(
        segments,
        key=lambda segment: (
            len(segment),
            _compute_duration_ms(segment, fps),
            sum(point.get("confidence", 0.0) for point in segment),
        ),
    )


def _is_plausible_segment(
    previous: dict[str, Any],
    current: dict[str, Any],
    object_type: str,
    fps: float,
) -> bool:
    frame_delta = current["frame_id"] - previous["frame_id"]
    if frame_delta <= 0 or frame_delta > settings.max_track_gap_frames:
        return False

    seconds = _resolve_segment_seconds(previous, current, fps)
    if seconds <= 0:
        return False

    distance = hypot(current["x"] - previous["x"], current["y"] - previous["y"])
    step_multiplier = settings.small_object_tracker_step_multiplier if object_type in settings.small_object_class_name_list else 1.0
    max_step = max(
        settings.tracker_step_px * step_multiplier * frame_delta,
        max(previous.get("scale", 0.0), current.get("scale", 0.0)) * (3.3 if step_multiplier > 1.0 else 3.0),
    )
    if distance > max_step:
        return False

    speed = distance / seconds
    speed_limit = (
        settings.small_object_max_segment_speed_px_s
        if object_type in settings.small_object_class_name_list
        else settings.max_segment_speed_px_s
    )
    return speed <= speed_limit


def _compute_track_motion(
    ordered_points: list[dict[str, Any]],
    object_type: str,
    fps: float,
) -> tuple[float, list[float]]:
    distance_px = 0.0
    speed_samples: list[float] = []

    for previous, current in zip(ordered_points, ordered_points[1:]):
        if not _is_plausible_segment(previous, current, object_type, fps):
            continue

        distance = hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        seconds = _resolve_segment_seconds(previous, current, fps)
        distance_px += distance
        if seconds > 0:
            speed_samples.append(distance / seconds)

    return distance_px, speed_samples


def _compute_duration_ms(path: list[dict[str, Any]], fps: float) -> float:
    if len(path) <= 1:
        if not path:
            return 0.0
        return round(1000.0 / max(fps, 1.0), 2)

    timestamp_delta_ms = path[-1]["timestamp_ms"] - path[0]["timestamp_ms"]
    if timestamp_delta_ms > 0:
        return float(timestamp_delta_ms)

    frame_delta = path[-1]["frame_id"] - path[0]["frame_id"]
    return round((frame_delta / max(fps, 1.0)) * 1000.0, 2)


def _resolve_segment_seconds(
    previous: dict[str, Any],
    current: dict[str, Any],
    fps: float,
) -> float:
    timestamp_delta_ms = current["timestamp_ms"] - previous["timestamp_ms"]
    if timestamp_delta_ms > 0:
        return timestamp_delta_ms / 1000.0

    frame_delta = current["frame_id"] - previous["frame_id"]
    if frame_delta <= 0 or fps <= 0:
        return 0.0
    return frame_delta / fps


def _is_track_qualified(
    ordered_points: list[dict[str, Any]],
    distance_px: float,
    duration_ms: float,
    avg_confidence: float,
    object_type: str,
    fps: float,
) -> bool:
    if len(ordered_points) < settings.min_track_points:
        return False

    minimum_avg_confidence = (
        settings.min_small_object_track_avg_confidence
        if object_type in settings.small_object_class_name_list
        else settings.min_track_avg_confidence
    )
    if avg_confidence < minimum_avg_confidence:
        return False

    if distance_px >= settings.min_track_distance_px:
        return True

    minimum_static_duration_ms = max(
        int((settings.min_track_points / max(fps, 1.0)) * 1000.0 * 2.0),
        750,
    )
    return duration_ms >= minimum_static_duration_ms


def _smooth_track_points(
    ordered_points: list[dict[str, Any]],
    object_type: str,
) -> list[dict[str, Any]]:
    if len(ordered_points) <= 2:
        return ordered_points

    smoothing_factor = (
        settings.small_object_trajectory_smoothing_factor
        if object_type in settings.small_object_class_name_list
        else settings.trajectory_smoothing_factor
    )
    smoothing_factor = min(max(smoothing_factor, 0.0), 1.0)
    if smoothing_factor <= 0.0:
        return ordered_points

    smoothed_points = [dict(point) for point in ordered_points]
    for index in range(1, len(ordered_points) - 1):
        previous_point = ordered_points[index - 1]
        current_point = ordered_points[index]
        next_point = ordered_points[index + 1]
        local_average_x = (previous_point["x"] + current_point["x"] + next_point["x"]) / 3.0
        local_average_y = (previous_point["y"] + current_point["y"] + next_point["y"]) / 3.0
        smoothed_points[index]["x"] = round(
            (current_point["x"] * (1.0 - smoothing_factor)) + (local_average_x * smoothing_factor),
            2,
        )
        smoothed_points[index]["y"] = round(
            (current_point["y"] * (1.0 - smoothing_factor)) + (local_average_y * smoothing_factor),
            2,
        )
    return smoothed_points


def _stabilize_track_path_timestamps(
    ordered_points: list[dict[str, Any]],
    fps: float,
) -> list[dict[str, Any]]:
    if not ordered_points:
        return []

    stabilized_points: list[dict[str, Any]] = []
    for index, point in enumerate(ordered_points):
        stabilized_point = dict(point)
        fallback_timestamp = (
            int(round((point["frame_id"] / max(fps, 1.0)) * 1000.0))
            if fps > 0
            else max(_coerce_int(point.get("timestamp_ms"), default=0), 0)
        )
        timestamp_ms = max(_coerce_int(point.get("timestamp_ms"), default=fallback_timestamp), 0)

        if index == 0:
            stabilized_point["timestamp_ms"] = timestamp_ms if timestamp_ms > 0 else fallback_timestamp
            stabilized_points.append(stabilized_point)
            continue

        previous_point = stabilized_points[-1]
        frame_delta = max(point["frame_id"] - previous_point["frame_id"], 1)
        minimum_step_ms = (
            max(int(round((frame_delta / max(fps, 1.0)) * 1000.0)), 1)
            if fps > 0
            else frame_delta
        )
        stabilized_point["timestamp_ms"] = max(
            timestamp_ms,
            fallback_timestamp,
            previous_point["timestamp_ms"] + minimum_step_ms,
        )
        stabilized_points.append(stabilized_point)

    return stabilized_points


def _normalize_detection(detection: Any) -> dict[str, Any] | None:
    if not isinstance(detection, dict):
        return None

    try:
        frame_id = int(detection["frame_id"])
        timestamp_ms = max(int(detection["timestamp_ms"]), 0)
        object_type = str(detection["object_type"]).strip().lower()
        bbox = [float(value) for value in detection["bbox"]]
        confidence = float(detection.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    if not object_type or frame_id < 0 or len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        return None

    x1, y1, x2, y2 = bbox
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left <= 0 or bottom - top <= 0:
        return None

    tracking_id_raw = detection.get("tracking_id")
    try:
        tracking_id = None if tracking_id_raw in (None, "") else int(tracking_id_raw)
    except (TypeError, ValueError):
        tracking_id = None

    class_id = _coerce_int(detection.get("class_id"), default=-1)
    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "object_type": object_type,
        "class_id": class_id,
        "bbox": [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)],
        "confidence": min(max(confidence, 0.0), 1.0),
        "tracking_id": tracking_id if tracking_id is None or tracking_id >= 0 else None,
    }


def _normalize_track_path(path_payload: Any, fps: float) -> list[dict[str, Any]]:
    if not isinstance(path_payload, list):
        return []

    deduped_by_frame: dict[int, dict[str, Any]] = {}
    for point in path_payload:
        if not isinstance(point, dict):
            continue
        x = _coerce_float(point.get("x"), default=float("nan"))
        y = _coerce_float(point.get("y"), default=float("nan"))
        if not isfinite(x) or not isfinite(y):
            continue

        frame_id = _coerce_int(point.get("frame_id"), default=-1)
        if frame_id < 0:
            continue

        deduped_by_frame[frame_id] = {
            "frame_id": frame_id,
            "timestamp_ms": max(_coerce_int(point.get("timestamp_ms"), default=0), 0),
            "x": round(x, 2),
            "y": round(y, 2),
            "confidence": min(max(_coerce_float(point.get("confidence"), default=0.0), 0.0), 1.0),
            "scale": max(round(_coerce_float(point.get("scale"), default=0.0), 2), 0.0),
        }

    ordered_points = [deduped_by_frame[frame_id] for frame_id in sorted(deduped_by_frame)]
    return _stabilize_track_path_timestamps(ordered_points, fps)


def _normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_events: list[dict[str, Any]] = []
    seen_event_keys: set[tuple[str, int, int, str, tuple[int, ...]]] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "")).strip().lower()
        object_type = str(event.get("object_type", "")).strip().lower() or "unknown"
        if not event_type:
            continue

        start_frame = max(_coerce_int(event.get("start_frame"), default=0), 0)
        end_frame = max(_coerce_int(event.get("end_frame"), default=start_frame), start_frame)
        tracking_ids_payload = event.get("tracking_ids", [])
        tracking_ids: list[int] = []
        if isinstance(tracking_ids_payload, list):
            tracking_ids = [
                tracking_id
                for tracking_id in (_coerce_int(value, default=-1) for value in tracking_ids_payload)
                if tracking_id >= 0
            ]
        normalized_event = {
            "event_type": event_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": max(end_frame - start_frame + 1, 1),
            "object_type": object_type,
            "tracking_ids": tracking_ids,
            "details": _normalize_event_details(event.get("details")),
        }
        event_key = (
            normalized_event["event_type"],
            normalized_event["start_frame"],
            normalized_event["end_frame"],
            normalized_event["object_type"],
            tuple(normalized_event["tracking_ids"]),
        )
        if event_key in seen_event_keys:
            continue
        seen_event_keys.add(event_key)
        normalized_events.append(normalized_event)

    normalized_events.sort(key=lambda item: (item["start_frame"], item["event_type"], item["object_type"]))
    return normalized_events


def _normalize_event_details(details: Any) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}

    normalized_details: dict[str, Any] = {}
    for key, value in details.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        if isinstance(value, bool):
            normalized_details[normalized_key] = value
            continue
        if isinstance(value, int):
            normalized_details[normalized_key] = value
            continue
        if isinstance(value, float):
            if isfinite(value):
                normalized_details[normalized_key] = round(value, 4)
            continue
        if isinstance(value, str):
            normalized_details[normalized_key] = value.strip()
            continue
        if isinstance(value, list):
            normalized_values: list[Any] = []
            for item in value:
                if isinstance(item, (bool, int)):
                    normalized_values.append(item)
                elif isinstance(item, float) and isfinite(item):
                    normalized_values.append(round(item, 4))
                elif isinstance(item, str):
                    normalized_values.append(item.strip())
            if normalized_values:
                normalized_details[normalized_key] = normalized_values

    return normalized_details


def _resolve_track_class_id(ordered_points: list[dict[str, Any]]) -> int:
    class_ids = [point.get("class_id", -1) for point in ordered_points]
    counter = Counter(class_id for class_id in class_ids if isinstance(class_id, int) and class_id >= 0)
    if not counter:
        return -1
    return counter.most_common(1)[0][0]


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(parsed):
        return default
    return parsed
