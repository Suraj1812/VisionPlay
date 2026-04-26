from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai.cricket.insight_builder import build_cricket_package


class CricketInsightBuilderTests(unittest.TestCase):
    def test_build_cricket_package_emits_specialized_contract_for_action_cam(self) -> None:
        package = build_cricket_package(
            heuristic_analytics={
                "scorecard": {"score": "14/0", "overs": "2.1", "total_balls": 13, "run_rate": 6.46},
                "timeline": [
                    {
                        "ball": 13,
                        "over": "2.1",
                        "runs": 4,
                        "boundary": True,
                        "shot": "cover_drive",
                        "length": "full",
                        "line": "off_stump",
                        "ts_start": 5400,
                        "ts_end": 6900,
                        "commentary": "Driven through cover for four.",
                        "confidence": 0.82,
                    }
                ],
            },
            score_timeline=[],
            team_summary={
                "camera_profile": "action-cam-end-on",
                "calibrated": True,
                "teams": {
                    "team_0": {"players": [{"id": 10, "role": "striker", "team_side": "batting", "confidence": 0.8}]},
                    "team_1": {"players": [{"id": 20, "role": "bowler", "team_side": "fielding", "confidence": 0.84}]},
                },
                "roles": {
                    10: {"role": "striker", "team_side": "batting", "confidence": 0.8},
                    20: {"role": "bowler", "team_side": "fielding", "confidence": 0.84},
                },
                "batting_team_id": 0,
                "fielding_team_id": 1,
                "role_method": "end-on action-cam heuristics",
            },
            delivery_summary={"total_deliveries": 13, "estimated_overs": "2.1"},
            visual_events=[
                {"event_type": "ball_released", "timestamp_ms": 5400, "confidence": 0.71, "details": {}},
                {"event_type": "ball_bounced", "timestamp_ms": 5900, "confidence": 0.77, "details": {"bounce_x": 980, "bounce_y": 640, "length": "full", "line": "off_stump"}},
                {"event_type": "bat_impact", "timestamp_ms": 6200, "confidence": 0.81, "details": {"shot_type": "cover_drive", "power": "medium"}},
                {"event_type": "four", "timestamp_ms": 6900, "confidence": 0.8, "details": {"runs": 4}},
            ],
            last_score=SimpleNamespace(batting_team="Team A"),
            camera_cuts=0,
            ball_trajectory=[(960.0, 520.0), (972.0, 580.0), (980.0, 640.0), (1030.0, 710.0)],
            transcript={
                "status": "available",
                "source": "faster-whisper",
                "confidence": 0.62,
                "speech_present": True,
                "segments": [
                    {"start_ms": 5600, "end_ms": 7000, "text": "Beautiful shot through the covers", "confidence": 0.66}
                ],
            },
            profile_report={
                "profile": "cricket_end_on_action_cam_v1",
                "specialized": True,
                "confidence": 0.83,
                "overlay_present": False,
                "reasons": ["camera remains stable from the bowler end"],
            },
            fps=60.0,
            frame_width=1920,
            frame_height=1080,
        )

        self.assertEqual(package["profile"], "cricket_end_on_action_cam_v1")
        self.assertEqual(package["mode"], "specialized")
        self.assertEqual(package["capabilities"]["feed_source"], "delivery-timeline")
        self.assertEqual(package["speech"]["mode"], "speech-led")
        self.assertTrue(package["subtitles"]["cues"])
        self.assertEqual(package["deliveries"][0]["result"], "four")
        self.assertTrue(package["ball_path"]["anchors"]["bounce"])
        self.assertIn("striker", package["roles"][10]["role"])
        self.assertIn("No scoreboard overlay was detected", package["warnings"][0])

    def test_build_cricket_package_uses_event_led_subtitles_when_speech_is_missing(self) -> None:
        package = build_cricket_package(
            heuristic_analytics={},
            score_timeline=[],
            team_summary={},
            delivery_summary={},
            visual_events=[
                {"event_type": "ball_released", "timestamp_ms": 1000, "confidence": 0.7, "details": {}},
                {"event_type": "bat_impact", "timestamp_ms": 1600, "confidence": 0.8, "details": {"shot_type": "defensive"}},
                {"event_type": "dot_ball", "timestamp_ms": 2600, "confidence": 0.72, "details": {}},
            ],
            last_score=SimpleNamespace(batting_team=""),
            camera_cuts=0,
            ball_trajectory=[],
            transcript={"status": "empty", "confidence": 0.0, "speech_present": False, "segments": []},
            profile_report={"profile": "generic", "specialized": False, "confidence": 0.12, "overlay_present": False},
        )

        self.assertEqual(package["mode"], "fallback")
        self.assertEqual(package["subtitles"]["mode"], "event-led")
        self.assertTrue(package["subtitles"]["cues"])
        self.assertEqual(package["timeline"][0]["source"], "delivery-timeline")
        self.assertIn("event-led", package["warnings"][1].lower())

    def test_build_cricket_package_uses_scoreboard_only_when_overlay_is_present(self) -> None:
        package = build_cricket_package(
            heuristic_analytics={"scorecard": {"score": "9/0", "overs": "1.5"}},
            score_timeline=[
                {
                    "score": "10/0",
                    "runs": 10,
                    "wickets": 0,
                    "overs": "2.0",
                    "overs_float": 2.0,
                    "run_rate": 5.0,
                    "batting_team": "IND",
                    "confidence": 0.74,
                    "frame_id": 120,
                    "timestamp_ms": 4800,
                },
            ],
            team_summary={},
            delivery_summary={},
            visual_events=[],
            last_score=SimpleNamespace(batting_team="IND"),
            camera_cuts=0,
            ball_trajectory=[],
            transcript={},
            profile_report={"profile": "generic", "specialized": False, "confidence": 0.2, "overlay_present": True},
        )

        self.assertEqual(package["scorecard"]["source"], "scoreboard")
        self.assertEqual(package["timeline"][0]["source"], "scoreboard")
        self.assertEqual(package["capabilities"]["score_source"], "scoreboard")


if __name__ == "__main__":
    unittest.main()
