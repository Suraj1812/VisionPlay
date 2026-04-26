from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.cricket.audio_transcriber import CricketAudioTranscriber


class CricketAudioTranscriberTests(unittest.TestCase):
    def test_transcribe_reports_missing_dependency_cleanly(self) -> None:
        transcriber = CricketAudioTranscriber()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            with patch("ai.cricket.audio_transcriber.WhisperModel", None):
                result = transcriber.transcribe(handle.name, profile="cricket_end_on_action_cam_v1")

        self.assertEqual(result.status, "unavailable")
        self.assertIn("faster-whisper", result.reason)

    def test_transcribe_reports_missing_file(self) -> None:
        transcriber = CricketAudioTranscriber()
        result = transcriber.transcribe(str(Path("/tmp/missing-visionplay-audio.mp4")))
        self.assertEqual(result.status, "unavailable")
        self.assertIn("unavailable", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
