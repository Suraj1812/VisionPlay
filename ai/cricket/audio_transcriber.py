from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging

from backend.utils.config import settings


logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - exercised by unit tests and environments without the dependency
    WhisperModel = None


@dataclass
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class TranscriptResult:
    status: str = "unavailable"
    source: str = "none"
    language: str = ""
    confidence: float = 0.0
    speech_present: bool = False
    segments: list[TranscriptSegment] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "speech_present": self.speech_present,
            "segments": [segment.to_dict() for segment in self.segments],
            "reason": self.reason,
        }


class CricketAudioTranscriber:
    def __init__(self) -> None:
        self._model: Any = None

    def transcribe(self, video_path: str, profile: str = "generic") -> TranscriptResult:
        if not settings.audio_transcription_enabled:
            return TranscriptResult(reason="Audio transcription is disabled.")

        path = Path(video_path)
        if not path.exists():
            return TranscriptResult(reason="Video file is unavailable for transcription.")

        try:
            model = self._get_model()
            if model is None:
                return TranscriptResult(reason="faster-whisper is not installed.")
        except Exception as exc:
            logger.exception("Unable to initialize audio transcription")
            return TranscriptResult(reason=str(exc))

        try:
            segments, info = model.transcribe(
                str(path),
                language=settings.audio_transcription_language or None,
                vad_filter=True,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                word_timestamps=False,
                condition_on_previous_text=False,
                vad_parameters={"min_silence_duration_ms": 450},
            )
        except Exception:
            logger.exception("Audio transcription failed for %s", video_path)
            return TranscriptResult(reason="Audio transcription failed.")

        transcript_segments: list[TranscriptSegment] = []
        confidences: list[float] = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(getattr(segment, "avg_logprob", -2.0) or -2.0)
            confidence = max(min((1.0 - no_speech_prob) * 0.65 + max(avg_logprob + 2.0, 0.0) * 0.2, 1.0), 0.0)
            transcript_segments.append(
                TranscriptSegment(
                    start_ms=max(int(float(segment.start) * 1000), 0),
                    end_ms=max(int(float(segment.end) * 1000), 0),
                    text=text,
                    confidence=confidence,
                )
            )
            confidences.append(confidence)

        speech_present = bool(transcript_segments)
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        status = "available" if speech_present else "empty"
        if profile == "cricket_end_on_action_cam_v1" and average_confidence < settings.audio_transcription_min_confidence:
            status = "low_confidence"

        return TranscriptResult(
            status=status,
            source="faster-whisper",
            language=str(getattr(info, "language", "") or ""),
            confidence=average_confidence,
            speech_present=speech_present and average_confidence >= settings.audio_transcription_min_confidence,
            segments=transcript_segments,
            reason="" if speech_present else "No confident speech segments were detected.",
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if WhisperModel is None:
            return None
        model_dir = settings.model_cache_path / "whisper"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            settings.audio_transcription_model_size,
            device="cpu",
            compute_type=settings.audio_transcription_compute_type,
            download_root=str(model_dir),
            cpu_threads=settings.audio_transcription_cpu_threads,
        )
        return self._model
