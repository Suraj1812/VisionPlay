
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    logger.info("EasyOCR not installed; scoreboard extraction disabled")

@dataclass
class ScoreboardData:
    raw_text: str = ""
    score: str = ""
    runs: int = -1
    wickets: int = -1
    overs: str = ""
    overs_float: float = -1.0
    run_rate: float = -1.0
    required_run_rate: float = -1.0
    target: int = -1
    batting_team: str = ""
    bowling_team: str = ""
    confidence: float = 0.0
    detected: bool = False
    frame_id: int = 0
    timestamp_ms: int = 0

_SCORE_RE = [
    re.compile(r'([A-Z]{2,5})\s+(\d{1,3})\s*[/\-]\s*(\d{1,2})'),
    re.compile(r'(\d{1,3})\s*[/\-]\s*(\d{1,2})'),
]
_OVER_RE = [
    re.compile(r'[\(]?\s*(\d{1,3}\.\d)\s*(?:ov(?:ers?)?)?\s*[\)]?'),
    re.compile(r'(?:Ov|Over|OV)\s*(\d{1,3}\.\d)'),
]
_RR_RE = re.compile(r'(?:RR|CRR|R/R)\s*[:\s]*(\d{1,2}\.\d{1,2})')
_RRR_RE = re.compile(r'(?:RRR|Req|REQ)\s*[:\s]*(\d{1,2}\.\d{1,2})')
_TGT_RE = re.compile(r'(?:Target|TGT|Need)\s*[:\s]*(\d{1,3})')

class ScoreboardOCR:

    def __init__(self, ocr_interval: int = 18):
        self._reader = None
        self._last = ScoreboardData()
        self._history: list[ScoreboardData] = []
        self._frame_count = 0
        self._ocr_interval = ocr_interval
        self._miss_count = 0
        self._cooldown_calls = 0
        self._overlay_scan_count = 0
        self._overlay_hit_count = 0

    def _get_reader(self):
        if self._reader is None and _HAS_OCR:
            try:
                self._reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            except Exception:
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def extract(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        fps: float = 25.0,
        force: bool = False,
    ) -> ScoreboardData:
        self._frame_count += 1
        timestamp_ms = self._frame_timestamp_ms(frame_id, fps)
        self._overlay_scan_count += 1
        overlay_present = self._overlay_present(frame)
        self._overlay_hit_count += int(overlay_present)
        if not overlay_present and not force:
            return self._last

        if self._cooldown_calls > 0:
            self._cooldown_calls -= 1
            return self._last

        if (self._frame_count - 1) % self._ocr_interval != 0:
            return self._last

        reader = self._get_reader()
        if reader is None:
            return ScoreboardData(frame_id=frame_id, timestamp_ms=timestamp_ms)

        h, w = frame.shape[:2]

        rois = self._scoreboard_rois(frame, h, w)

        texts = []
        total_conf = 0.0
        n = 0

        for roi in rois:
            if roi.size == 0 or roi.shape[0] < 10:
                continue
            try:
                for candidate in self._preprocess_roi_variants(roi):
                    results = reader.readtext(candidate, detail=1, paragraph=False)
                    for _, text, conf in results:
                        if conf > 0.35:
                            texts.append(text)
                            total_conf += conf
                            n += 1
            except Exception:
                continue

        if not texts:
            self._register_miss()
            return self._last

        raw = " ".join(texts)
        data = self._parse(raw)
        data.raw_text = raw
        data.confidence = total_conf / max(n, 1)
        data.detected = True
        data.frame_id = max(int(frame_id), 0)
        data.timestamp_ms = timestamp_ms

        if self._validate(data):
            self._last = data
            self._history.append(data)
            self._miss_count = 0
            self._cooldown_calls = 0
        else:
            self._register_miss()

        return self._last

    def _parse(self, text: str) -> ScoreboardData:
        d = ScoreboardData()
        for p in _SCORE_RE:
            m = p.search(text)
            if m:
                g = m.groups()
                if len(g) == 3:
                    d.batting_team, d.runs, d.wickets = g[0], int(g[1]), int(g[2])
                else:
                    d.runs, d.wickets = int(g[0]), int(g[1])
                d.score = f"{d.runs}/{d.wickets}"
                break
        for p in _OVER_RE:
            m = p.search(text)
            if m:
                d.overs = m.group(1)
                d.overs_float = float(d.overs)
                break
        if d.runs >= 0 and d.overs_float > 0:
            completed = int(d.overs_float) + (d.overs_float % 1) * 10 / 6
            d.run_rate = round(d.runs / max(completed, 0.1), 2) if completed > 0 else 0.0
        m = _RR_RE.search(text)
        if m:
            d.run_rate = float(m.group(1))
        m = _RRR_RE.search(text)
        if m:
            d.required_run_rate = float(m.group(1))
        m = _TGT_RE.search(text)
        if m:
            d.target = int(m.group(1))
        return d

    def _validate(self, new: ScoreboardData) -> bool:
        old = self._last
        if not old.detected:
            return True
        if new.runs < 0:
            return False
        if new.score == old.score and new.overs == old.overs:
            return False
        old_balls = self._overs_to_balls(old.overs_float)
        new_balls = self._overs_to_balls(new.overs_float)
        if new_balls < old_balls:
            if old.runs > 50 and new.runs < 10:
                return True  # New innings
            return False
        if new_balls == old_balls and new.runs != old.runs:
            return False
        if new_balls - old_balls > 12:
            return False
        if old.runs >= 0 and new.runs - old.runs > 12:
            return False
        if old.wickets >= 0 and new.wickets - old.wickets > 2:
            return False
        if new.runs >= old.runs:
            return True
        if new.wickets > old.wickets:
            return True
        if old.runs > 50 and new.runs < 10:
            return True  # New innings
        return False

    def get_score_timeline(self) -> list[dict]:
        return [{
            "score": s.score, "runs": s.runs, "wickets": s.wickets,
            "overs": s.overs, "overs_float": s.overs_float, "run_rate": s.run_rate,
            "batting_team": s.batting_team,
            "confidence": s.confidence,
            "frame_id": s.frame_id,
            "timestamp_ms": s.timestamp_ms,
        } for s in self._history]

    @property
    def last_score(self) -> ScoreboardData:
        return self._last

    def reset(self):
        self._last = ScoreboardData()
        self._history.clear()
        self._frame_count = 0
        self._miss_count = 0
        self._cooldown_calls = 0
        self._overlay_scan_count = 0
        self._overlay_hit_count = 0

    def _register_miss(self) -> None:
        if self._last.detected:
            return
        self._miss_count += 1
        if self._miss_count >= 4:
            self._cooldown_calls = max(self._ocr_interval * 10, 30)
            self._miss_count = 0

    @staticmethod
    def _frame_timestamp_ms(frame_id: int, fps: float) -> int:
        safe_frame_id = max(int(frame_id), 0)
        safe_fps = fps if fps and fps > 0 else 25.0
        return int((safe_frame_id / safe_fps) * 1000)

    @staticmethod
    def _overs_to_balls(overs_float: float) -> int:
        whole = int(max(overs_float, 0))
        partial = int(round((overs_float - whole) * 10))
        return whole * 6 + max(partial, 0)

    @staticmethod
    def _scoreboard_rois(frame: np.ndarray, height: int, width: int) -> list[np.ndarray]:
        return [
            frame[0:int(height * 0.14), 0:int(width * 0.58)],
            frame[0:int(height * 0.12), int(width * 0.22):int(width * 0.82)],
            frame[int(height * 0.84):height, 0:int(width * 0.55)],
            frame[int(height * 0.84):height, int(width * 0.45):width],
        ]

    @staticmethod
    def _preprocess_roi_variants(roi: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants = [upscaled]

        normalized = cv2.normalize(upscaled, None, 0, 255, cv2.NORM_MINMAX)
        variants.append(normalized)

        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(binary)

        adaptive = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append(adaptive)
        return variants

    @property
    def overlay_detected_ratio(self) -> float:
        if self._overlay_scan_count <= 0:
            return 0.0
        return self._overlay_hit_count / self._overlay_scan_count

    @staticmethod
    def _overlay_present(frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        regions = [
            frame[0:int(height * 0.12), 0:width],
            frame[int(height * 0.84):height, 0:width],
        ]
        matches = 0
        for region in regions:
            if region.size == 0:
                continue
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float((edges > 0).mean())
            std_dev = float(gray.std())
            if 0.016 <= edge_density <= 0.14 and std_dev <= 62.0:
                matches += 1
        return matches >= 1
