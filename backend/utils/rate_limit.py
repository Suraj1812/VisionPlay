from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
import time
from typing import Deque

class SlidingWindowRateLimiter:
    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = max(int(window_seconds), 1)
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0

        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            self._prune(events, now)

            if len(events) >= limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                return False, retry_after

            events.append(now)
            return True, 0

    def _prune(self, events: Deque[float], now: float) -> None:
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
