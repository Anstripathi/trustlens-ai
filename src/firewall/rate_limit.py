"""Optional bounded in-memory request limiter; keys should be session IDs, not user data."""
from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic


class RateLimiter:
    def __init__(self, *, max_keys: int = 10_000):
        if isinstance(max_keys, bool) or not isinstance(max_keys, int) or not 1 <= max_keys <= 100_000:
            raise ValueError("max_keys must be between 1 and 100,000.")
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, *, limit: int, window_seconds: int, now: float | None = None) -> bool:
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("Rate-limit key must be a non-empty opaque identifier of at most 128 characters.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("Rate-limit request count is invalid.")
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, int) or not 1 <= window_seconds <= 86_400:
            raise ValueError("Rate-limit window is invalid.")
        timestamp = monotonic() if now is None else float(now)
        with self._lock:
            queue = self._events.get(key)
            if queue is None:
                if len(self._events) >= self._max_keys:
                    oldest = next(iter(self._events))
                    self._events.pop(oldest, None)
                queue = self._events[key]
            while queue and timestamp - queue[0] >= window_seconds:
                queue.popleft()
            if len(queue) >= limit:
                return False
            queue.append(timestamp)
            return True
