from __future__ import annotations

import time
from collections import deque


class PerformanceMeter:
    def __init__(self, window=120):
        self.times = deque(maxlen=max(10, int(window)))
        self.last = None
        self.last_process_ms = 0.0

    def begin(self):
        return time.perf_counter()

    def end(self, started):
        now = time.perf_counter()
        self.last_process_ms = (now - started) * 1000.0

        if self.last is not None:
            delta = now - self.last
            if delta > 0:
                self.times.append(delta)
        self.last = now

    @property
    def fps(self):
        if not self.times:
            return 0.0
        average = sum(self.times) / len(self.times)
        return 1.0 / average if average > 0 else 0.0
