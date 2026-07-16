from __future__ import annotations

import time
from pathlib import Path

from .acquisition import AcquisitionEngine
from .calibration import CalibrationController
from .config import ConfigStore
from .needle import NeedleDetector
from .overlay import draw_runtime_overlay
from .performance import PerformanceMeter
from .shared import RuntimeSnapshot
from .tracker import SkillCheckTracker
from .ui import UIProcess
from .zones import ZoneDetector


class DbdRuntime:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

        self.config = ConfigStore()
        self.performance = PerformanceMeter()
        self.calibration = CalibrationController(self.config)
        self.acquisition = AcquisitionEngine()
        self.tracker = SkillCheckTracker()
        self.needle = NeedleDetector()
        self.zones = ZoneDetector()

        script_dir = Path(__file__).resolve().parent.parent
        self.ui = UIProcess(script_dir)

        self.snapshot = RuntimeSnapshot(
            state="ACQUIRE",
            status="Searching",
            details="Fixed-radius template acquisition ready.",
            debug_candidates=[],
        )

        self.last_runtime_publish = 0.0
        self.last_acquisition = 0.0
        self.pending_circle = None
        self.pending_hits = 0

        self.config.update(
            runtime_status="Acquisition engine ready",
            runtime_details="Fixed-radius template matcher loaded",
            runtime_width=self.width,
            runtime_height=self.height,
            runtime_state="ACQUIRE",
        )

    @staticmethod
    def _circles_close(first, second):
        if first is None or second is None:
            return False
        radius = max(1.0, (first.radius + second.radius) * 0.5)
        return (
            abs(first.x - second.x) <= radius * 0.22
            and abs(first.y - second.y) <= radius * 0.22
            and abs(first.radius - second.radius) <= radius * 0.16
        )

    def _confirm_candidate(self, candidate, settings):
        required = max(
            1, int(settings.get("lock_confirmation_frames", 2))
        )
        if required <= 1:
            return candidate

        if self._circles_close(candidate, self.pending_circle):
            self.pending_hits += 1
        else:
            self.pending_circle = candidate
            self.pending_hits = 1

        if self.pending_hits >= required:
            confirmed = candidate
            self.pending_circle = None
            self.pending_hits = 0
            return confirmed
        return None

    def process(self, frame):
        started = self.performance.begin()
        settings = self.config.load()

        self.ui.ensure_running(settings.get("ui_enabled", True))
        frame, settings = self.calibration.process(frame, settings)

        if settings.get("calibration_active", False):
            self.snapshot.state = "CALIBRATION"
            self.snapshot.status = "Adjust radius on frozen Helios frame"
            self.snapshot.details = "Press P to save"
        elif not settings.get("enabled", True):
            self.tracker.reset()
            self.acquisition.reset()
            self.snapshot.state = "DISABLED"
            self.snapshot.status = "Detector disabled"
            self.snapshot.details = "Enable it from the Dashboard"
            self.snapshot.circle = None
        else:
            now = time.monotonic()

            if self.tracker.circle is None:
                self.snapshot.state = "ACQUIRE"
                acquisition_rate = max(
                    1, min(120, int(settings.get("acquisition_rate", 60)))
                )
                interval = 1.0 / acquisition_rate

                if now - self.last_acquisition >= interval:
                    self.last_acquisition = now
                    candidate = self.acquisition.search(frame, settings)
                    if candidate is not None:
                        confirmed = self._confirm_candidate(
                            candidate, settings
                        )
                        if confirmed is not None:
                            self.tracker.lock(confirmed)
                    else:
                        self.pending_circle = None
                        self.pending_hits = 0

                self.snapshot.circle = self.tracker.circle
                self.snapshot.status = (
                    "Circle locked"
                    if self.tracker.circle is not None
                    else "Searching fixed radius"
                )
                self.snapshot.details = (
                    f"search={self.acquisition.last_search_ms:.2f}ms | "
                    f"candidates={self.acquisition.last_candidate_count} | "
                    f"best={self.acquisition.last_best_score:.3f}"
                )
            else:
                self.snapshot.state = "TRACK"
                circle = self.tracker.update(frame, settings)
                self.snapshot.circle = circle
                if circle is None:
                    self.snapshot.status = "Lock lost"
                    self.snapshot.details = "Returning to acquisition"
                    self.snapshot.state = "ACQUIRE"
                else:
                    self.snapshot.status = "Tracking circle"
                    self.snapshot.details = (
                        f"track score={self.tracker.last_score:.3f} | "
                        f"lost={self.tracker.lost_frames}/"
                        f"{settings.get('tracker_lost_frames', 18)}"
                    )

        self.snapshot.candidate_count = self.acquisition.last_candidate_count
        self.snapshot.best_score = self.acquisition.last_best_score
        self.snapshot.debug_candidates = self.acquisition.last_candidates

        frame = draw_runtime_overlay(
            frame,
            settings,
            self.snapshot,
            self.performance.fps,
            self.performance.last_process_ms,
        )

        self.performance.end(started)

        now = time.monotonic()
        if now - self.last_runtime_publish >= 0.5:
            self.last_runtime_publish = now
            self.config.update(
                runtime_status=self.snapshot.status,
                runtime_details=self.snapshot.details,
                runtime_width=frame.shape[1],
                runtime_height=frame.shape[0],
                runtime_fps=self.performance.fps,
                runtime_ms=self.performance.last_process_ms,
                runtime_state=self.snapshot.state,
            )

        return frame, bytearray()
