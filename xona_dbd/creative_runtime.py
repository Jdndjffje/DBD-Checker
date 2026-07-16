from __future__ import annotations

import time
from pathlib import Path

from .acquisition import AcquisitionEngine
from .calibration import CalibrationController
from .config import ConfigStore
from .needle import NeedleDetector
from .overlay import draw_runtime_overlay
from .performance import PerformanceMeter
from .reference_library import ReferenceCaptureController
from .shared import RuntimeSnapshot
from .tracker import SkillCheckTracker
from .ui import UIProcess
from .zones import ZoneDetector


class CreativeRuntime:
    def __init__(self, width, height, script_dir):
        self.width = int(width)
        self.height = int(height)

        self.config = ConfigStore()
        self.performance = PerformanceMeter()
        self.calibration = CalibrationController(self.config)
        self.reference_capture = ReferenceCaptureController(self.config)
        self.acquisition = AcquisitionEngine(self.config)
        self.tracker = SkillCheckTracker()
        self.needle = NeedleDetector()
        self.zones = ZoneDetector()
        self.ui = UIProcess(Path(script_dir))

        self.snapshot = RuntimeSnapshot(
            state="ACQUIRE",
            status="Searching",
            details="Fixed-radius template acquisition ready.",
            debug_candidates=[],
        )

        self.last_runtime_publish = 0.0
        self.last_acquisition = 0.0
        self.acquisition_blocked_until = 0.0
        self.recovering_from_loss = False
        self.last_needle_analysis = 0.0
        self.last_zone_analysis = 0.0
        self.pending_circle = None
        self.pending_hits = 0

        self.press_frames_remaining = 0
        self.last_press_time = 0.0
        self.press_armed = True
        self.simulated_press_frames = 0

        self.config.update(
            runtime_status="Creative runtime ready",
            runtime_details="V7 runtime active; load only XonasDBDChecker.py in Creative",
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

    @staticmethod
    def _angle_distance(first, second):
        if first is None or second is None:
            return 999.0
        return abs((first - second + 180.0) % 360.0 - 180.0)

    def consume_press_request(self, settings):
        """
        Returns (button_index, value) or None.

        The Creative entrypoint applies this to Combo so runtime vision code
        remains isolated from controller output code.
        """
        if self.press_frames_remaining <= 0:
            return None

        self.press_frames_remaining -= 1
        return (
            int(settings.get("press_button_index", 15)),
            100,
        )

    def process(self, frame):
        started = self.performance.begin()
        settings = self.config.load()

        # UI is a separate normal Python process. It must never be loaded as a
        # Creative script.
        self.ui.ensure_running(settings.get("ui_enabled", True))

        frame, settings = self.calibration.process(frame, settings)
        settings = self.reference_capture.process(frame, settings)

        if not settings.get("enabled", True):
            self.tracker.reset()
            self.acquisition.reset()
            self.snapshot.state = "DISABLED"
            self.snapshot.status = "Detector disabled"
            self.snapshot.details = "Enable it from Dashboard"
            self.snapshot.circle = None
        else:
            now = time.monotonic()

            if self.tracker.circle is None:
                self.snapshot.state = "ACQUIRE"
                configured_rate = int(
                    settings.get("acquisition_rate", 60)
                )
                recovery_rate = int(
                    settings.get("recovery_acquisition_rate", 20)
                )
                acquisition_rate = max(
                    1,
                    min(
                        120,
                        recovery_rate
                        if self.recovering_from_loss
                        else configured_rate,
                    ),
                )
                interval = 1.0 / acquisition_rate

                if (
                    now >= self.acquisition_blocked_until
                    and now - self.last_acquisition >= interval
                ):
                    self.last_acquisition = now
                    candidate = self.acquisition.search(frame, settings)
                    if candidate is not None:
                        confirmed = self._confirm_candidate(candidate, settings)
                        if confirmed is not None:
                            self.tracker.lock(confirmed)
                            self.recovering_from_loss = False
                    else:
                        self.pending_circle = None
                        self.pending_hits = 0

                self.snapshot.circle = self.tracker.circle
                self.snapshot.status = (
                    "Circle locked"
                    if self.tracker.circle is not None
                    else "Searching fixed radius"
                )
                best = (
                    self.acquisition.last_candidates[0]
                    if self.acquisition.last_candidates
                    else None
                )
                if best is None:
                    self.snapshot.details = (
                        f"reference search={self.acquisition.last_search_ms:.2f}ms | "
                        f"references={settings.get('reference_positive_count', 0)} | "
                        f"candidates=0"
                    )
                else:
                    self.snapshot.details = (
                        f"reference search={self.acquisition.last_search_ms:.2f}ms | "
                        f"references={settings.get('reference_positive_count', 0)} | "
                        f"candidates={self.acquisition.last_candidate_count} | "
                        f"best={best.score:.3f} "
                        f"edge={best.edge_template_score:.3f} "
                        f"gray={best.white_template_score:.3f}"
                    )
            else:
                self.snapshot.state = "TRACK"
                self.acquisition.last_candidates = []
                self.acquisition.last_candidate_count = 0
                circle = self.tracker.update(frame, settings)
                self.snapshot.circle = circle
                if circle is None:
                    self.needle.reset()
                    self.zones.reset()
                    self.snapshot.geometry = None
                    self.press_armed = True
                    self.press_frames_remaining = 0

                    cooldown_seconds = (
                        float(
                            settings.get(
                                "lost_detection_cooldown_ms",
                                140,
                            )
                        )
                        / 1000.0
                    )
                    self.acquisition_blocked_until = (
                        now + cooldown_seconds
                    )
                    self.recovering_from_loss = True

                    self.snapshot.status = "Lock lost"
                    self.snapshot.details = (
                        "Cooling down before low-rate reacquisition"
                    )
                    self.snapshot.state = "ACQUIRE"
                else:
                    needle_rate = max(
                        1,
                        min(
                            120,
                            int(settings.get("needle_analysis_rate", 120)),
                        ),
                    )
                    zone_rate = max(
                        1,
                        min(
                            120,
                            int(settings.get("zone_analysis_rate", 30)),
                        ),
                    )

                    previous_geometry = self.snapshot.geometry
                    needle_angle = (
                        previous_geometry.needle_angle
                        if previous_geometry is not None
                        else None
                    )
                    zone_data = None

                    needle_due = (
                        now - self.last_needle_analysis >= 1.0 / needle_rate
                    )
                    zone_due = (
                        now - self.last_zone_analysis >= 1.0 / zone_rate
                    )

                    if needle_due:
                        self.last_needle_analysis = now
                        needle_angle = self.needle.detect(
                            frame,
                            circle,
                            settings,
                        )

                    if zone_due:
                        self.last_zone_analysis = now
                        zone_data = self.zones.detect(
                            frame,
                            circle,
                            settings,
                        )
                    elif self.zones.last_good_angle is not None:
                        zone_data = {
                            "good_angle": self.zones.last_good_angle,
                            "good_width": self.zones.last_good_width,
                            "great_angle": self.zones.last_great_angle,
                            "great_width": self.zones.last_great_width,
                            "target_type": self.zones.last_target_type,
                            "baseline": self.zones.last_baseline,
                            "confidence": self.zones.last_confidence,
                        }

                    if needle_due or zone_due:

                        good_angle = None
                        great_angle = None
                        target_type = "NONE"
                        target_center = None
                        target_width = 0.0

                        if zone_data is not None:
                            good_angle = zone_data.get("good_angle")
                            great_angle = zone_data.get("great_angle")
                            target_type = zone_data.get("target_type", "GOOD")

                            if target_type == "GREAT" and great_angle is not None:
                                target_center = great_angle
                                target_width = float(
                                    zone_data.get("great_width", 0.0)
                                )
                            else:
                                target_center = good_angle
                                target_width = float(
                                    zone_data.get("good_width", 0.0)
                                )

                        press_angle = None
                        if target_center is not None:
                            lead = float(settings.get("hit_lead_degrees", 2.0))
                            # Always show the planned spot once a zone exists.
                            # Needle direction only determines which side receives
                            # the lead adjustment.
                            if needle_angle is None:
                                press_angle = target_center
                            else:
                                delta = (
                                    target_center - needle_angle + 540.0
                                ) % 360.0 - 180.0
                                direction = 1.0 if delta >= 0.0 else -1.0
                                press_angle = (
                                    target_center - direction * lead
                                ) % 360.0

                        from .shared import Geometry
                        needle_confidence = float(
                            getattr(self.needle, "last_confidence", 0.0)
                        )
                        zone_confidence = float(
                            zone_data.get("confidence", 0.0)
                            if zone_data is not None
                            else getattr(self.zones, "last_confidence", 0.0)
                        )

                        press_ready = False
                        if (
                            needle_angle is not None
                            and press_angle is not None
                        ):
                            distance = self._angle_distance(
                                needle_angle,
                                press_angle,
                            )
                            press_ready = distance <= float(
                                settings.get("press_window_degrees", 3.0)
                            )

                            rearm_distance = float(
                                settings.get("rearm_distance_degrees", 12.0)
                            )
                            if distance >= rearm_distance:
                                self.press_armed = True

                            cooldown_seconds = (
                                float(settings.get("press_cooldown_ms", 90))
                                / 1000.0
                            )
                            trigger_crossing = (
                                press_ready
                                and self.press_armed
                                and now - self.last_press_time
                                >= cooldown_seconds
                            )

                            if trigger_crossing:
                                if settings.get(
                                    "simulate_press_enabled",
                                    True,
                                ):
                                    self.simulated_press_frames = max(
                                        1,
                                        int(
                                            settings.get(
                                                "simulate_press_flash_frames",
                                                8,
                                            )
                                        ),
                                    )

                                if settings.get(
                                    "auto_press_enabled",
                                    False,
                                ):
                                    self.press_frames_remaining = max(
                                        1,
                                        int(
                                            settings.get(
                                                "press_duration_frames",
                                                2,
                                            )
                                        ),
                                    )

                                self.last_press_time = now
                                self.press_armed = False
                                self.snapshot.status = (
                                    f"PRESS {target_type}"
                                )

                        self.snapshot.geometry = Geometry(
                            needle_angle=needle_angle,
                            good_angle=good_angle,
                            great_angle=great_angle,
                            press_angle=press_angle,
                            target_type=target_type,
                            press_ready=press_ready,
                            needle_confidence=needle_confidence,
                            zone_confidence=zone_confidence,
                            good_width=float(
                                zone_data.get("good_width", 0.0)
                                if zone_data is not None else 0.0
                            ),
                            great_width=float(
                                zone_data.get("great_width", 0.0)
                                if zone_data is not None else 0.0
                            ),
                            simulated_press=self.simulated_press_frames > 0,
                        )

                    geometry = self.snapshot.geometry
                    needle_ok = (
                        geometry is not None
                        and geometry.needle_angle is not None
                    )
                    zone_ok = (
                        geometry is not None
                        and geometry.good_angle is not None
                    )

                    self.snapshot.status = (
                        f"Tracking {geometry.target_type}"
                        if zone_ok
                        else "Tracking circle"
                    )
                    self.snapshot.details = (
                        f"track={self.tracker.last_score:.3f} | "
                        f"needle={'yes' if needle_ok else 'no'} "
                        f"px={self.needle.last_pixel_count} "
                        f"conf={getattr(self.needle, 'last_confidence', 0.0):.2f} | "
                        f"zone={'yes' if zone_ok else 'no'} "
                        f"conf={getattr(self.zones, 'last_confidence', 0.0):.2f} "
                        f"base={self.zones.last_baseline:.1f} | "
                        f"needle={needle_rate}/s zone={zone_rate}/s"
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
