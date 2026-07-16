from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from threading import RLock


APP_NAME = "XonasDBDCheckerV24"
DATA_DIR = Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / APP_NAME
CONFIG_PATH = DATA_DIR / "config.json"
FROZEN_FRAME_PATH = DATA_DIR / "calibration_frame.png"
REFERENCE_CAPTURE_PATH = DATA_DIR / "reference_capture.png"
REFERENCES_DIR = DATA_DIR / "references"
POSITIVE_REFERENCES_DIR = REFERENCES_DIR / "positive"
NEGATIVE_REFERENCES_DIR = REFERENCES_DIR / "negative"
REFERENCE_CACHE_PATH = DATA_DIR / "reference_cache.npz"

DEFAULTS = {
    "enabled": True,
    "ui_enabled": True,
    "overlay_enabled": True,
    "debug_overlay": True,

    "detector_mode": "HSV + Geometry",
    "state": "IDLE",

    "capture_fps_target": 120,
    "acquisition_rate": 30,
    "tracking_rate": 120,
    "tracking_recenter_rate": 30,

    # Fixed-radius acquisition/tracking.
    "search_width": 480,

    # Reference-library acquisition.
    "reference_detection_enabled": True,
    "reference_match_threshold": 0.29,
    "reference_edge_weight": 0.64,
    "reference_gray_weight": 0.36,
    "reference_crop_radius_multiplier": 1.45,
    "reference_normalized_size": 128,
    "reference_max_peaks": 6,
    "reference_peak_suppression": 0.72,
    "reference_reload_request": 0,
    "reference_positive_count": 0,
    "reference_negative_count": 0,

    # Reference capture/editor.
    "reference_capture_request": False,
    "reference_capture_ready": False,
    "reference_capture_request_id": 0,
    "reference_capture_center_x": 0.0,
    "reference_capture_center_y": 0.0,
    "reference_capture_radius": 89.164,

    # Search-region controls.
    "center_only_detection": False,
    "center_region_width_ratio": 0.58,
    "center_region_height_ratio": 0.62,

    # Recovery throttling. This prevents acquisition from monopolizing the
    # Creative worker after a tracked skill check disappears.
    "lost_detection_cooldown_ms": 140,
    "recovery_acquisition_rate": 20,
    "hough_timeout_guard_ms": 8.0,
    "acquisition_frame_budget_ms": 8.0,
    "acquisition_radius_steps": 3,
    "acquisition_use_red_seeds": True,
    "red_seed_min_length_ratio": 0.10,
    "red_seed_max_length_ratio": 1.45,
    "red_seed_min_aspect": 2.0,

    # Geometry-first acquisition.
    "geometry_first_enabled": True,
    "template_candidate_floor": 0.04,
    "template_always_keep": 4,
    "hough_fallback_enabled": False,
    "hough_fallback_rate": 4,
    "hough_dp": 1.20,
    "hough_param1": 105,
    "hough_param2": 9,
    "hough_radius_tolerance_ratio": 0.18,
    "geometry_accept_threshold": 0.30,
    "geometry_edge_coverage_target": 0.38,
    "geometry_edge_peak_target": 0.20,
    "geometry_dark_center_weight": 0.10,

    # Red-needle candidate seeding.
    "red_seed_enabled": True,
    "red_seed_min_area": 4,
    "red_seed_max_area": 600,
    "red_seed_max_components": 8,
    "red_seed_local_search": 3,

    # False-lock rejection.
    "acquisition_require_needle": True,
    "acquisition_min_red_score": 0.30,
    "acquisition_min_needle_line_score": 0.54,
    "acquisition_min_edge_ring": 0.20,
    "acquisition_min_white_ring": 0.10,
    "acquisition_min_center_prompt_score": 0.18,

    # Heavy validation is never allowed on the 120 FPS tracking path.
    "tracking_heavy_validation_rate": 20,
    "tracking_require_needle": True,
    "tracking_needle_grace_frames": 6,
    "acquisition_score_threshold": 0.34,
    "tracking_score_threshold": 0.48,
    "template_peak_threshold": 0.20,
    "max_acquisition_candidates": 8,
    "edge_template_weight": 0.58,
    "white_template_weight": 0.42,
    "edge_low_threshold": 35,
    "edge_high_threshold": 110,
    "edge_radial_weight": 0.38,
    "white_radial_weight": 0.22,
    "template_score_weight": 0.24,
    "red_score_weight": 0.08,
    "inner_rejection_weight": 0.08,
    "tracker_search_pixels": 7,
    "tracker_radius_pixels": 2,
    "tracker_lost_frames": 18,
    "lock_confirmation_frames": 2,
    "draw_acquisition_candidates": False,

    "radius_reference_width": 2559,
    "radius_reference_height": 1435,
    "radius_expected": 89.164,
    "radius_acquisition_tolerance": 10.0,
    "radius_tracking_tolerance": 14.0,

    "calibration_active": False,
    "calibration_capture_request": False,
    "calibration_capture_ready": False,
    "calibration_request_id": 0,
    "calibration_center_x": 1280.0,
    "calibration_center_y": 720.0,
    "calibration_radius": 89.164,
    "calibration_move_step": 1.0,
    "calibration_radius_step": 1.0,

    "ring_lower": [0, 0, 140],
    "ring_upper": [179, 45, 255],
    "zone_lower": [125, 0, 210],
    "zone_upper": [142, 35, 255],
    "great_lower": [125, 0, 210],
    "great_upper": [165, 20, 255],
    "red_lower_a": [172, 220, 150],
    "red_upper_a": [179, 255, 255],
    "red_lower_b": [0, 220, 150],
    "red_upper_b": [4, 255, 255],

    # Needle / zone analysis.
    "needle_analysis_rate": 120,
    "zone_analysis_rate": 20,
    "needle_min_pixels": 3,
    "needle_min_span_ratio": 0.22,
    "needle_center_ratio": 0.70,
    "needle_discovery_lower_a": [166, 120, 70],
    "needle_discovery_upper_a": [179, 255, 255],
    "needle_discovery_lower_b": [0, 120, 70],
    "needle_discovery_upper_b": [12, 255, 255],
    "needle_angle_smoothing": 0.42,
    "needle_angle_bins": 120,
    "needle_peak_min_score": 0.18,
    "needle_hold_frames": 5,
    "needle_max_jump_degrees": 55.0,

    "zone_bins": 120,
    "zone_brightness_min": 155,
    "zone_saturation_max": 110,
    "zone_density_multiplier": 1.35,
    "great_density_multiplier": 1.75,
    "zone_inner_ratio": 0.67,
    "zone_outer_ratio": 1.22,
    "zone_min_run_degrees": 5,
    "great_min_run_degrees": 3,
    "zone_thickness_multiplier": 1.16,
    "great_thickness_multiplier": 1.42,
    "zone_angle_smoothing": 0.30,
    "zone_hold_frames": 8,

    "hit_lead_degrees": 2.0,
    "press_window_degrees": 3.0,
    "auto_press_enabled": False,
    "press_button_index": 15,
    "press_duration_frames": 2,
    "press_cooldown_ms": 90,
    "rearm_distance_degrees": 12.0,

    "draw_needle_line": True,
    "draw_good_line": True,
    "draw_great_line": True,
    "draw_press_line": True,
    "draw_zone_arcs": True,
    "draw_analysis_confidence": False,
    "simulate_press_enabled": True,
    "simulate_press_flash_frames": 8,

    "runtime_status": "Waiting for CVWorker",
    "runtime_details": "—",
    "runtime_width": 0,
    "runtime_height": 0,
    "runtime_fps": 0.0,
    "runtime_ms": 0.0,
    "runtime_state": "IDLE",
}


class ConfigStore:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cache = dict(DEFAULTS)
        self._mtime = None
        self._last_check = 0.0
        self.load(force=True)

    def load(self, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_check < 0.20:
                return dict(self._cache)
            self._last_check = now

            try:
                mtime = CONFIG_PATH.stat().st_mtime
                if not force and self._mtime == mtime:
                    return dict(self._cache)

                loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                merged = dict(DEFAULTS)
                if isinstance(loaded, dict):
                    merged.update(loaded)
                self._cache = merged
                self._mtime = mtime
            except (OSError, ValueError, TypeError):
                if not CONFIG_PATH.exists():
                    self.save(self._cache)

            return dict(self._cache)

    def save(self, values):
        with self._lock:
            merged = dict(DEFAULTS)
            merged.update(values)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temporary = CONFIG_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            temporary.replace(CONFIG_PATH)
            self._cache = merged
            try:
                self._mtime = CONFIG_PATH.stat().st_mtime
            except OSError:
                self._mtime = None
            return dict(self._cache)

    def update(self, **changes):
        current = self.load(force=True)
        current.update(changes)
        return self.save(current)
