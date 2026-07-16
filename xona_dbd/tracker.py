from __future__ import annotations

import time

import cv2
import numpy as np

from .acquisition import AcquisitionEngine
from .shared import Circle


class SkillCheckTracker:
    """
    High-FPS tracker.

    Every frame:
      - validates the current fixed-radius ring with one radial sample.

    At a lower recenter rate:
      - runs one small local edge-template match to update the center.

    This replaces the old 1,000+ candidate evaluations per frame.
    """

    def __init__(self):
        self.circle = None
        self.lost_frames = 0
        self.last_score = 0.0
        self.last_recenter = 0.0
        self.missing_needle_frames = 0
        self.last_heavy_validation = 0.0
        self.last_needle_line_score = 1.0
        self.missing_needle_frames = 0
        self.last_heavy_validation = 0.0
        self.last_needle_line_score = 1.0
        self.last_heavy_validation = 0.0
        self.last_needle_line_score = 1.0
        self._scorer = AcquisitionEngine()

    def lock(self, circle):
        self.circle = circle
        self.lost_frames = 0
        self.last_score = circle.score
        self.last_recenter = 0.0
        self.missing_needle_frames = 0

    def reset(self):
        self.circle = None
        self.lost_frames = 0
        self.last_score = 0.0
        self.last_recenter = 0.0
        self.missing_needle_frames = 0

    def _masks(self, roi, settings):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = AcquisitionEngine._hsv_mask(
            hsv,
            settings.get("ring_lower", [0, 0, 140]),
            settings.get("ring_upper", [179, 45, 255]),
        )
        red = cv2.bitwise_or(
            AcquisitionEngine._hsv_mask(
                hsv,
                settings.get("needle_discovery_lower_a", [166, 120, 70]),
                settings.get("needle_discovery_upper_a", [179, 255, 255]),
            ),
            AcquisitionEngine._hsv_mask(
                hsv,
                settings.get("needle_discovery_lower_b", [0, 120, 70]),
                settings.get("needle_discovery_upper_b", [12, 255, 255]),
            ),
        )
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            int(settings.get("edge_low_threshold", 35)),
            int(settings.get("edge_high_threshold", 110)),
            L2gradient=False,
        )
        return white, red, edges

    def update(self, frame, settings):
        if self.circle is None:
            return None

        cx = float(self.circle.x)
        cy = float(self.circle.y)
        radius = float(self.circle.radius)

        margin = max(10, int(settings.get("tracker_search_pixels", 7)) + 4)
        pad = int(radius * 1.38) + margin

        x0 = max(0, int(round(cx)) - pad)
        y0 = max(0, int(round(cy)) - pad)
        x1 = min(frame.shape[1], int(round(cx)) + pad + 1)
        y1 = min(frame.shape[0], int(round(cy)) + pad + 1)

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            self.lost_frames += 1
            return self.circle

        white, red, edges = self._masks(roi, settings)
        local_x = cx - x0
        local_y = cy - y0

        # One cheap validation score per frame.
        metrics = self._scorer.score_circle(
            white,
            red,
            local_x,
            local_y,
            radius,
            edge_mask=edges,
            settings=settings,
            compute_heavy_validation=False,
        )
        self.last_score = float(metrics["score"])

        if settings.get("tracking_require_needle", True):
            heavy_rate = max(
                1,
                int(settings.get("tracking_heavy_validation_rate", 20)),
            )
            now_for_validation = time.monotonic()

            if (
                now_for_validation - self.last_heavy_validation
                >= 1.0 / heavy_rate
            ):
                self.last_heavy_validation = now_for_validation
                heavy_metrics = self._scorer.score_circle(
                    white,
                    red,
                    local_x,
                    local_y,
                    radius,
                    edge_mask=edges,
                    settings=settings,
                    compute_heavy_validation=True,
                )
                self.last_needle_line_score = float(
                    heavy_metrics.get("needle_line_score", 0.0)
                )

                minimum_line = float(
                    settings.get(
                        "acquisition_min_needle_line_score",
                        0.54,
                    )
                ) * 0.72

                if self.last_needle_line_score < minimum_line:
                    self.missing_needle_frames += 1
                else:
                    self.missing_needle_frames = 0

            if self.missing_needle_frames > int(
                settings.get("tracking_needle_grace_frames", 6)
            ):
                self.reset()
                return None

        threshold = float(settings.get("tracking_score_threshold", 0.48))
        if self.last_score < threshold:
            self.lost_frames += 1
        else:
            self.lost_frames = max(0, self.lost_frames - 2)

        # Recenter only 30 times/sec by default.
        now = time.monotonic()
        recenter_rate = max(
            5, min(120, int(settings.get("tracking_recenter_rate", 30)))
        )

        if now - self.last_recenter >= 1.0 / recenter_rate:
            self.last_recenter = now

            expected_small = radius
            _white_template, edge_template = self._scorer._get_template(
                expected_small
            )

            if (
                edges.shape[0] >= edge_template.shape[0]
                and edges.shape[1] >= edge_template.shape[1]
            ):
                response = cv2.matchTemplate(
                    edges.astype(np.float32) / 255.0,
                    edge_template,
                    cv2.TM_CCOEFF_NORMED,
                )

                _, peak, _, location = cv2.minMaxLoc(response)
                if peak >= float(
                    settings.get("template_peak_threshold", 0.20)
                ) * 0.75:
                    new_local_x = (
                        location[0] + edge_template.shape[1] / 2.0
                    )
                    new_local_y = (
                        location[1] + edge_template.shape[0] / 2.0
                    )

                    maximum_move = max(
                        3.0,
                        float(settings.get("tracker_search_pixels", 7)) * 1.8,
                    )
                    dx = np.clip(
                        new_local_x - local_x,
                        -maximum_move,
                        maximum_move,
                    )
                    dy = np.clip(
                        new_local_y - local_y,
                        -maximum_move,
                        maximum_move,
                    )

                    alpha = 0.35
                    cx += float(dx) * alpha
                    cy += float(dy) * alpha

        self.circle = Circle(
            x=cx,
            y=cy,
            radius=radius,
            score=self.last_score,
            source="fast-local-tracker",
        )

        if self.lost_frames >= int(
            settings.get("tracker_lost_frames", 18)
        ):
            self.reset()
            return None

        return self.circle
