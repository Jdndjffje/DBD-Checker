from __future__ import annotations

import math

import cv2
import numpy as np

from .acquisition import AcquisitionEngine


class NeedleDetector:
    """
    Angular-histogram needle detector.

    Rather than depending on one exact red contour touching the center, this
    scores every angle by how much red evidence lies along a radial spoke.
    This tolerates anti-aliasing, broken red segments, and motion blur.
    """

    def __init__(self):
        self.last_angle = None
        self.last_confidence = 0.0
        self.last_pixel_count = 0
        self.last_span_ratio = 0.0
        self.missed_frames = 0

    @staticmethod
    def _smooth_angle(previous, current, alpha):
        if current is None:
            return previous
        if previous is None:
            return float(current) % 360.0
        delta = (float(current) - float(previous) + 180.0) % 360.0 - 180.0
        return (float(previous) + delta * float(alpha)) % 360.0

    @staticmethod
    def _angle_distance(first, second):
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    def reset(self):
        self.last_angle = None
        self.last_confidence = 0.0
        self.last_pixel_count = 0
        self.last_span_ratio = 0.0
        self.missed_frames = 0

    def detect(self, frame, circle, settings):
        if circle is None:
            self.reset()
            return None

        cx = float(circle.x)
        cy = float(circle.y)
        radius = float(circle.radius)

        pad = int(radius * 1.20) + 4
        x0 = max(0, int(cx) - pad)
        y0 = max(0, int(cy) - pad)
        x1 = min(frame.shape[1], int(cx) + pad + 1)
        y1 = min(frame.shape[0], int(cy) + pad + 1)

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return self.last_angle

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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

        local_cx = cx - x0
        local_cy = cy - y0

        ys, xs = np.where(red > 0)
        self.last_pixel_count = int(len(xs))

        if len(xs) < int(settings.get("needle_min_pixels", 3)):
            self.missed_frames += 1
            if self.missed_frames > int(settings.get("needle_hold_frames", 5)):
                self.last_angle = None
                self.last_confidence = 0.0
            return self.last_angle

        dx = xs.astype(np.float32) - local_cx
        dy = ys.astype(np.float32) - local_cy
        radial = np.hypot(dx, dy)

        valid = (
            (radial >= radius * 0.03)
            & (radial <= radius * 1.16)
        )
        if not np.any(valid):
            self.missed_frames += 1
            return self.last_angle

        dx = dx[valid]
        dy = dy[valid]
        radial = radial[valid]
        saturation = hsv[:, :, 1][ys[valid], xs[valid]].astype(np.float32)
        value = hsv[:, :, 2][ys[valid], xs[valid]].astype(np.float32)

        self.last_span_ratio = float(np.max(radial) / max(radius, 1.0))

        bins = max(
            72,
            min(360, int(settings.get("needle_angle_bins", 180))),
        )
        angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        indices = np.floor(angle * bins / 360.0).astype(np.int32)
        indices = np.clip(indices, 0, bins - 1)

        # Radial position and color intensity weight pixels closer to the actual
        # needle tip more heavily, while still allowing broken center segments.
        radial_weight = np.clip(radial / max(radius, 1.0), 0.08, 1.15)
        color_weight = (
            np.clip(saturation / 255.0, 0.0, 1.0) * 0.55
            + np.clip(value / 255.0, 0.0, 1.0) * 0.45
        )
        weights = radial_weight * color_weight

        histogram = np.bincount(
            indices,
            weights=weights,
            minlength=bins,
        ).astype(np.float32)

        # Circular smoothing joins anti-aliased neighboring angle bins.
        kernel = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)
        padded = np.concatenate([histogram[-2:], histogram, histogram[:2]])
        smooth = np.convolve(padded, kernel, mode="valid")

        peak_index = int(np.argmax(smooth))
        peak_value = float(smooth[peak_index])
        total = float(np.sum(smooth))
        confidence = peak_value / max(total, 1e-6) * bins / 8.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        minimum_confidence = float(
            settings.get("needle_peak_min_score", 0.18)
        )
        minimum_span = float(
            settings.get("needle_min_span_ratio", 0.22)
        )

        if confidence < minimum_confidence or self.last_span_ratio < minimum_span:
            self.missed_frames += 1
            if self.missed_frames > int(settings.get("needle_hold_frames", 5)):
                self.last_angle = None
                self.last_confidence = 0.0
            return self.last_angle

        detected_angle = (
            (peak_index + 0.5) * 360.0 / bins
        ) % 360.0

        # Reject impossible one-frame jumps unless confidence is very strong.
        if self.last_angle is not None:
            jump = self._angle_distance(detected_angle, self.last_angle)
            max_jump = float(settings.get("needle_max_jump_degrees", 55.0))
            if jump > max_jump and confidence < 0.70:
                self.missed_frames += 1
                return self.last_angle

        self.missed_frames = 0
        self.last_confidence = confidence
        self.last_angle = self._smooth_angle(
            self.last_angle,
            detected_angle,
            float(settings.get("needle_angle_smoothing", 0.42)),
        )
        return self.last_angle
