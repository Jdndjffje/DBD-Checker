from __future__ import annotations

import cv2
import numpy as np


class ZoneDetector:
    def __init__(self):
        self.reset()

    @staticmethod
    def _smooth_angle(previous, current, alpha):
        if current is None:
            return previous
        if previous is None:
            return float(current) % 360.0
        delta = (float(current) - float(previous) + 180.0) % 360.0 - 180.0
        return (float(previous) + delta * float(alpha)) % 360.0

    @staticmethod
    def _circular_runs(signal):
        signal = np.asarray(signal, dtype=np.uint8)
        count = int(signal.size)
        if count == 0 or not np.any(signal):
            return []

        doubled = np.concatenate([signal, signal])
        padded = np.pad(doubled, (1, 1))
        changes = np.diff(padded.astype(np.int8))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        output = []
        for start, end in zip(starts, ends):
            if start >= count:
                continue
            length = min(int(end - start + 1), count)
            output.append((int(start), length))
        return output

    @staticmethod
    def _center_angle(start, length, bins):
        return ((start + (length - 1) * 0.5) * 360.0 / bins) % 360.0

    def reset(self):
        self.last_good_angle = None
        self.last_great_angle = None
        self.last_good_width = 0.0
        self.last_great_width = 0.0
        self.last_target_type = "NONE"
        self.last_baseline = 0.0
        self.last_confidence = 0.0
        self.missed_frames = 0

    def _held_result(self, settings):
        self.missed_frames += 1
        if self.missed_frames > int(settings.get("zone_hold_frames", 8)):
            self.reset()
            return None

        if self.last_good_angle is None:
            return None

        return {
            "good_angle": self.last_good_angle,
            "good_width": self.last_good_width,
            "great_angle": self.last_great_angle,
            "great_width": self.last_great_width,
            "target_type": self.last_target_type,
            "baseline": self.last_baseline,
            "confidence": self.last_confidence,
        }

    def detect(self, frame, circle, settings):
        if circle is None:
            self.reset()
            return None

        cx = float(circle.x)
        cy = float(circle.y)
        radius = float(circle.radius)

        pad = int(radius * 1.28) + 3
        x0 = max(0, int(cx) - pad)
        y0 = max(0, int(cy) - pad)
        x1 = min(frame.shape[1], int(cx) + pad + 1)
        y1 = min(frame.shape[0], int(cy) + pad + 1)

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return self._held_result(settings)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        local_x = cx - x0
        local_y = cy - y0

        yy, xx = np.indices(hsv.shape[:2], dtype=np.float32)
        dx = xx - local_x
        dy = yy - local_y
        radial = np.hypot(dx, dy)
        angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        annulus = (
            (radial >= radius * float(settings.get("zone_inner_ratio", 0.67)))
            & (radial <= radius * float(settings.get("zone_outer_ratio", 1.22)))
        )

        bright = (
            annulus
            & (hsv[:, :, 2] >= int(settings.get("zone_brightness_min", 155)))
            & (hsv[:, :, 1] <= int(settings.get("zone_saturation_max", 110)))
        )

        ys, xs = np.where(bright)
        if len(xs) < 12:
            return self._held_result(settings)

        bins = max(90, min(360, int(settings.get("zone_bins", 120))))
        indices = np.floor(angles[ys, xs] * bins / 360.0).astype(np.int32)
        indices = np.clip(indices, 0, bins - 1)

        counts = np.bincount(indices, minlength=bins).astype(np.float32)
        radial_values = radial[ys, xs].astype(np.float32)

        radial_min = np.full(bins, np.inf, dtype=np.float32)
        radial_max = np.full(bins, -np.inf, dtype=np.float32)
        np.minimum.at(radial_min, indices, radial_values)
        np.maximum.at(radial_max, indices, radial_values)

        span = np.zeros(bins, dtype=np.float32)
        valid = np.isfinite(radial_min) & np.isfinite(radial_max)
        span[valid] = radial_max[valid] - radial_min[valid]

        kernel = np.array([0.20, 0.60, 0.20], dtype=np.float32)
        counts_smooth = np.convolve(
            np.concatenate([counts[-1:], counts, counts[:1]]),
            kernel,
            mode="valid",
        )
        span_smooth = np.convolve(
            np.concatenate([span[-1:], span, span[:1]]),
            kernel,
            mode="valid",
        )

        positive_counts = counts_smooth[counts_smooth > 0]
        positive_spans = span_smooth[span_smooth > 0]
        if positive_counts.size < 8 or positive_spans.size < 8:
            return self._held_result(settings)

        count_baseline = float(np.median(positive_counts))
        span_baseline = float(np.median(positive_spans))
        self.last_baseline = span_baseline

        good_signal = (
            counts_smooth
            >= count_baseline
            * float(settings.get("zone_density_multiplier", 1.35))
        ) & (
            span_smooth
            >= max(
                1.0,
                span_baseline
                * float(settings.get("zone_thickness_multiplier", 1.16)),
            )
        )

        great_signal = (
            counts_smooth
            >= count_baseline
            * float(settings.get("great_density_multiplier", 1.75))
        ) & (
            span_smooth
            >= max(
                1.0,
                span_baseline
                * float(settings.get("great_thickness_multiplier", 1.42)),
            )
        )

        degrees_per_bin = 360.0 / bins
        minimum_good = max(
            2,
            int(
                round(
                    float(settings.get("zone_min_run_degrees", 5))
                    / degrees_per_bin
                )
            ),
        )
        minimum_great = max(
            1,
            int(
                round(
                    float(settings.get("great_min_run_degrees", 3))
                    / degrees_per_bin
                )
            ),
        )

        good_runs = [
            run
            for run in self._circular_runs(good_signal)
            if minimum_good <= run[1] <= int(bins * 0.28)
        ]
        if not good_runs:
            return self._held_result(settings)

        def run_score(run):
            start, length = run
            idx = np.arange(start, start + length) % bins
            return float(
                np.sum(counts_smooth[idx])
                + np.sum(span_smooth[idx]) * 1.8
            )

        good_run = max(good_runs, key=run_score)
        good_angle = self._center_angle(*good_run, bins)
        good_width = good_run[1] * degrees_per_bin

        good_indices = set(
            (np.arange(good_run[0], good_run[0] + good_run[1]) % bins).tolist()
        )

        valid_great = []
        for run in self._circular_runs(great_signal):
            if not (minimum_great <= run[1] <= int(bins * 0.12)):
                continue
            run_indices = set(
                (np.arange(run[0], run[0] + run[1]) % bins).tolist()
            )
            overlap = len(run_indices & good_indices) / max(1, len(run_indices))
            if overlap >= 0.40:
                valid_great.append(run)

        great_angle = None
        great_width = 0.0
        if valid_great:
            great_run = max(valid_great, key=run_score)
            great_angle = self._center_angle(*great_run, bins)
            great_width = great_run[1] * degrees_per_bin

        selected = np.arange(
            good_run[0],
            good_run[0] + good_run[1],
        ) % bins
        selected_strength = float(
            np.mean(counts_smooth[selected]) / max(count_baseline, 1e-6)
        )
        selected_span = float(
            np.mean(span_smooth[selected]) / max(span_baseline, 1e-6)
        )
        zone_confidence = float(
            np.clip(
                (selected_strength - 1.0) * 0.35
                + (selected_span - 1.0) * 0.50
                + min(good_width / 30.0, 1.0) * 0.15,
                0.0,
                1.0,
            )
        )

        alpha = float(settings.get("zone_angle_smoothing", 0.30))
        self.last_good_angle = self._smooth_angle(
            self.last_good_angle,
            good_angle,
            alpha,
        )
        self.last_good_width = good_width

        if great_angle is not None:
            self.last_great_angle = self._smooth_angle(
                self.last_great_angle,
                great_angle,
                alpha,
            )
            self.last_great_width = great_width
            self.last_target_type = "GREAT"
        else:
            self.last_great_angle = None
            self.last_great_width = 0.0
            self.last_target_type = "GOOD"

        self.missed_frames = 0
        self.last_confidence = zone_confidence

        return {
            "good_angle": self.last_good_angle,
            "good_width": self.last_good_width,
            "great_angle": self.last_great_angle,
            "great_width": self.last_great_width,
            "target_type": self.last_target_type,
            "baseline": self.last_baseline,
            "confidence": self.last_confidence,
        }
