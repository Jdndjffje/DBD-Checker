from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .reference_library import ReferenceLibrary
from .shared import Circle


@dataclass
class CandidateDebug:
    x: float
    y: float
    radius: float
    score: float
    template_score: float = 0.0
    white_template_score: float = 0.0
    edge_template_score: float = 0.0
    ring_coverage: float = 0.0
    edge_ring_coverage: float = 0.0
    edge_radial_peak: float = 0.0
    radial_peak: float = 0.0
    inner_fill: float = 0.0
    red_score: float = 0.0
    source: str = "reference"


class AcquisitionEngine:
    """
    Reference-based acquisition.

    The expensive geometry/Hough stack is gone. Each acquisition pass performs:
      - one resize
      - one grayscale conversion
      - one Canny pass
      - two matchTemplate calls against the averaged reference
      - scoring only the strongest peaks
    """

    def __init__(self, config_store=None):
        self.config_store = config_store
        self.references = (
            ReferenceLibrary(config_store)
            if config_store is not None
            else None
        )

        self.last_candidate_count = 0
        self.last_best_score = 0.0
        self.last_candidates = []
        self.last_search_ms = 0.0

        self._angles = np.linspace(0.0, math.tau, 96, endpoint=False)
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

        self._tracker_template_key = None
        self._tracker_white_template = None
        self._tracker_edge_template = None

    def reset(self):
        self.last_candidate_count = 0
        self.last_best_score = 0.0
        self.last_candidates = []
        self.last_search_ms = 0.0

    @staticmethod
    def _hsv_mask(hsv, lower, upper):
        lower = np.asarray(lower, dtype=np.uint8)
        upper = np.asarray(upper, dtype=np.uint8)

        if int(lower[0]) <= int(upper[0]):
            return cv2.inRange(hsv, lower, upper)

        lower_a = lower.copy()
        upper_a = upper.copy()
        upper_a[0] = 179
        lower_b = lower.copy()
        lower_b[0] = 0
        upper_b = upper.copy()

        return cv2.bitwise_or(
            cv2.inRange(hsv, lower_a, upper_a),
            cv2.inRange(hsv, lower_b, upper_b),
        )

    @staticmethod
    def scaled_radius(frame_width, frame_height, settings):
        reference_width = max(
            1.0,
            float(settings.get("radius_reference_width", 2559)),
        )
        reference_height = max(
            1.0,
            float(settings.get("radius_reference_height", 1435)),
        )
        reference_radius = max(
            2.0,
            float(settings.get("radius_expected", 89.164)),
        )
        return reference_radius * min(
            frame_width / reference_width,
            frame_height / reference_height,
        )

    @staticmethod
    def _search_region(frame, settings):
        if not settings.get("center_only_detection", False):
            return frame, 0, 0

        height, width = frame.shape[:2]
        width_ratio = min(
            max(float(settings.get("center_region_width_ratio", 0.58)), 0.15),
            1.0,
        )
        height_ratio = min(
            max(float(settings.get("center_region_height_ratio", 0.62)), 0.15),
            1.0,
        )
        crop_width = max(160, int(width * width_ratio))
        crop_height = max(160, int(height * height_ratio))
        x0 = max(0, (width - crop_width) // 2)
        y0 = max(0, (height - crop_height) // 2)
        return frame[y0:y0 + crop_height, x0:x0 + crop_width], x0, y0

    @staticmethod
    def _remove_red(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(
                hsv,
                np.array([166, 100, 55], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            ),
            cv2.inRange(
                hsv,
                np.array([0, 100, 55], dtype=np.uint8),
                np.array([12, 255, 255], dtype=np.uint8),
            ),
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.any(red):
            gray[red > 0] = int(np.median(gray))
        return gray

    @staticmethod
    def _peak_locations(response, maximum, suppression_radius):
        working = response.copy()
        peaks = []

        for _ in range(maximum):
            _minimum, maximum_value, _minimum_location, location = cv2.minMaxLoc(
                working
            )
            if not np.isfinite(maximum_value):
                break
            peaks.append((location[0], location[1], float(maximum_value)))

            x, y = location
            x0 = max(0, x - suppression_radius)
            y0 = max(0, y - suppression_radius)
            x1 = min(working.shape[1], x + suppression_radius + 1)
            y1 = min(working.shape[0], y + suppression_radius + 1)
            working[y0:y1, x0:x1] = -1.0

        return peaks

    def search(self, frame, settings):
        started = time.perf_counter()

        if self.references is None:
            self.reset()
            return None

        self.references.reload_if_requested(settings)
        if (
            self.references.positive_gray is None
            or self.references.positive_edge is None
        ):
            self.reset()
            self.last_search_ms = (time.perf_counter() - started) * 1000.0
            return None

        frame_height, frame_width = frame.shape[:2]
        search_frame, offset_x, offset_y = self._search_region(frame, settings)
        search_height, search_width = search_frame.shape[:2]

        target_width = max(320, int(settings.get("search_width", 480)))
        scale = min(1.0, target_width / max(float(search_width), 1.0))

        if scale < 1.0:
            small = cv2.resize(
                search_frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = search_frame

        gray = self._remove_red(small)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        edge = cv2.Canny(gray, 35, 105)

        expected_radius = self.scaled_radius(
            frame_width,
            frame_height,
            settings,
        )
        expected_radius_small = expected_radius * scale
        crop_multiplier = float(
            settings.get("reference_crop_radius_multiplier", 1.45)
        )
        template_size = max(
            24,
            int(round(expected_radius_small * crop_multiplier * 2.0)),
        )
        if template_size % 2 == 0:
            template_size += 1

        if (
            gray.shape[0] < template_size
            or gray.shape[1] < template_size
        ):
            self.reset()
            return None

        reference_gray = cv2.resize(
            self.references.positive_gray,
            (template_size, template_size),
            interpolation=cv2.INTER_AREA,
        )
        reference_edge = cv2.resize(
            self.references.positive_edge,
            (template_size, template_size),
            interpolation=cv2.INTER_AREA,
        )

        gray_source = gray.astype(np.float32) / 255.0
        edge_source = edge.astype(np.float32) / 255.0

        gray_response = cv2.matchTemplate(
            gray_source,
            reference_gray,
            cv2.TM_CCOEFF_NORMED,
        )
        edge_response = cv2.matchTemplate(
            edge_source,
            reference_edge,
            cv2.TM_CCOEFF_NORMED,
        )

        edge_weight = float(settings.get("reference_edge_weight", 0.64))
        gray_weight = float(settings.get("reference_gray_weight", 0.36))
        denominator = max(edge_weight + gray_weight, 1e-6)
        response = (
            edge_response * edge_weight
            + gray_response * gray_weight
        ) / denominator

        maximum_peaks = max(
            1,
            int(settings.get("reference_max_peaks", 6)),
        )
        suppression = max(
            4,
            int(
                template_size
                * float(settings.get("reference_peak_suppression", 0.72))
            ),
        )
        peaks = self._peak_locations(response, maximum_peaks, suppression)

        half = template_size / 2.0
        candidates = []
        for x, y, score in peaks:
            candidates.append(
                CandidateDebug(
                    x=(x + half) / scale + offset_x,
                    y=(y + half) / scale + offset_y,
                    radius=expected_radius,
                    score=score,
                    template_score=score,
                    edge_template_score=float(edge_response[y, x]),
                    white_template_score=float(gray_response[y, x]),
                    source="reference",
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        self.last_candidates = candidates
        self.last_candidate_count = len(candidates)
        self.last_best_score = candidates[0].score if candidates else 0.0
        self.last_search_ms = (time.perf_counter() - started) * 1000.0

        threshold = float(settings.get("reference_match_threshold", 0.48))
        if not candidates or candidates[0].score < threshold:
            return None

        best = candidates[0]
        return Circle(
            x=best.x,
            y=best.y,
            radius=best.radius,
            score=best.score,
            source="reference",
        )

    # ------------------------------------------------------------------
    # Compatibility methods used by the fast tracker.
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_binary(mask, xs, ys):
        height, width = mask.shape
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        valid = (
            (xi >= 0)
            & (xi < width)
            & (yi >= 0)
            & (yi < height)
        )
        output = np.zeros_like(xs, dtype=np.float32)
        output[valid] = (mask[yi[valid], xi[valid]] > 0).astype(np.float32)
        return output

    def _radial_evidence(self, mask, cx, cy, radius):
        samples = []
        for offset in (-0.08, -0.03, 0.03, 0.08):
            sample_radius = radius * (1.0 + offset)
            samples.append(
                self._sample_binary(
                    mask,
                    cx + self._cos * sample_radius,
                    cy + self._sin * sample_radius,
                )
            )
        ring = np.maximum.reduce(samples)
        coverage = float(ring.mean())
        inner = self._sample_binary(
            mask,
            cx + self._cos * radius * 0.72,
            cy + self._sin * radius * 0.72,
        )
        outer = self._sample_binary(
            mask,
            cx + self._cos * radius * 1.24,
            cy + self._sin * radius * 1.24,
        )
        peak = max(
            0.0,
            coverage - float(inner.mean()) * 0.46 - float(outer.mean()) * 0.40,
        )
        return coverage, peak

    def score_circle(
        self,
        white_mask,
        red_mask,
        cx,
        cy,
        radius,
        edge_mask=None,
        gray_image=None,
        settings=None,
        compute_heavy_validation=False,
    ):
        white_coverage, white_peak = self._radial_evidence(
            white_mask, cx, cy, radius
        )
        edge_coverage, edge_peak = self._radial_evidence(
            edge_mask, cx, cy, radius
        )
        score = float(
            np.clip(
                edge_coverage * 0.48
                + edge_peak * 0.30
                + white_coverage * 0.14
                + white_peak * 0.08,
                0.0,
                1.0,
            )
        )
        return {
            "score": score,
            "ring_coverage": white_coverage,
            "radial_peak": white_peak,
            "edge_ring_coverage": edge_coverage,
            "edge_radial_peak": edge_peak,
            "inner_fill": 0.0,
            "red_score": 0.0,
            "needle_line_score": 1.0,
            "center_prompt_score": 1.0,
        }

    def _get_template(self, radius):
        radius = max(4, int(round(radius)))
        size = radius * 2 + 13
        if size % 2 == 0:
            size += 1

        key = (radius, size)
        if key == self._tracker_template_key:
            return self._tracker_white_template, self._tracker_edge_template

        center = size // 2
        yy, xx = np.ogrid[:size, :size]
        distance = np.hypot(xx - center, yy - center)

        white_template = np.zeros((size, size), dtype=np.float32)
        white_template[
            np.abs(distance - radius) <= max(1.5, radius * 0.12)
        ] = 1.0
        white_template -= float(white_template.mean())

        edge_template = np.zeros((size, size), dtype=np.float32)
        edge_template[
            np.abs(distance - radius) <= max(1.2, radius * 0.075)
        ] = 1.0
        edge_template -= float(edge_template.mean())

        self._tracker_template_key = key
        self._tracker_white_template = white_template
        self._tracker_edge_template = edge_template
        return white_template, edge_template
