from __future__ import annotations

import cv2
import math
import numpy as np


def _draw_arc(frame, cx, cy, radius, center_angle, width, color, thickness):
    if center_angle is None or width <= 0.0:
        return

    start = float(center_angle) - float(width) / 2.0
    end = float(center_angle) + float(width) / 2.0
    points = []
    for angle in np.linspace(start, end, max(8, int(width * 1.4))):
        radians = math.radians(angle)
        points.append([
            int(cx + math.cos(radians) * radius),
            int(cy + math.sin(radians) * radius),
        ])

    if len(points) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_runtime_overlay(frame, settings, snapshot, fps, process_ms):
    if not settings.get("overlay_enabled", True):
        return frame

    cv2.rectangle(frame, (12, 12), (620, 172), (12, 12, 12), -1)
    cv2.rectangle(frame, (12, 12), (620, 172), (85, 85, 85), 1)

    lines = [
        f"Xona's DBD Checker | {snapshot.state}",
        f"Status: {snapshot.status}",
        f"FPS: {fps:.1f} | Process: {process_ms:.2f} ms",
        f"Candidates: {snapshot.candidate_count} | Best score: {snapshot.best_score:.1f}",
        f"Radius: {settings.get('radius_expected', 0):.2f}px "
        f"(acq ±{settings.get('radius_acquisition_tolerance', 0):.1f})",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (26, 42 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    if (
        snapshot.state == "ACQUIRE"
        and settings.get("draw_acquisition_candidates", False)
    ):
        candidates = getattr(snapshot, "debug_candidates", [])
        for candidate in candidates[:8]:
            center = (int(candidate.x), int(candidate.y))
            radius = int(candidate.radius)
            color = (0, 140, 255)
            cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{candidate.score:.2f}",
                (center[0] + radius + 3, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    if snapshot.circle is not None:
        center = (int(snapshot.circle.x), int(snapshot.circle.y))
        radius = int(snapshot.circle.radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 3, cv2.LINE_AA)

    if snapshot.geometry is not None and snapshot.circle is not None:
        cx = snapshot.circle.x
        cy = snapshot.circle.y
        radius = snapshot.circle.radius

        if settings.get("draw_zone_arcs", True):
            _draw_arc(
                frame,
                cx,
                cy,
                radius * 1.03,
                snapshot.geometry.good_angle,
                snapshot.geometry.good_width,
                (255, 255, 0),
                5,
            )
            _draw_arc(
                frame,
                cx,
                cy,
                radius * 1.09,
                snapshot.geometry.great_angle,
                snapshot.geometry.great_width,
                (255, 0, 255),
                6,
            )

        line_specs = [
            (
                snapshot.geometry.needle_angle,
                (0, 0, 255),
                3,
                settings.get("draw_needle_line", True),
                "NEEDLE",
            ),
            (
                snapshot.geometry.good_angle,
                (255, 255, 0),
                2,
                settings.get("draw_good_line", True),
                "GOOD",
            ),
            (
                snapshot.geometry.great_angle,
                (255, 0, 255),
                2,
                settings.get("draw_great_line", True),
                "GREAT",
            ),
            (
                snapshot.geometry.press_angle,
                (255, 80, 0),
                4,
                settings.get("draw_press_line", True),
                "PRESS",
            ),
        ]

        for value, color, thickness, enabled, label in line_specs:
            if value is None or not enabled:
                continue

            radians = math.radians(value)
            end = (
                int(cx + math.cos(radians) * radius),
                int(cy + math.sin(radians) * radius),
            )
            cv2.line(
                frame,
                (int(cx), int(cy)),
                end,
                color,
                thickness,
                cv2.LINE_AA,
            )

            if label == "PRESS":
                cv2.circle(frame, end, 6, color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"{snapshot.geometry.target_type} PRESS",
                    (end[0] + 8, max(18, end[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if settings.get("draw_analysis_confidence", True):
            cv2.putText(
                frame,
                (
                    f"Needle {snapshot.geometry.needle_confidence:.2f} | "
                    f"Zone {snapshot.geometry.zone_confidence:.2f}"
                ),
                (
                    int(cx - radius),
                    max(18, int(cy + radius + 28)),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )

        if snapshot.geometry.simulated_press:
            cv2.circle(
                frame,
                (int(cx), int(cy)),
                int(radius * 1.20),
                (0, 255, 255),
                7,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "SIMULATED PRESS",
                (int(cx - radius), int(cy - radius - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


    if (
        settings.get("center_only_detection", False)
        and snapshot.state == "ACQUIRE"
    ):
        height, width = frame.shape[:2]
        width_ratio = float(
            settings.get("center_region_width_ratio", 0.58)
        )
        height_ratio = float(
            settings.get("center_region_height_ratio", 0.62)
        )
        box_width = int(width * width_ratio)
        box_height = int(height * height_ratio)
        x0 = (width - box_width) // 2
        y0 = (height - box_height) // 2

        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (255, 180, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "CENTER-ONLY SEARCH",
            (x0 + 8, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 180, 40),
            2,
            cv2.LINE_AA,
        )

    return frame
