from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DATA_DIR,
    NEGATIVE_REFERENCES_DIR,
    POSITIVE_REFERENCES_DIR,
    REFERENCE_CACHE_PATH,
    REFERENCE_CAPTURE_PATH,
)


class ReferenceLibrary:
    def __init__(self, config_store):
        self.config_store = config_store
        self.package_root = Path(__file__).resolve().parent.parent
        self.bundled_root = self.package_root / "bundled_references"
        self.last_reload_request = None
        self.positive_gray = None
        self.positive_edge = None
        self.negative_gray = []
        self.loaded_count = 0

        self._ensure_directories()
        self._install_bundled_references()
        self.rebuild_cache()

    def _ensure_directories(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        POSITIVE_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        NEGATIVE_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    def _install_bundled_references(self):
        for category in ("positive", "negative"):
            source = self.bundled_root / category
            destination = (
                POSITIVE_REFERENCES_DIR
                if category == "positive"
                else NEGATIVE_REFERENCES_DIR
            )
            if not source.exists():
                continue
            for item in source.glob("*.png"):
                target = destination / item.name
                if not target.exists():
                    shutil.copy2(item, target)

    @staticmethod
    def _prepare(image, normalized_size):
        image = cv2.resize(
            image,
            (normalized_size, normalized_size),
            interpolation=cv2.INTER_AREA,
        )

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
            # Remove the variable-angle needle from references.
            replacement = int(np.median(gray[red == 0])) if np.any(red == 0) else 0
            gray[red > 0] = replacement

        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        gray_f = gray.astype(np.float32) / 255.0

        edge = cv2.Canny(gray, 35, 105).astype(np.float32) / 255.0

        # The red needle has already been removed. Keep the complete crop so
        # matchTemplate compares the black circular interior, white ring, and
        # center prompt against the same unmasked source representation.
        return gray_f, edge

    @staticmethod
    def _load_images(directory):
        images = []
        for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for path in sorted(directory.glob(extension)):
                image = cv2.imread(str(path))
                if image is not None and image.size:
                    images.append((path, image))
        return images

    def rebuild_cache(self, normalized_size=128):
        self._ensure_directories()

        positives = self._load_images(POSITIVE_REFERENCES_DIR)
        negatives = self._load_images(NEGATIVE_REFERENCES_DIR)

        prepared_gray = []
        prepared_edge = []

        for _path, image in positives:
            gray, edge = self._prepare(image, normalized_size)
            prepared_gray.append(gray)
            prepared_edge.append(edge)

        if prepared_gray:
            self.positive_gray = np.mean(prepared_gray, axis=0).astype(np.float32)
            self.positive_edge = np.mean(prepared_edge, axis=0).astype(np.float32)
        else:
            self.positive_gray = None
            self.positive_edge = None

        self.negative_gray = [
            self._prepare(image, normalized_size)[0]
            for _path, image in negatives
        ]
        self.loaded_count = len(positives)

        if self.positive_gray is not None:
            np.savez_compressed(
                REFERENCE_CACHE_PATH,
                positive_gray=self.positive_gray,
                positive_edge=self.positive_edge,
            )

        self.config_store.update(
            reference_positive_count=len(positives),
            reference_negative_count=len(negatives),
            runtime_status="Reference cache rebuilt",
            runtime_details=(
                f"positive={len(positives)} negative={len(negatives)}"
            ),
        )

    def reload_if_requested(self, settings):
        request = settings.get("reference_reload_request", 0)
        if request == self.last_reload_request:
            return
        self.last_reload_request = request
        self.rebuild_cache(
            int(settings.get("reference_normalized_size", 128))
        )

    def save_crop(self, frame, center_x, center_y, radius, category="positive"):
        multiplier = 1.45
        half = max(8, int(round(float(radius) * multiplier)))

        x0 = max(0, int(round(center_x)) - half)
        y0 = max(0, int(round(center_y)) - half)
        x1 = min(frame.shape[1], int(round(center_x)) + half + 1)
        y1 = min(frame.shape[0], int(round(center_y)) + half + 1)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            raise RuntimeError("The selected reference crop is empty.")

        directory = (
            POSITIVE_REFERENCES_DIR
            if category == "positive"
            else NEGATIVE_REFERENCES_DIR
        )
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{category}_{int(time.time() * 1000)}.png"
        path = directory / filename

        if not cv2.imwrite(str(path), crop):
            raise RuntimeError("cv2.imwrite returned False")

        return path


class ReferenceCaptureController:
    def __init__(self, config_store):
        self.config_store = config_store
        self.last_request_id = None

    def process(self, frame, settings):
        request_id = settings.get("reference_capture_request_id")
        requested = bool(settings.get("reference_capture_request", False))

        if not requested:
            self.last_request_id = request_id
            return settings

        self.last_request_id = request_id

        try:
            REFERENCE_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(REFERENCE_CAPTURE_PATH), frame):
                raise RuntimeError("cv2.imwrite returned False")

            return self.config_store.update(
                reference_capture_request=False,
                reference_capture_ready=True,
                reference_capture_center_x=frame.shape[1] / 2.0,
                reference_capture_center_y=frame.shape[0] / 2.0,
                reference_capture_radius=float(
                    settings.get("radius_expected", 89.164)
                ),
                runtime_status="Reference frame captured",
                runtime_details=(
                    f"{frame.shape[1]}x{frame.shape[0]} reference editor ready"
                ),
            )
        except Exception as exc:
            return self.config_store.update(
                reference_capture_request=False,
                reference_capture_ready=False,
                runtime_status="Reference capture failed",
                runtime_details=str(exc),
            )
