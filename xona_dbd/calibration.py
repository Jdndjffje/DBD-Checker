from __future__ import annotations

import cv2

from .config import FROZEN_FRAME_PATH


class CalibrationController:
    """
    Captures one exact Helios frame for the separate calibration window.

    The live Helios Video Display is never frozen or replaced. The requested
    frame is copied to disk once, then normal processing continues.
    """

    def __init__(self, config_store):
        self.config_store = config_store
        self.last_request_id = None

    def process(self, live_frame, settings):
        request_id = settings.get("calibration_request_id")
        requested = bool(settings.get("calibration_capture_request", False))

        if not requested:
            self.last_request_id = request_id
            return live_frame, settings

        self.last_request_id = request_id

        try:
            FROZEN_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(FROZEN_FRAME_PATH), live_frame)
            if not success:
                raise RuntimeError("cv2.imwrite returned False")

            settings = self.config_store.update(
                calibration_capture_request=False,
                calibration_capture_ready=True,
                calibration_center_x=live_frame.shape[1] / 2.0,
                calibration_center_y=live_frame.shape[0] / 2.0,
                calibration_radius=float(
                    settings.get("radius_expected", 89.164)
                ),
                runtime_status="Calibration frame captured",
                runtime_details=(
                    f"Saved exact Helios frame "
                    f"{live_frame.shape[1]}x{live_frame.shape[0]}"
                ),
            )
            print(
                f"[Xona's DBD Checker] Calibration frame captured: "
                f"{live_frame.shape[1]}x{live_frame.shape[0]}"
            )
        except Exception as exc:
            settings = self.config_store.update(
                calibration_capture_request=False,
                calibration_capture_ready=False,
                runtime_status="Calibration capture failed",
                runtime_details=str(exc),
            )
            print(f"[Xona's DBD Checker] Calibration capture failed: {exc}")

        return live_frame, settings
