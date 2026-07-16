#!/usr/bin/env python3
"""
Xona's DBD Checker — Helios Creative entrypoint.

Load ONLY this file in AimEngine Creative.
Do not load dbdcheck_ui.py as a Creative script.
"""

from __future__ import annotations

import sys

import cv2

# Helios already owns the high-frequency frame worker. Prevent OpenCV from
# spawning its own thread pool and stalling the worker through oversubscription.
cv2.setUseOptimized(True)
cv2.setNumThreads(1)
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gtuner import *
from creative_helper import Combo
from xona_dbd.creative_runtime import CreativeRuntime


_runtime = None


def _get_runtime(frame):
    global _runtime
    if _runtime is None:
        height, width = frame.shape[:2]
        _runtime = CreativeRuntime(width, height, SCRIPT_DIR)
        print(
            f"[Xona's DBD Checker] Runtime loaded at {width}x{height}"
        )
    return _runtime


def scan(button_bytes, stick_bytes, **kwargs):
    return


def iterate(button_bytes, stick_bytes, **kwargs):
    combo = Combo()
    combo.buttons = button_bytes
    combo.sticks = stick_bytes

    frame = kwargs.get("frame")
    if frame is None:
        return combo.buttons, combo.sticks

    runtime = _get_runtime(frame)
    processed_frame, cvdata = runtime.process(frame)

    settings = runtime.config.load()
    press_consumer = getattr(runtime, "consume_press_request", None)
    if callable(press_consumer):
        press = press_consumer(settings)
        if press is not None:
            button_index, value = press
            combo.set_val(button_index, value)

    return combo.buttons, combo.sticks, processed_frame
