#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import shutil
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from xona_dbd.config import (
    ConfigStore,
    FROZEN_FRAME_PATH,
    NEGATIVE_REFERENCES_DIR,
    POSITIVE_REFERENCES_DIR,
    REFERENCE_CAPTURE_PATH,
)

BUTTON_OPTIONS = [
    ("BUTTON_1 — RT / R2", 1),
    ("BUTTON_2 — RB / R1", 2),
    ("BUTTON_3 — Right Stick", 3),
    ("BUTTON_4 — LT / L2", 4),
    ("BUTTON_5 — LB / L1", 5),
    ("BUTTON_6 — Left Stick", 6),
    ("BUTTON_7 — Xbox / PS", 7),
    ("BUTTON_8 — View / Share", 8),
    ("BUTTON_9 — Menu / Options", 9),
    ("BUTTON_10 — D-Pad Up", 10),
    ("BUTTON_11 — D-Pad Down", 11),
    ("BUTTON_12 — D-Pad Left", 12),
    ("BUTTON_13 — D-Pad Right", 13),
    ("BUTTON_14 — Y / Triangle", 14),
    ("BUTTON_15 — B / Circle", 15),
    ("BUTTON_16 — A / Cross", 16),
    ("BUTTON_17 — X / Square", 17),
    ("BUTTON_18", 18),
    ("BUTTON_19", 19),
    ("BUTTON_20", 20),
]

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QListWidget,
    QMessageBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)



class RadiusCalibrationCanvas(QWidget):
    """
    Separate optimized calibration viewer.

    The captured image is cached. Radius edits repaint only vector overlays,
    so there is no repeated image processing or zoom loop.
    """

    def __init__(self, image_path, settings, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(780, 480)

        self.original = QPixmap(str(image_path))
        if self.original.isNull():
            raise RuntimeError(f"Could not load image: {image_path}")

        self.image_width = self.original.width()
        self.image_height = self.original.height()

        self.center_x = float(
            settings.get("calibration_center_x", self.image_width / 2.0)
        )
        self.center_y = float(
            settings.get("calibration_center_y", self.image_height / 2.0)
        )
        self.radius = float(
            settings.get("calibration_radius", settings.get("radius_expected", 89.164))
        )
        self.move_step = float(settings.get("calibration_move_step", 1.0))
        self.radius_step = float(settings.get("calibration_radius_step", 1.0))

        self.cached_size = None
        self.cached_pixmap = None
        self.draw_rect = QRectF()
        self.saved_callback = None

    def _update_cache(self):
        size = self.size()
        if self.cached_size == size and self.cached_pixmap is not None:
            return

        self.cached_size = size
        self.cached_pixmap = self.original.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (size.width() - self.cached_pixmap.width()) / 2.0
        y = (size.height() - self.cached_pixmap.height()) / 2.0
        self.draw_rect = QRectF(
            x,
            y,
            self.cached_pixmap.width(),
            self.cached_pixmap.height(),
        )

    def paintEvent(self, _event):
        self._update_cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(8, 10, 14))
        painter.drawPixmap(
            int(self.draw_rect.left()),
            int(self.draw_rect.top()),
            self.cached_pixmap,
        )

        scale_x = self.draw_rect.width() / max(1.0, self.image_width)
        scale_y = self.draw_rect.height() / max(1.0, self.image_height)
        center_x = self.draw_rect.left() + self.center_x * scale_x
        center_y = self.draw_rect.top() + self.center_y * scale_y

        painter.setPen(QPen(QColor(0, 255, 90), 3))
        painter.drawEllipse(
            QRectF(
                center_x - self.radius * scale_x,
                center_y - self.radius * scale_y,
                self.radius * scale_x * 2.0,
                self.radius * scale_y * 2.0,
            )
        )
        painter.drawLine(
            int(center_x - 10), int(center_y),
            int(center_x + 10), int(center_y),
        )
        painter.drawLine(
            int(center_x), int(center_y - 10),
            int(center_x), int(center_y + 10),
        )

        painter.fillRect(12, 12, 720, 70, QColor(0, 0, 0, 190))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(
            24,
            40,
            (
                f"Center=({self.center_x:.2f}, {self.center_y:.2f})  "
                f"Radius={self.radius:.3f}px  Diameter={self.radius * 2.0:.3f}px"
            ),
        )
        painter.setPen(QColor(150, 215, 255))
        painter.drawText(
            24,
            66,
            "WASD move | Up/+ grow | Down/- shrink | Shift x5 | Ctrl x0.25 | P save",
        )

    def keyPressEvent(self, event: QKeyEvent):
        modifiers = event.modifiers()
        move_step = self.move_step
        radius_step = self.radius_step

        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            move_step *= 5.0
            radius_step *= 5.0
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            move_step *= 0.25
            radius_step *= 0.25

        changed = False
        key = event.key()

        if key == Qt.Key.Key_W:
            self.center_y -= move_step
            changed = True
        elif key == Qt.Key.Key_S:
            self.center_y += move_step
            changed = True
        elif key == Qt.Key.Key_A:
            self.center_x -= move_step
            changed = True
        elif key == Qt.Key.Key_D:
            self.center_x += move_step
            changed = True
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.radius += radius_step
            changed = True
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.radius = max(2.0, self.radius - radius_step)
            changed = True
        elif key == Qt.Key.Key_P:
            self.save()
            return
        elif key == Qt.Key.Key_Escape:
            self.window().close()
            return

        if changed:
            self.center_x = min(max(self.center_x, 0.0), self.image_width - 1.0)
            self.center_y = min(max(self.center_y, 0.0), self.image_height - 1.0)
            self.update()

    def save(self):
        store = ConfigStore()
        latest = store.update(
            radius_reference_width=self.image_width,
            radius_reference_height=self.image_height,
            radius_expected=float(self.radius),
            calibration_radius=float(self.radius),
            calibration_center_x=float(self.center_x),
            calibration_center_y=float(self.center_y),
            calibration_capture_ready=False,
            runtime_status="Calibration saved",
            runtime_details=(
                f"radius={self.radius:.3f}px at "
                f"{self.image_width}x{self.image_height}"
            ),
        )
        if callable(self.saved_callback):
            self.saved_callback(latest)


class RadiusCalibrationDialog(QDialog):
    def __init__(self, image_path, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DbdCheck Radius Calibration")
        self.resize(1180, 760)

        layout = QVBoxLayout(self)
        info = QLabel(
            "This is a captured image from the Helios video frame. "
            "The live Video Display continues normally."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.canvas = RadiusCalibrationCanvas(image_path, settings, self)
        layout.addWidget(self.canvas, 1)

        self.status = QLabel("Press P to save.")
        layout.addWidget(self.status)

        self.canvas.saved_callback = self.saved
        QTimer.singleShot(0, self.canvas.setFocus)

    def saved(self, latest):
        self.status.setText(
            f"Saved radius {latest['radius_expected']:.3f}px"
        )



class ReferenceCropCanvas(QWidget):
    def __init__(self, image_path, settings, parent=None):
        super().__init__(parent)
        self.setMinimumSize(820, 500)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.pixmap = QPixmap(str(image_path))
        if self.pixmap.isNull():
            raise RuntimeError(f"Could not open {image_path}")

        self.image_width = self.pixmap.width()
        self.image_height = self.pixmap.height()
        self.center_x = float(
            settings.get("reference_capture_center_x", self.image_width / 2)
        )
        self.center_y = float(
            settings.get("reference_capture_center_y", self.image_height / 2)
        )
        self.radius = float(
            settings.get(
                "reference_capture_radius",
                settings.get("radius_expected", 89.164),
            )
        )

        self.cached_size = None
        self.cached_pixmap = None
        self.draw_rect = QRectF()
        self.save_callback = None

    def _cache(self):
        if self.cached_size == self.size() and self.cached_pixmap is not None:
            return
        self.cached_size = self.size()
        self.cached_pixmap = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - self.cached_pixmap.width()) / 2.0
        y = (self.height() - self.cached_pixmap.height()) / 2.0
        self.draw_rect = QRectF(
            x,
            y,
            self.cached_pixmap.width(),
            self.cached_pixmap.height(),
        )

    def _image_coordinates(self, point):
        if not self.draw_rect.contains(point):
            return None
        scale_x = self.image_width / self.draw_rect.width()
        scale_y = self.image_height / self.draw_rect.height()
        return (
            (point.x() - self.draw_rect.left()) * scale_x,
            (point.y() - self.draw_rect.top()) * scale_y,
        )

    def mousePressEvent(self, event):
        coordinates = self._image_coordinates(event.position())
        if coordinates is not None:
            self.center_x, self.center_y = coordinates
            self.update()

    def wheelEvent(self, event):
        step = 1.0 if event.angleDelta().y() > 0 else -1.0
        self.radius = max(4.0, self.radius + step)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        key = event.key()
        if key == Qt.Key.Key_W:
            self.center_y -= step
        elif key == Qt.Key.Key_S:
            self.center_y += step
        elif key == Qt.Key.Key_A:
            self.center_x -= step
        elif key == Qt.Key.Key_D:
            self.center_x += step
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.radius += step
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Minus):
            self.radius = max(4.0, self.radius - step)
        else:
            return
        self.update()

    def paintEvent(self, _event):
        self._cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(8, 10, 14))
        painter.drawPixmap(
            int(self.draw_rect.left()),
            int(self.draw_rect.top()),
            self.cached_pixmap,
        )

        scale_x = self.draw_rect.width() / self.image_width
        scale_y = self.draw_rect.height() / self.image_height
        center_x = self.draw_rect.left() + self.center_x * scale_x
        center_y = self.draw_rect.top() + self.center_y * scale_y

        painter.setPen(QPen(QColor(0, 210, 255), 3))
        painter.drawEllipse(
            QRectF(
                center_x - self.radius * scale_x,
                center_y - self.radius * scale_y,
                self.radius * scale_x * 2,
                self.radius * scale_y * 2,
            )
        )
        crop_radius = self.radius * 1.45
        painter.setPen(QPen(QColor(255, 190, 60), 2))
        painter.drawRect(
            QRectF(
                center_x - crop_radius * scale_x,
                center_y - crop_radius * scale_y,
                crop_radius * scale_x * 2,
                crop_radius * scale_y * 2,
            )
        )

        painter.fillRect(14, 14, 770, 68, QColor(0, 0, 0, 205))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(
            28,
            42,
            (
                f"Center=({self.center_x:.1f}, {self.center_y:.1f}) "
                f"Radius={self.radius:.1f}px"
            ),
        )
        painter.setPen(QColor(160, 215, 255))
        painter.drawText(
            28,
            68,
            "Click the skill-check center | Mouse wheel changes radius | WASD fine-tunes",
        )


class ReferenceCropDialog(QDialog):
    def __init__(self, settings, saved_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xona's DBD Checker — Add Reference")
        self.resize(1180, 760)
        self.saved_callback = saved_callback

        layout = QVBoxLayout(self)
        info = QLabel(
            "Click the exact center of the skill check. The cyan circle is the "
            "calibrated radius; the orange box is the crop that will be saved."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.canvas = ReferenceCropCanvas(
            REFERENCE_CAPTURE_PATH,
            settings,
            self,
        )
        layout.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        positive = QPushButton("Save Positive Skill Check")
        positive.clicked.connect(lambda: self.save("positive"))
        negative = QPushButton("Save Negative / False Object")
        negative.clicked.connect(lambda: self.save("negative"))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        row.addWidget(positive)
        row.addWidget(negative)
        row.addStretch(1)
        row.addWidget(cancel)
        layout.addLayout(row)
        QTimer.singleShot(0, self.canvas.setFocus)

    def save(self, category):
        image = cv2.imread(str(REFERENCE_CAPTURE_PATH))
        if image is None:
            QMessageBox.critical(self, "Reference Error", "Captured frame is missing.")
            return

        multiplier = 1.45
        half = max(8, int(round(self.canvas.radius * multiplier)))
        cx = int(round(self.canvas.center_x))
        cy = int(round(self.canvas.center_y))
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(image.shape[1], cx + half + 1)
        y1 = min(image.shape[0], cy + half + 1)
        crop = image[y0:y1, x0:x1]

        if crop.size == 0:
            QMessageBox.critical(self, "Reference Error", "The selected crop is empty.")
            return

        directory = (
            POSITIVE_REFERENCES_DIR
            if category == "positive"
            else NEGATIVE_REFERENCES_DIR
        )
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{category}_{int(time.time() * 1000)}.png"
        path = directory / filename

        if not cv2.imwrite(str(path), crop):
            QMessageBox.critical(self, "Reference Error", "Could not save the image.")
            return

        store = ConfigStore()
        latest = store.load(force=True)
        store.update(
            reference_capture_ready=False,
            reference_reload_request=int(
                latest.get("reference_reload_request", 0)
            ) + 1,
            runtime_status="Reference saved",
            runtime_details=str(path),
        )
        self.saved_callback()
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.store = ConfigStore()
        self.settings = self.store.load(force=True)
        self.awaiting_calibration = False
        self.calibration_request_ticks = 0
        self.calibration_dialog = None
        self.awaiting_reference_capture = False
        self.reference_capture_ticks = 0
        self.reference_dialog = None

        self.setWindowTitle("Xona's DBD Checker")
        self.resize(980, 680)
        self.setMinimumSize(860, 600)

        self.pages = QStackedWidget()
        self.sidebar = QVBoxLayout()
        self.buttons = []

        root = QWidget()
        root_layout = QHBoxLayout(root)

        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(185)
        sidebar_widget.setLayout(self.sidebar)

        for name, builder in (
            ("Dashboard", self.build_dashboard),
            ("Detection", self.build_detection),
            ("Calibration", self.build_calibration),
            ("References", self.build_references),
            ("Zones & Needle", self.build_zones),
            ("Overlay", self.build_overlay),
            ("Runtime", self.build_runtime),
            ("Advanced", self.build_advanced),
        ):
            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, index=len(self.buttons): self.select_page(index)
            )
            self.sidebar.addWidget(button)
            self.buttons.append(button)
            self.pages.addWidget(builder())

        self.sidebar.addStretch(1)
        root_layout.addWidget(sidebar_widget)
        root_layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.select_page(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_runtime)
        self.timer.start(1000)

    def select_page(self, index):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)

    def build_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("DbdCheck")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        layout.addWidget(title)

        self.master_enabled = QCheckBox("Enable detector")
        self.master_enabled.setChecked(bool(self.settings.get("enabled", True)))
        self.master_enabled.toggled.connect(
            lambda value: self.store.update(enabled=bool(value))
        )
        layout.addWidget(self.master_enabled)

        cards = QGroupBox("Live Status")
        form = QFormLayout(cards)
        self.dashboard_state = QLabel("—")
        self.dashboard_fps = QLabel("—")
        self.dashboard_process = QLabel("—")
        self.dashboard_resolution = QLabel("—")
        form.addRow("State", self.dashboard_state)
        form.addRow("Processing FPS", self.dashboard_fps)
        form.addRow("Frame time", self.dashboard_process)
        form.addRow("Resolution", self.dashboard_resolution)
        layout.addWidget(cards)
        layout.addStretch(1)
        return page

    def build_detection(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Detection")
        form = QFormLayout(group)

        self.center_only = QCheckBox(
            "Only search near the center of the screen"
        )
        self.center_only.setChecked(
            bool(self.settings.get("center_only_detection", False))
        )

        self.center_width = QDoubleSpinBox()
        self.center_width.setRange(0.15, 1.0)
        self.center_width.setDecimals(2)
        self.center_width.setSingleStep(0.05)
        self.center_width.setValue(
            float(self.settings.get("center_region_width_ratio", 0.58))
        )

        self.center_height = QDoubleSpinBox()
        self.center_height.setRange(0.15, 1.0)
        self.center_height.setDecimals(2)
        self.center_height.setSingleStep(0.05)
        self.center_height.setValue(
            float(self.settings.get("center_region_height_ratio", 0.62))
        )

        self.detector_mode = QComboBox()
        self.detector_mode.addItems(
            ["HSV + Geometry", "HSV Only", "Geometry Only", "AI", "Hybrid"]
        )
        self.detector_mode.setCurrentText(
            str(self.settings.get("detector_mode", "HSV + Geometry"))
        )

        self.acquisition_rate = QSpinBox()
        self.acquisition_rate.setRange(60, 120)
        self.acquisition_rate.setValue(
            int(self.settings.get("acquisition_rate", 20))
        )

        self.tracking_rate = QSpinBox()
        self.tracking_rate.setRange(60, 120)
        self.tracking_rate.setValue(
            int(self.settings.get("tracking_rate", 120))
        )

        self.tracking_recenter_rate = QSpinBox()
        self.tracking_recenter_rate.setRange(5, 120)
        self.tracking_recenter_rate.setValue(
            int(self.settings.get("tracking_recenter_rate", 30))
        )

        self.radius_expected = QDoubleSpinBox()
        self.radius_expected.setRange(2.0, 500.0)
        self.radius_expected.setDecimals(3)
        self.radius_expected.setValue(
            float(self.settings.get("radius_expected", 89.164))
        )

        self.radius_acquisition = QDoubleSpinBox()
        self.radius_acquisition.setRange(0.5, 80.0)
        self.radius_acquisition.setValue(
            float(self.settings.get("radius_acquisition_tolerance", 10.0))
        )

        self.radius_tracking = QDoubleSpinBox()
        self.radius_tracking.setRange(0.5, 100.0)
        self.radius_tracking.setValue(
            float(self.settings.get("radius_tracking_tolerance", 14.0))
        )

        self.search_width = QSpinBox()
        self.search_width.setRange(360, 1280)
        self.search_width.setValue(
            int(self.settings.get("search_width", 420))
        )

        self.acquisition_threshold = QDoubleSpinBox()
        self.acquisition_threshold.setRange(0.10, 0.99)
        self.acquisition_threshold.setDecimals(3)
        self.acquisition_threshold.setSingleStep(0.01)
        self.acquisition_threshold.setValue(
            float(self.settings.get("acquisition_score_threshold", 0.44))
        )

        self.tracking_threshold = QDoubleSpinBox()
        self.tracking_threshold.setRange(0.10, 0.99)
        self.tracking_threshold.setDecimals(3)
        self.tracking_threshold.setSingleStep(0.01)
        self.tracking_threshold.setValue(
            float(self.settings.get("tracking_score_threshold", 0.48))
        )

        form.addRow("Method", self.detector_mode)
        form.addRow(self.center_only)
        form.addRow("Center region width", self.center_width)
        form.addRow("Center region height", self.center_height)
        form.addRow("Acquisition updates/sec", self.acquisition_rate)
        form.addRow("Tracking validation/sec", self.tracking_rate)
        form.addRow("Tracking recenter/sec", self.tracking_recenter_rate)
        form.addRow("Expected radius", self.radius_expected)
        form.addRow("Acquisition tolerance", self.radius_acquisition)
        form.addRow("Tracking tolerance", self.radius_tracking)
        form.addRow("Search width", self.search_width)
        form.addRow("Acquisition score", self.acquisition_threshold)
        form.addRow("Tracking score", self.tracking_threshold)
        layout.addWidget(group)

        save = QPushButton("Save Detection Settings")
        save.clicked.connect(self.save_detection)
        layout.addWidget(save)
        layout.addStretch(1)
        return page

    def save_detection(self):
        self.store.update(
            detector_mode=self.detector_mode.currentText(),
            center_only_detection=self.center_only.isChecked(),
            center_region_width_ratio=self.center_width.value(),
            center_region_height_ratio=self.center_height.value(),
            acquisition_rate=self.acquisition_rate.value(),
            tracking_rate=self.tracking_rate.value(),
            tracking_recenter_rate=self.tracking_recenter_rate.value(),
            radius_expected=self.radius_expected.value(),
            radius_acquisition_tolerance=self.radius_acquisition.value(),
            radius_tracking_tolerance=self.radius_tracking.value(),
            search_width=self.search_width.value(),
            acquisition_score_threshold=self.acquisition_threshold.value(),
            tracking_score_threshold=self.tracking_threshold.value(),
        )

    def build_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Live Helios Radius Calibration")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        instructions = QLabel(
            "Capture one exact frame from the Helios video pipeline. A separate "
            "calibration window opens with that frame as a still image. The live "
            "Helios Video Display remains completely unchanged."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        controls = QGroupBox("Radius")
        form = QFormLayout(controls)

        self.cal_radius = QDoubleSpinBox()
        self.cal_radius.setRange(2.0, 500.0)
        self.cal_radius.setDecimals(3)
        self.cal_radius.setValue(
            float(self.settings.get("radius_expected", 89.164))
        )
        self.cal_radius.valueChanged.connect(self.update_calibration_radius)

        self.move_step = QDoubleSpinBox()
        self.move_step.setRange(0.1, 20.0)
        self.move_step.setValue(
            float(self.settings.get("calibration_move_step", 1.0))
        )

        self.radius_step = QDoubleSpinBox()
        self.radius_step.setRange(0.1, 20.0)
        self.radius_step.setValue(
            float(self.settings.get("calibration_radius_step", 1.0))
        )

        form.addRow("Radius", self.cal_radius)
        form.addRow("Move step", self.move_step)
        form.addRow("Radius step", self.radius_step)
        layout.addWidget(controls)

        self.calibration_status = QLabel("Calibration is inactive.")
        self.calibration_status.setWordWrap(True)
        layout.addWidget(self.calibration_status)

        button_row = QHBoxLayout()
        start = QPushButton("Capture Frame and Open Radius Editor")
        start.clicked.connect(self.start_calibration)
        stop = QPushButton("Cancel Calibration")
        stop.clicked.connect(self.cancel_calibration)
        button_row.addWidget(start)
        button_row.addWidget(stop)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page

    def update_calibration_radius(self, value):
        self.store.update(calibration_radius=float(value))

    def start_calibration(self):
        latest = self.store.load(force=True)
        request_id = int(latest.get("calibration_request_id", 0)) + 1

        self.store.update(
            calibration_move_step=self.move_step.value(),
            calibration_radius_step=self.radius_step.value(),
            calibration_radius=self.cal_radius.value(),
            calibration_request_id=request_id,
            calibration_capture_request=True,
            calibration_capture_ready=False,
            runtime_status="Calibration requested",
            runtime_details="Capturing the next exact Helios frame",
        )
        self.awaiting_calibration = True
        self.calibration_request_ticks = 0
        self.calibration_status.setText(
            "Waiting for one Helios video frame to be captured..."
        )

    def cancel_calibration(self):
        self.awaiting_calibration = False
        self.store.update(
            calibration_capture_request=False,
            calibration_capture_ready=False,
            runtime_status="Calibration cancelled",
        )
        self.calibration_status.setText("Calibration cancelled.")

    def open_calibration_dialog(self, settings):
        if not FROZEN_FRAME_PATH.exists():
            self.calibration_status.setText(
                "Capture completed, but the image file was not found."
            )
            return

        try:
            dialog = RadiusCalibrationDialog(
                FROZEN_FRAME_PATH,
                settings,
                self,
            )
            dialog.canvas.saved_callback = self.calibration_saved
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.calibration_dialog = dialog
            self.awaiting_calibration = False
            self.calibration_status.setText(
                "Radius editor opened. Press P inside it to save."
            )
        except Exception as exc:
            self.calibration_status.setText(
                f"Could not open radius editor: {exc}"
            )

    def calibration_saved(self, latest):
        self.radius_expected.setValue(
            float(latest["radius_expected"])
        )
        self.cal_radius.setValue(
            float(latest["radius_expected"])
        )
        self.calibration_status.setText(
            f"Saved radius {latest['radius_expected']:.3f}px"
        )

    def build_references(self):
        page, layout = self.page_shell(
            "Reference Library",
            "Capture real skill checks directly from Helios. Add as many as you "
            "want without sending screenshots through chat.",
        )

        group = QGroupBox("Library")
        form = QFormLayout(group)

        self.reference_positive_count = QLabel("0")
        self.reference_negative_count = QLabel("0")
        self.reference_status = QLabel("Ready")
        self.reference_status.setWordWrap(True)

        self.reference_threshold = QDoubleSpinBox()
        self.reference_threshold.setRange(0.10, 0.95)
        self.reference_threshold.setDecimals(2)
        self.reference_threshold.setSingleStep(0.02)
        self.reference_threshold.setValue(
            float(self.settings.get("reference_match_threshold", 0.29))
        )

        form.addRow("Positive references", self.reference_positive_count)
        form.addRow("Negative references", self.reference_negative_count)
        form.addRow("Match threshold", self.reference_threshold)
        form.addRow("Status", self.reference_status)
        layout.addWidget(group)

        buttons = QHBoxLayout()

        capture = QPushButton("Capture Current Helios Frame")
        capture.clicked.connect(self.capture_reference_frame)

        import_positive = QPushButton("Import Positive Crops")
        import_positive.clicked.connect(
            lambda: self.import_reference_files("positive")
        )

        import_negative = QPushButton("Import Negative Crops")
        import_negative.clicked.connect(
            lambda: self.import_reference_files("negative")
        )

        open_folder = QPushButton("Open Reference Folder")
        open_folder.clicked.connect(self.open_reference_folder)

        rebuild = QPushButton("Rebuild Cache")
        rebuild.clicked.connect(self.rebuild_reference_cache)

        save = QPushButton("Save Threshold")
        save.clicked.connect(
            lambda: self.store.update(
                reference_match_threshold=self.reference_threshold.value()
            )
        )

        for button in (
            capture,
            import_positive,
            import_negative,
            open_folder,
            rebuild,
            save,
        ):
            buttons.addWidget(button)

        layout.addLayout(buttons)

        note = QLabel(
            "Imported files should already be cropped around the skill check. "
            "For full gameplay frames, use Capture Current Helios Frame and "
            "click the skill-check center in the editor."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def capture_reference_frame(self):
        latest = self.store.load(force=True)
        request_id = int(
            latest.get("reference_capture_request_id", 0)
        ) + 1

        self.store.update(
            reference_capture_request_id=request_id,
            reference_capture_request=True,
            reference_capture_ready=False,
            runtime_status="Reference capture requested",
        )
        self.awaiting_reference_capture = True
        self.reference_capture_ticks = 0
        self.reference_status.setText("Capturing the next Helios frame…")

    def open_reference_editor(self, settings):
        if not REFERENCE_CAPTURE_PATH.exists():
            self.reference_status.setText("Captured frame file is missing.")
            self.awaiting_reference_capture = False
            return

        dialog = ReferenceCropDialog(
            settings,
            self.reference_saved,
            self,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.reference_dialog = dialog
        self.awaiting_reference_capture = False
        self.reference_status.setText("Reference editor opened.")

    def reference_saved(self):
        self.reference_status.setText("Reference saved. Cache rebuild requested.")
        self.refresh_runtime()

    def import_reference_files(self, category):
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import Cropped Reference Images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not files:
            return

        directory = (
            POSITIVE_REFERENCES_DIR
            if category == "positive"
            else NEGATIVE_REFERENCES_DIR
        )
        directory.mkdir(parents=True, exist_ok=True)

        imported = 0
        for filename in files:
            source = Path(filename)
            target = directory / (
                f"{category}_{int(time.time() * 1000)}_{imported}"
                f"{source.suffix.lower()}"
            )
            try:
                shutil.copy2(source, target)
                imported += 1
            except OSError:
                continue

        latest = self.store.load(force=True)
        self.store.update(
            reference_reload_request=int(
                latest.get("reference_reload_request", 0)
            ) + 1
        )
        self.reference_status.setText(f"Imported {imported} image(s).")

    def open_reference_folder(self):
        POSITIVE_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(POSITIVE_REFERENCES_DIR.parent))
        except AttributeError:
            self.reference_status.setText(str(POSITIVE_REFERENCES_DIR.parent))

    def rebuild_reference_cache(self):
        latest = self.store.load(force=True)
        self.store.update(
            reference_reload_request=int(
                latest.get("reference_reload_request", 0)
            ) + 1
        )
        self.reference_status.setText("Cache rebuild requested.")

    def build_zones(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        needle_group = QGroupBox("Needle Detection")
        needle_form = QFormLayout(needle_group)

        self.analysis_rate = QSpinBox()
        self.analysis_rate.setRange(60, 120)
        self.analysis_rate.setValue(
            int(self.settings.get("analysis_rate", 60))
        )

        self.needle_min_pixels = QSpinBox()
        self.needle_min_pixels.setRange(1, 100)
        self.needle_min_pixels.setValue(
            int(self.settings.get("needle_min_pixels", 5))
        )

        self.needle_span = QDoubleSpinBox()
        self.needle_span.setRange(0.05, 1.20)
        self.needle_span.setDecimals(3)
        self.needle_span.setSingleStep(0.01)
        self.needle_span.setValue(
            float(self.settings.get("needle_min_span_ratio", 0.30))
        )

        self.needle_confidence = QDoubleSpinBox()
        self.needle_confidence.setRange(0.01, 1.0)
        self.needle_confidence.setDecimals(2)
        self.needle_confidence.setSingleStep(0.02)
        self.needle_confidence.setValue(
            float(self.settings.get("needle_peak_min_score", 0.18))
        )

        self.needle_hold = QSpinBox()
        self.needle_hold.setRange(0, 30)
        self.needle_hold.setValue(
            int(self.settings.get("needle_hold_frames", 5))
        )

        self.needle_smoothing = QDoubleSpinBox()
        self.needle_smoothing.setRange(0.01, 1.0)
        self.needle_smoothing.setDecimals(2)
        self.needle_smoothing.setSingleStep(0.05)
        self.needle_smoothing.setValue(
            float(self.settings.get("needle_angle_smoothing", 0.34))
        )

        needle_form.addRow("Analysis updates/sec", self.analysis_rate)
        needle_form.addRow("Minimum red pixels", self.needle_min_pixels)
        needle_form.addRow("Minimum needle span", self.needle_span)
        needle_form.addRow("Minimum confidence", self.needle_confidence)
        needle_form.addRow("Hold missing frames", self.needle_hold)
        needle_form.addRow("Angle smoothing", self.needle_smoothing)
        layout.addWidget(needle_group)

        zone_group = QGroupBox("Good / Great Zone Analysis")
        zone_form = QFormLayout(zone_group)

        self.zone_multiplier = QDoubleSpinBox()
        self.zone_multiplier.setRange(1.0, 4.0)
        self.zone_multiplier.setDecimals(2)
        self.zone_multiplier.setSingleStep(0.05)
        self.zone_multiplier.setValue(
            float(self.settings.get("zone_thickness_multiplier", 1.16))
        )

        self.great_multiplier = QDoubleSpinBox()
        self.great_multiplier.setRange(1.0, 5.0)
        self.great_multiplier.setDecimals(2)
        self.great_multiplier.setSingleStep(0.05)
        self.great_multiplier.setValue(
            float(self.settings.get("great_thickness_multiplier", 1.42))
        )

        self.hit_lead = QDoubleSpinBox()
        self.hit_lead.setRange(0.0, 20.0)
        self.hit_lead.setDecimals(2)
        self.hit_lead.setSingleStep(0.25)
        self.hit_lead.setValue(
            float(self.settings.get("hit_lead_degrees", 2.0))
        )

        zone_form.addRow("Good thickness multiplier", self.zone_multiplier)
        zone_form.addRow("Great thickness multiplier", self.great_multiplier)
        zone_form.addRow("Planned hit lead", self.hit_lead)
        layout.addWidget(zone_group)

        press_group = QGroupBox("Automatic Press")
        press_form = QFormLayout(press_group)

        self.simulate_press = QCheckBox("Show simulated press flash")
        self.simulate_press.setChecked(
            bool(self.settings.get("simulate_press_enabled", True))
        )

        self.auto_press = QCheckBox("Enable actual automatic skill-check press")
        self.auto_press.setChecked(
            bool(self.settings.get("auto_press_enabled", False))
        )

        self.press_button = QComboBox()
        for label, button_index in BUTTON_OPTIONS:
            self.press_button.addItem(label, button_index)

        configured_button = int(
            self.settings.get("press_button_index", 5)
        )
        configured_index = self.press_button.findData(configured_button)
        if configured_index >= 0:
            self.press_button.setCurrentIndex(configured_index)

        self.press_window = QDoubleSpinBox()
        self.press_window.setRange(0.25, 20.0)
        self.press_window.setDecimals(2)
        self.press_window.setSingleStep(0.25)
        self.press_window.setValue(
            float(self.settings.get("press_window_degrees", 3.0))
        )

        self.press_frames = QSpinBox()
        self.press_frames.setRange(1, 10)
        self.press_frames.setValue(
            int(self.settings.get("press_duration_frames", 2))
        )

        press_form.addRow(self.simulate_press)
        press_form.addRow(self.auto_press)
        press_form.addRow("Skill-check bind", self.press_button)
        press_form.addRow("Trigger window", self.press_window)
        press_form.addRow("Hold frames", self.press_frames)
        layout.addWidget(press_group)

        save = QPushButton("Save Zone & Needle Settings")
        save.clicked.connect(self.save_zones)
        layout.addWidget(save)
        layout.addStretch(1)
        return page

    def save_zones(self):
        self.store.update(
            analysis_rate=self.analysis_rate.value(),
            needle_min_pixels=self.needle_min_pixels.value(),
            needle_min_span_ratio=self.needle_span.value(),
            needle_peak_min_score=self.needle_confidence.value(),
            needle_hold_frames=self.needle_hold.value(),
            needle_angle_smoothing=self.needle_smoothing.value(),
            zone_thickness_multiplier=self.zone_multiplier.value(),
            great_thickness_multiplier=self.great_multiplier.value(),
            hit_lead_degrees=self.hit_lead.value(),
            simulate_press_enabled=self.simulate_press.isChecked(),
            auto_press_enabled=self.auto_press.isChecked(),
            press_button_index=int(self.press_button.currentData()),
            press_window_degrees=self.press_window.value(),
            press_duration_frames=self.press_frames.value(),
        )

    def build_overlay(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Video Display Overlay")
        form = QFormLayout(group)

        self.overlay_enabled = QCheckBox("Draw overlay")
        self.overlay_enabled.setChecked(
            bool(self.settings.get("overlay_enabled", True))
        )

        self.draw_needle = QCheckBox("Red needle line")
        self.draw_needle.setChecked(
            bool(self.settings.get("draw_needle_line", True))
        )

        self.draw_good = QCheckBox("Cyan Good-zone line")
        self.draw_good.setChecked(
            bool(self.settings.get("draw_good_line", True))
        )

        self.draw_great = QCheckBox("Magenta Great-zone line")
        self.draw_great.setChecked(
            bool(self.settings.get("draw_great_line", True))
        )

        self.draw_arcs = QCheckBox("Draw Good/Great zone arcs")
        self.draw_arcs.setChecked(
            bool(self.settings.get("draw_zone_arcs", True))
        )

        self.draw_confidence = QCheckBox("Draw analysis confidence")
        self.draw_confidence.setChecked(
            bool(self.settings.get("draw_analysis_confidence", True))
        )

        self.draw_press = QCheckBox("Blue planned-press line and dot")
        self.draw_press.setChecked(
            bool(self.settings.get("draw_press_line", True))
        )

        form.addRow(self.overlay_enabled)
        form.addRow(self.draw_needle)
        form.addRow(self.draw_good)
        form.addRow(self.draw_great)
        form.addRow(self.draw_arcs)
        form.addRow(self.draw_confidence)
        form.addRow(self.draw_press)
        layout.addWidget(group)

        save = QPushButton("Save Overlay Settings")
        save.clicked.connect(self.save_overlay)
        layout.addWidget(save)
        layout.addStretch(1)
        return page

    def save_overlay(self):
        self.store.update(
            overlay_enabled=self.overlay_enabled.isChecked(),
            draw_needle_line=self.draw_needle.isChecked(),
            draw_good_line=self.draw_good.isChecked(),
            draw_great_line=self.draw_great.isChecked(),
            draw_zone_arcs=self.draw_arcs.isChecked(),
            draw_analysis_confidence=self.draw_confidence.isChecked(),
            draw_press_line=self.draw_press.isChecked(),
        )

    def build_runtime(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Runtime Diagnostics")
        form = QFormLayout(group)
        self.runtime_status = QLabel("—")
        self.runtime_details = QLabel("—")
        self.runtime_details.setWordWrap(True)
        self.runtime_state = QLabel("—")
        form.addRow("Status", self.runtime_status)
        form.addRow("Details", self.runtime_details)
        form.addRow("State", self.runtime_state)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def build_advanced(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        note = QLabel(
            "During a visible skill check, Runtime should show candidates and "
            "edgeRing/edgePeak values. If candidates remain 0, template matching "
            "is not producing peaks. Orange rings can be enabled temporarily."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        group = QGroupBox("Acquisition Debug")
        form = QFormLayout(group)

        self.template_peak = QDoubleSpinBox()
        self.template_peak.setRange(0.05, 0.95)
        self.template_peak.setDecimals(3)
        self.template_peak.setSingleStep(0.01)
        self.template_peak.setValue(
            float(self.settings.get("template_peak_threshold", 0.20))
        )

        self.max_candidates = QSpinBox()
        self.max_candidates.setRange(1, 40)
        self.max_candidates.setValue(
            int(self.settings.get("max_acquisition_candidates", 18))
        )

        self.confirm_frames = QSpinBox()
        self.confirm_frames.setRange(1, 10)
        self.confirm_frames.setValue(
            int(self.settings.get("lock_confirmation_frames", 2))
        )

        self.lost_frames = QSpinBox()
        self.lost_frames.setRange(1, 120)
        self.lost_frames.setValue(
            int(self.settings.get("tracker_lost_frames", 18))
        )

        self.edge_low = QSpinBox()
        self.edge_low.setRange(1, 250)
        self.edge_low.setValue(
            int(self.settings.get("edge_low_threshold", 35))
        )

        self.edge_high = QSpinBox()
        self.edge_high.setRange(2, 255)
        self.edge_high.setValue(
            int(self.settings.get("edge_high_threshold", 110))
        )

        self.edge_weight = QDoubleSpinBox()
        self.edge_weight.setRange(0.0, 1.0)
        self.edge_weight.setDecimals(2)
        self.edge_weight.setSingleStep(0.05)
        self.edge_weight.setValue(
            float(self.settings.get("edge_template_weight", 0.58))
        )

        self.geometry_accept = QDoubleSpinBox()
        self.geometry_accept.setRange(0.10, 0.95)
        self.geometry_accept.setDecimals(3)
        self.geometry_accept.setSingleStep(0.01)
        self.geometry_accept.setValue(
            float(self.settings.get("geometry_accept_threshold", 0.30))
        )

        self.hough_fallback = QCheckBox("Enable Hough fallback (expensive)")
        self.hough_fallback.setChecked(
            bool(self.settings.get("hough_fallback_enabled", True))
        )

        self.hough_rate = QSpinBox()
        self.hough_rate.setRange(1, 60)
        self.hough_rate.setValue(
            int(self.settings.get("hough_fallback_rate", 15))
        )

        self.require_needle = QCheckBox(
            "Require red needle before locking"
        )
        self.require_needle.setChecked(
            bool(self.settings.get("acquisition_require_needle", True))
        )

        self.minimum_red_score = QDoubleSpinBox()
        self.minimum_red_score.setRange(0.05, 1.0)
        self.minimum_red_score.setDecimals(2)
        self.minimum_red_score.setSingleStep(0.02)
        self.minimum_red_score.setValue(
            float(self.settings.get("acquisition_min_red_score", 0.46))
        )

        self.minimum_needle_line = QDoubleSpinBox()
        self.minimum_needle_line.setRange(0.10, 1.0)
        self.minimum_needle_line.setDecimals(2)
        self.minimum_needle_line.setSingleStep(0.02)
        self.minimum_needle_line.setValue(
            float(
                self.settings.get(
                    "acquisition_min_needle_line_score",
                    0.58,
                )
            )
        )

        self.minimum_white_ring = QDoubleSpinBox()
        self.minimum_white_ring.setRange(0.0, 1.0)
        self.minimum_white_ring.setDecimals(2)
        self.minimum_white_ring.setSingleStep(0.02)
        self.minimum_white_ring.setValue(
            float(self.settings.get("acquisition_min_white_ring", 0.10))
        )

        self.minimum_prompt = QDoubleSpinBox()
        self.minimum_prompt.setRange(0.0, 1.0)
        self.minimum_prompt.setDecimals(2)
        self.minimum_prompt.setSingleStep(0.02)
        self.minimum_prompt.setValue(
            float(
                self.settings.get(
                    "acquisition_min_center_prompt_score",
                    0.18,
                )
            )
        )

        self.draw_candidates = QCheckBox("Draw orange acquisition candidates")
        self.draw_candidates.setChecked(
            bool(self.settings.get("draw_acquisition_candidates", False))
        )

        form.addRow("Template peak", self.template_peak)
        form.addRow("Maximum candidates", self.max_candidates)
        form.addRow("Lock confirmation frames", self.confirm_frames)
        form.addRow("Tracker lost frames", self.lost_frames)
        form.addRow("Canny low", self.edge_low)
        form.addRow("Canny high", self.edge_high)
        form.addRow("Edge template weight", self.edge_weight)
        form.addRow("Geometry accept score", self.geometry_accept)
        form.addRow(self.hough_fallback)
        form.addRow("Fallback checks/sec", self.hough_rate)
        form.addRow(self.require_needle)
        form.addRow("Minimum red coverage", self.minimum_red_score)
        form.addRow("Minimum needle-line score", self.minimum_needle_line)
        form.addRow("Minimum white-ring coverage", self.minimum_white_ring)
        form.addRow("Minimum center-prompt score", self.minimum_prompt)
        form.addRow(self.draw_candidates)
        layout.addWidget(group)

        save = QPushButton("Save Advanced Settings")
        save.clicked.connect(self.save_advanced)
        layout.addWidget(save)
        layout.addStretch(1)
        return page

    def save_advanced(self):
        self.store.update(
            template_peak_threshold=self.template_peak.value(),
            max_acquisition_candidates=self.max_candidates.value(),
            lock_confirmation_frames=self.confirm_frames.value(),
            tracker_lost_frames=self.lost_frames.value(),
            edge_low_threshold=self.edge_low.value(),
            edge_high_threshold=self.edge_high.value(),
            edge_template_weight=self.edge_weight.value(),
            white_template_weight=max(0.0, 1.0 - self.edge_weight.value()),
            geometry_accept_threshold=self.geometry_accept.value(),
            hough_fallback_enabled=self.hough_fallback.isChecked(),
            hough_fallback_rate=self.hough_rate.value(),
            acquisition_require_needle=self.require_needle.isChecked(),
            acquisition_min_red_score=self.minimum_red_score.value(),
            acquisition_min_needle_line_score=self.minimum_needle_line.value(),
            acquisition_min_white_ring=self.minimum_white_ring.value(),
            acquisition_min_center_prompt_score=self.minimum_prompt.value(),
            draw_acquisition_candidates=self.draw_candidates.isChecked(),
        )

    def refresh_runtime(self):
        settings = self.store.load(force=True)
        self.dashboard_state.setText(str(settings.get("runtime_state", "—")))
        self.dashboard_fps.setText(
            f"{float(settings.get('runtime_fps', 0.0)):.1f}"
        )
        self.dashboard_process.setText(
            f"{float(settings.get('runtime_ms', 0.0)):.2f} ms"
        )
        width = int(settings.get("runtime_width", 0))
        height = int(settings.get("runtime_height", 0))
        self.dashboard_resolution.setText(
            f"{width} × {height}" if width and height else "—"
        )
        self.runtime_status.setText(str(settings.get("runtime_status", "—")))
        self.runtime_details.setText(str(settings.get("runtime_details", "—")))
        self.runtime_state.setText(str(settings.get("runtime_state", "—")))

        if hasattr(self, "reference_positive_count"):
            self.reference_positive_count.setText(
                str(settings.get("reference_positive_count", 0))
            )
            self.reference_negative_count.setText(
                str(settings.get("reference_negative_count", 0))
            )

            if self.awaiting_reference_capture:
                self.reference_capture_ticks += 1
                if settings.get("reference_capture_ready", False):
                    self.open_reference_editor(settings)
                elif self.reference_capture_ticks >= 12:
                    self.awaiting_reference_capture = False
                    self.reference_status.setText(
                        "No frame was captured after six seconds."
                    )

        if hasattr(self, "calibration_status"):
            if self.awaiting_calibration:
                self.calibration_request_ticks += 1
                if settings.get("calibration_capture_ready", False):
                    self.open_calibration_dialog(settings)
                elif self.calibration_request_ticks >= 12:
                    self.awaiting_calibration = False
                    self.calibration_status.setText(
                        "No frame was captured after 6 seconds. Confirm "
                        "XonasDBDChecker.py is actively running in Creative, then try again."
                    )
            elif settings.get("runtime_status") == "Calibration saved":
                self.calibration_status.setText(
                    f"Saved radius: "
                    f"{float(settings.get('radius_expected', 0.0)):.3f}px"
                )



DARK_STYLE = """
QWidget {
    background-color: #111318;
    color: #e8eaf0;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow {
    background-color: #0d0f14;
}
QGroupBox {
    border: 1px solid #303641;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    font-weight: 600;
    background-color: #171a21;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #242936;
    border: 1px solid #3b4352;
    border-radius: 7px;
    padding: 9px 12px;
}
QPushButton:hover {
    background-color: #303747;
}
QPushButton:pressed,
QPushButton:checked {
    background-color: #3867d6;
    border-color: #5b83e3;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1c2029;
    border: 1px solid #3b4352;
    border-radius: 6px;
    padding: 6px;
    min-height: 24px;
}
QCheckBox {
    spacing: 8px;
}
QLabel {
    background: transparent;
}
"""

def main():
    if "--standalone-ui" not in sys.argv:
        print(
            "[DbdCheck UI] This file is not a Creative script. "
            "Load XonasDBDChecker.py only."
        )
        return 0

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
