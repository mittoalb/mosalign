#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosalign - Motor Scan Plugin for PyStream
------------------------------------------
Performs 2D motor scan with live stitched preview.

Features:
- X-Y motor grid scan with image stitching
- Optional second preview at another angle (90°: Z × Y grid; 180°: X × Y,
  for double field of view); tiles are edited in both mosaics
- Tomoscan integration at each position
- Test mode for offline development
"""

import logging
import math
import os
import subprocess
import time
import numpy as np
from typing import Optional
from PyQt5 import QtWidgets, QtCore, QtGui

import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')

# Import works both when run as a package (`mosalign.gui`) and when the
# pystream launcher invokes this file as a script (no parent package).
try:
    from .tiles import TileManager, TileRunWorker, TileSidePanel
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tiles import TileManager, TileRunWorker, TileSidePanel


class LiveCameraThread(QtCore.QThread):
    """Reads the Image PV in the background: reports the connection status
    (always) and, when `want_frames` is set, sends display-size frames for
    the Live view. Its own pvaccess channel, so it never blocks the GUI or
    the scan workers."""
    status = QtCore.pyqtSignal(bool, str)        # connected, text
    frame = QtCore.pyqtSignal(object)            # 2-D numpy array (display size)

    def __init__(self, pv: str, parent=None):
        super().__init__(parent)
        self.pv = pv
        self.want_frames = False
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import pvaccess as pva
        except ImportError:
            self.status.emit(False, "Camera: pvaccess not installed")
            return
        ch, last_uid, rate = None, None, None
        uids = []                                  # (time, uid) for the frame rate
        next_status = 0.0
        while not self._stop:
            try:
                if ch is None:
                    ch = pva.Channel(self.pv)
                uid = int(ch.get("field(uniqueId)")["uniqueId"])
                now = time.monotonic()
                uids.append((now, uid))
                uids = [(t, u) for t, u in uids if now - t < 5.0]
                span = uids[-1][0] - uids[0][0]
                rate = (uids[-1][1] - uids[0][1]) / span if span >= 1.0 else None
                if self.want_frames and uid != last_uid:
                    st = ch.get("")
                    img = MotorScanDialog._ntnda_to_array(st)
                    h, w = img.shape
                    step = max(1, int(np.ceil(max(h, w) / 1200)))
                    self.frame.emit(np.ascontiguousarray(img[::step, ::step]))
                    self._shape = (w, h)
                    last_uid = uid
                if now >= next_status:
                    shape = getattr(self, "_shape", None)
                    size = f", {shape[0]}×{shape[1]}" if shape else ""
                    if rate is None:
                        fps = ""                    # not measured yet
                    elif rate == 0:
                        fps = ", no new frames (camera not acquiring?)"
                    else:
                        fps = f", {rate:.1f} fps"
                    self.status.emit(True, f"Camera: connected{size}{fps}")
                    next_status = now + 2.0
            except Exception as e:
                ch = None
                self.status.emit(False, f"Camera: no connection to {self.pv} "
                                        f"({type(e).__name__}: {str(e)[:80]})")
                for _ in range(20):                # retry every 2 s
                    if self._stop:
                        return
                    time.sleep(0.1)
                continue
            time.sleep(0.25 if self.want_frames else 0.5)


class MotorScanDialog(QtWidgets.QDialog):
    """Dialog for configuring and running 2D motor scans"""

    BUTTON_TEXT = "Mosalign"
    HANDLER_TYPE = 'singleton'  # Keep one instance, show/hide it

    def __init__(self, parent=None, logger: Optional[logging.Logger] = None):
        super().__init__(parent)
        self.logger = logger
        self.scanning = False
        self.scan_thread = None
        self.current_image = None

        # For stitched preview (stitched_image_b: second-angle mosaic)
        self.stitched_image = None
        self.stitched_image_b = None
        self.stitched_lock = QtCore.QMutex()

        # Interactive tiles (created after view_box exists in _build_ui)
        self.tile_manager: Optional[TileManager] = None
        self.tile_run_worker: Optional[TileRunWorker] = None
        # Last grid-scan geometry, stashed by ScanWorker for seed_from_grid
        self._last_scan_geom: Optional[dict] = None
        self._last_scan_geom_b: Optional[dict] = None

        # Test/Mock mode
        self.test_mode = False

        # Frame capture state (see _get_image_now)
        self._pva_channels: dict = {}
        self._last_capture_error = ""

        # Timer to refresh preview during scan
        self.preview_timer = QtCore.QTimer()
        self.preview_timer.timeout.connect(self._refresh_stitched_preview)

        self.setWindowTitle("Mosalign")
        self.resize(1200, 800)

        self.live_thread: Optional[LiveCameraThread] = None
        self._live_levels = None

        self._build_ui()
        self._load_config()
        self._update_angle_info()

        self._on_preview_toggled(self.enable_preview.isChecked())
        self.image_pv.editingFinished.connect(self._restart_camera)
        self._restart_camera()
        self._set_view_mode(live=True)

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # ── Left panel: settings tabs only (retractable) ───────────────
        # Scan = operational, PVs = IOC config. Start/Stop/progress live in
        # the toolbar above the image so they stay visible when this panel
        # is hidden.
        self.left_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.left_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self._build_scan_tab(), "Scan")
        self.left_tabs.addTab(self._build_pv_tab(),   "PVs")
        layout.addWidget(self.left_tabs, 1)

        main_layout.addWidget(self.left_panel, 1)

        # Right panel - Image preview & log
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # ── Toolbar: panel toggles + all actions ───────────────────────
        tile_btns = QtWidgets.QHBoxLayout()

        self.toggle_left_btn = QtWidgets.QToolButton()
        self.toggle_left_btn.setCheckable(True)
        self.toggle_left_btn.setChecked(True)
        self.toggle_left_btn.setToolTip("Show/hide the settings panel")
        self.toggle_left_btn.toggled.connect(self._toggle_left_panel)
        tile_btns.addWidget(self.toggle_left_btn)

        self.start_btn = QtWidgets.QPushButton("Start Scan")
        self.start_btn.setAutoDefault(False)
        self.start_btn.setDefault(False)
        self.start_btn.clicked.connect(self._start_scan)
        tile_btns.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setAutoDefault(False)
        self.stop_btn.setDefault(False)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_scan)
        tile_btns.addWidget(self.stop_btn)

        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setMinimumSize(300, 400)  # Ensure preview is visible
        self.title_a = self.graphics_view.addLabel("", row=0, col=0)
        self.view_box = self.graphics_view.addViewBox(row=1, col=0)
        self.view_box.setAspectLocked(True)
        self.image_view = pg.ImageItem()
        self.view_box.addItem(self.image_view)

        # Second-angle mosaic (90° / 180°), shown next to the first one
        self.graphics_view_b = pg.GraphicsLayoutWidget()
        self.graphics_view_b.setMinimumSize(300, 400)
        self.title_b = self.graphics_view_b.addLabel("", row=0, col=0)
        self.view_box_b = self.graphics_view_b.addViewBox(row=1, col=0)
        self.view_box_b.setAspectLocked(True)
        self.image_view_b = pg.ImageItem()
        self.view_box_b.addItem(self.image_view_b)
        self.graphics_view_b.setVisible(False)

        # Tile buttons stay disabled until a grid scan completes (and
        # _scan_finished seeds the tile manager).
        self.add_tile_btn = QtWidgets.QPushButton("+ Tile at Center")
        self.add_tile_btn.setAutoDefault(False)
        self.add_tile_btn.setDefault(False)
        self.add_tile_btn.setEnabled(False)
        self.add_tile_btn.clicked.connect(self._add_tile_at_center)
        tile_btns.addWidget(self.add_tile_btn)

        self.run_tiles_btn = QtWidgets.QPushButton("Run Tiles")
        self.run_tiles_btn.setAutoDefault(False)
        self.run_tiles_btn.setDefault(False)
        self.run_tiles_btn.setEnabled(False)
        self.run_tiles_btn.clicked.connect(self._run_tiles)
        tile_btns.addWidget(self.run_tiles_btn)

        self.view_live = QtWidgets.QRadioButton("Live")
        self.view_live.setToolTip("Current camera image (Image PV) — to find and focus the sample")
        self.view_mosaic = QtWidgets.QRadioButton("Mosaic")
        self.view_mosaic.setToolTip("Stitched grid-scan mosaic with the tiles")
        self._view_group = QtWidgets.QButtonGroup(self)
        self._view_group.addButton(self.view_live)
        self._view_group.addButton(self.view_mosaic)
        self.view_live.toggled.connect(lambda on: self._set_view_mode(live=on))
        tile_btns.addWidget(self.view_live)
        tile_btns.addWidget(self.view_mosaic)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        tile_btns.addWidget(self.progress)
        self.position_label = QtWidgets.QLabel("Position: -")
        tile_btns.addWidget(self.position_label)
        tile_btns.addStretch()

        self.toggle_right_btn = QtWidgets.QToolButton()
        self.toggle_right_btn.setCheckable(True)
        self.toggle_right_btn.setChecked(True)
        self.toggle_right_btn.setToolTip("Show/hide the tile panel")
        self.toggle_right_btn.toggled.connect(self._toggle_tile_panel)
        tile_btns.addWidget(self.toggle_right_btn)
        right_layout.addLayout(tile_btns)

        # Tile manager owns ROIs on view_box; created here so it's ready for
        # config-load-time restore_rois() calls.
        self.tile_manager = TileManager(self.view_box, self)
        # The same tiles on the second-angle mosaic: dragging there sets Z
        # (90°) or X (180°); Y is shared. Each view follows the other.
        self.tile_manager_b = TileManager(self.view_box_b, self, h_attr="z_mm",
                                          primary=self.tile_manager)
        self.tile_manager.link(self.tile_manager_b)
        self.tile_manager_b.link(self.tile_manager)

        # Canvas + side panel in a resizable horizontal splitter.
        # Side panel shows the currently-selected tile's motor X/Y +
        # NumAngles / ExposureTime overrides; double-click a tile to edit.
        self.tile_side_panel = TileSidePanel(self.tile_manager, parent=self)
        self.tile_side_panel.add_view(self.tile_manager_b)
        views = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        views.addWidget(self.graphics_view)
        views.addWidget(self.graphics_view_b)
        canvas_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        canvas_splitter.addWidget(views)
        canvas_splitter.addWidget(self.tile_side_panel)
        canvas_splitter.setStretchFactor(0, 1)
        canvas_splitter.setStretchFactor(1, 0)
        # Side panel needs room for PV-name labels like "32idcSOFT:m17:";
        # 320 px keeps both columns of the form layout visible.
        canvas_splitter.setSizes([760, 320])
        right_layout.addWidget(canvas_splitter, 2)

        # Log output with terminal styling
        right_layout.addWidget(QtWidgets.QLabel("Scan Log / Console"))

        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        # Terminal-style console like xanes_gui
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                font-family: 'Courier New', 'Monospace', monospace;
                font-size: 10pt;
                padding: 5px;
            }
        """)
        right_layout.addWidget(self.log_output, 1)

        main_layout.addWidget(right_panel, 2)
        self._toggle_left_panel(True)
        self._toggle_tile_panel(True)

    def _toggle_left_panel(self, shown: bool):
        self.left_panel.setVisible(shown)
        self.toggle_left_btn.setText("◀ Settings" if shown else "Settings ▶")

    def _toggle_tile_panel(self, shown: bool):
        self.tile_side_panel.setVisible(shown)
        self.toggle_right_btn.setText("Tiles ▶" if shown else "◀ Tiles")

    # ── live camera / status ────────────────────────────────────────────

    def _restart_camera(self):
        """(Re)start the background camera reader for the current Image PV."""
        if self.live_thread is not None:
            self.live_thread.stop()
            self.live_thread.wait(3000)
            self.live_thread = None
        pv = self.image_pv.text().strip()
        if not pv:
            self.connection_status.setText("Camera: no Image PV set")
            return
        self.connection_status.setText(f"Camera: connecting to {pv}…")
        t = LiveCameraThread(pv, self)
        t.status.connect(self._on_camera_status)
        t.frame.connect(self._on_live_frame)
        t.want_frames = self._live_mode()
        t.start()
        self.live_thread = t

    def _on_camera_status(self, ok: bool, text: str):
        if self.test_mode:
            return                                  # test mode shows its own status
        self.connection_status.setText(text)
        self.connection_status.setStyleSheet("" if ok else "color: #d62728;")

    def _live_mode(self) -> bool:
        return hasattr(self, "view_live") and self.view_live.isChecked()

    def _set_view_mode(self, live: bool):
        """Live: camera frames, tiles hidden. Mosaic: stitched preview + tiles."""
        for btn, on in ((self.view_live, live), (self.view_mosaic, not live)):
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)
        if self.live_thread is not None:
            self.live_thread.want_frames = live
        if self.tile_manager is not None:
            for roi in self.tile_manager._rois.values():
                roi.setVisible(not live)
            for lbl in getattr(self.tile_manager, "_labels", {}).values():
                lbl.setVisible(not live)
        self._live_levels = None
        self._live_autorange = True
        self._update_view_titles(live)
        if live:
            self.image_view.clear()
        else:
            self._preview_initialized_view = False
            for attr in ("_preview_initialized", "_preview_initialized_b"):
                if hasattr(self, attr):
                    delattr(self, attr)
            self._refresh_stitched_preview()

    def _has_view_b(self) -> bool:
        return (self.stitched_image_b is not None
                or (self.tile_manager_b is not None and bool(self.tile_manager_b.is_active())))

    def _update_view_titles(self, live: Optional[bool] = None):
        if live is None:
            live = self._live_mode()
        show_b = (not live) and self._has_view_b()
        self.graphics_view_b.setVisible(show_b)
        if live:
            self.title_a.setText("Live camera")
            return
        ga = self._last_scan_geom or {}
        gb = self._last_scan_geom_b or {}
        if show_b:
            self.title_a.setText(f"{ga.get('angle', self.angle_a.value()):g}°  —  X × Y")
            attr = gb.get("h_attr", self.tile_manager_b.h_attr)
            sign = gb.get("h_sign", self.tile_manager_b.h_sign)
            motor = "Z" if attr == "z_mm" else "X"
            note = "  (reversed)" if sign < 0 else ""
            self.title_b.setText(f"{gb.get('angle', self.angle_b.value()):g}°  —  "
                                 f"{motor} × Y{note}")
        else:
            self.title_a.setText("Mosaic")

    def _on_live_frame(self, img):
        if not self._live_mode():
            return
        disp = np.flipud(img)                       # same orientation as in the mosaic
        if self.auto_contrast.isChecked() or self._live_levels is None:
            lo, hi = np.percentile(disp, (0.5, 99.5))
            self._live_levels = (float(lo), float(hi if hi > lo else lo + 1))
        self.image_view.setImage(disp, levels=self._live_levels)
        if getattr(self, "_live_autorange", False):
            self.view_box.autoRange()
            self._live_autorange = False

    # ── Tab builders ────────────────────────────────────────────────────

    def _build_scan_tab(self) -> QtWidgets.QWidget:
        """Operational scan controls — grid parameters, timing, preview."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        # Scan parameters group
        params_group = QtWidgets.QGroupBox("Scan Parameters")
        params_layout = QtWidgets.QGridLayout()

        params_layout.addWidget(QtWidgets.QLabel("<b>Axis</b>"),  0, 0)
        params_layout.addWidget(QtWidgets.QLabel("<b>Start</b>"), 0, 1)
        params_layout.addWidget(QtWidgets.QLabel("<b>Step</b>"),  0, 2)
        params_layout.addWidget(QtWidgets.QLabel("<b>Steps</b>"), 0, 3)

        params_layout.addWidget(QtWidgets.QLabel("X:"), 1, 0)
        self.x_start = QtWidgets.QDoubleSpinBox()
        self.x_start.setRange(-1000, 1000)
        self.x_start.setDecimals(3)
        self.x_start.setValue(-0.16)
        params_layout.addWidget(self.x_start, 1, 1)

        self.x_step = QtWidgets.QDoubleSpinBox()
        self.x_step.setRange(-1000, 1000)
        self.x_step.setDecimals(3)
        self.x_step.setValue(4.0)
        params_layout.addWidget(self.x_step, 1, 2)

        self.x_step_size = QtWidgets.QSpinBox()
        self.x_step_size.setRange(1, 1000)
        self.x_step_size.setValue(2)
        params_layout.addWidget(self.x_step_size, 1, 3)

        params_layout.addWidget(QtWidgets.QLabel("Y:"), 2, 0)
        self.y_start = QtWidgets.QDoubleSpinBox()
        self.y_start.setRange(-1000, 1000)
        self.y_start.setDecimals(3)
        self.y_start.setValue(0.0)
        params_layout.addWidget(self.y_start, 2, 1)

        self.y_step = QtWidgets.QDoubleSpinBox()
        self.y_step.setRange(-1000, 1000)
        self.y_step.setDecimals(3)
        self.y_step.setValue(1.4)
        params_layout.addWidget(self.y_step, 2, 2)

        self.y_step_size = QtWidgets.QSpinBox()
        self.y_step_size.setRange(1, 1000)
        self.y_step_size.setValue(3)
        params_layout.addWidget(self.y_step_size, 2, 3)

        # Z: only used by a 90° second-angle preview
        self.z_label = QtWidgets.QLabel("Z:")
        self.z_label.setToolTip("Horizontal motor seen at 90° — the grid of the 90° preview")
        params_layout.addWidget(self.z_label, 3, 0)
        self.z_start = QtWidgets.QDoubleSpinBox()
        self.z_start.setRange(-1000, 1000)
        self.z_start.setDecimals(3)
        params_layout.addWidget(self.z_start, 3, 1)
        self.z_step = QtWidgets.QDoubleSpinBox()
        self.z_step.setRange(-1000, 1000)
        self.z_step.setDecimals(3)
        self.z_step.setValue(4.0)
        params_layout.addWidget(self.z_step, 3, 2)
        self.z_step_size = QtWidgets.QSpinBox()
        self.z_step_size.setRange(1, 1000)
        self.z_step_size.setValue(2)
        params_layout.addWidget(self.z_step_size, 3, 3)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Second angle: preview the sample also at angle B
        angle_group = QtWidgets.QGroupBox("Second angle")
        angle_form = QtWidgets.QFormLayout()
        self.two_angle = QtWidgets.QCheckBox("Also preview at a second angle")
        self.two_angle.setToolTip(
            "After the X × Y mosaic at angle A, rotate to angle B and take a second "
            "mosaic. Near 90° it scans Z × Y (Z = horizontal motor at 90°); near 180° "
            "it scans X × Y again (double field of view). Then back to angle A.")
        angle_form.addRow(self.two_angle)
        self.angle_a = QtWidgets.QDoubleSpinBox()
        self.angle_a.setRange(-720, 720)
        self.angle_a.setDecimals(2)
        self.angle_a.setSuffix(" °")
        self.angle_a.setValue(0.0)
        angle_form.addRow("Angle A:", self.angle_a)
        self.angle_b = QtWidgets.QDoubleSpinBox()
        self.angle_b.setRange(-720, 720)
        self.angle_b.setDecimals(2)
        self.angle_b.setSuffix(" °")
        self.angle_b.setValue(90.0)
        angle_form.addRow("Angle B:", self.angle_b)
        self.reverse_b = QtWidgets.QCheckBox("Reverse direction at angle B")
        self.reverse_b.setToolTip("Tick if the frames of the angle-B mosaic do not join "
                                  "up (the motor moves the sample the other way on the "
                                  "camera than assumed).")
        angle_form.addRow(self.reverse_b)
        self.angle_info = QtWidgets.QLabel("")
        self.angle_info.setWordWrap(True)
        angle_form.addRow(self.angle_info)
        angle_group.setLayout(angle_form)
        layout.addWidget(angle_group)
        for w_ in (self.two_angle, self.reverse_b):
            w_.toggled.connect(self._update_angle_info)
        for w_ in (self.angle_a, self.angle_b):
            w_.valueChanged.connect(self._update_angle_info)

        # Operational settings (scan timing + test mode + tomoscan mode)
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout()

        self.test_mode_checkbox = QtWidgets.QCheckBox("Test Mode (Mock PVs/Tomoscan)")
        self.test_mode_checkbox.toggled.connect(self._toggle_test_mode)
        settings_layout.addRow("", self.test_mode_checkbox)

        self.pixel_size = QtWidgets.QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 100)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setValue(1.0)
        self.pixel_size.setSuffix(" µm")
        settings_layout.addRow("Pixel Size:", self.pixel_size)

        self.settle_time = QtWidgets.QDoubleSpinBox()
        self.settle_time.setRange(0, 60)
        self.settle_time.setDecimals(1)
        self.settle_time.setValue(5.0)
        self.settle_time.setSuffix(" s")
        settings_layout.addRow("Settle Time:", self.settle_time)

        self.motor_tolerance = QtWidgets.QDoubleSpinBox()
        self.motor_tolerance.setRange(0.0001, 1.0)
        self.motor_tolerance.setDecimals(4)
        self.motor_tolerance.setValue(0.0010)
        self.motor_tolerance.setSuffix(" mm")
        settings_layout.addRow("Motor Tolerance:", self.motor_tolerance)

        self.caput_timeout = QtWidgets.QDoubleSpinBox()
        self.caput_timeout.setRange(1, 300)
        self.caput_timeout.setValue(10.0)
        self.caput_timeout.setSuffix(" s")
        settings_layout.addRow("Caput Timeout:", self.caput_timeout)

        self.tile_timeout = QtWidgets.QDoubleSpinBox()
        self.tile_timeout.setRange(1, 86400)
        self.tile_timeout.setDecimals(0)
        self.tile_timeout.setValue(3600.0)
        self.tile_timeout.setSuffix(" s")
        settings_layout.addRow("Tile Tomoscan Timeout:", self.tile_timeout)

        self.run_tomoscan = QtWidgets.QCheckBox("Run tomoscan at each position")
        settings_layout.addRow("", self.run_tomoscan)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Preview options
        self.enable_preview = QtWidgets.QCheckBox("Enable Live Preview")
        self.enable_preview.setChecked(True)
        self.enable_preview.toggled.connect(self._on_preview_toggled)
        layout.addWidget(self.enable_preview)

        self.auto_contrast = QtWidgets.QCheckBox("Auto Contrast")
        self.auto_contrast.setChecked(True)
        layout.addWidget(self.auto_contrast)

        layout.addStretch()
        return w

    def _build_pv_tab(self) -> QtWidgets.QWidget:
        """All PV-name / IOC-prefix configuration in one place.

        These rarely change once the beamline is set up — putting them on
        a second tab keeps the main Scan tab uncluttered. Labels in the
        side panel mirror motor1_pv / motor2_pv live (see
        `TileSidePanel._refresh_pv_labels`).
        """
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        # Motor PVs group
        pv_group = QtWidgets.QGroupBox("Motor PVs")
        pv_layout = QtWidgets.QFormLayout()

        # No motor default: the right X/Y stages depend on the setup, and a
        # wrong default would move the wrong hardware. _start_scan refuses
        # to run until both are set.
        self.motor1_pv = QtWidgets.QLineEdit("")
        self.motor1_pv.setPlaceholderText("horizontal motor PV")
        self.motor2_pv = QtWidgets.QLineEdit("")
        self.motor2_pv.setPlaceholderText("vertical motor PV")
        self.tomoscan_path = QtWidgets.QLineEdit("tomoscan")

        self.motor3_pv = QtWidgets.QLineEdit("")
        self.motor3_pv.setPlaceholderText("horizontal motor at 90° (for a 90° preview)")
        self.rotation_pv = QtWidgets.QLineEdit("32idbTXM:ens:c1:m1")
        self.rotation_pv.setPlaceholderText("rotation motor PV (second-angle preview)")

        pv_layout.addRow("Motor 1 (X):",   self.motor1_pv)
        pv_layout.addRow("Motor 2 (Y):",   self.motor2_pv)
        pv_layout.addRow("Motor 3 (Z):",   self.motor3_pv)
        pv_layout.addRow("Rotation:",      self.rotation_pv)
        pv_layout.addRow("Tomoscan Path:", self.tomoscan_path)

        pv_group.setLayout(pv_layout)
        layout.addWidget(pv_group)

        # IOC / Camera prefixes + image PV
        ioc_group = QtWidgets.QGroupBox("IOC / Camera")
        ioc_layout = QtWidgets.QFormLayout()

        self.image_pv = QtWidgets.QLineEdit("32idbSP1:Pva1:Image")
        ioc_layout.addRow("Image PV:", self.image_pv)

        self.connection_status = QtWidgets.QLabel("Status: Not connected")
        ioc_layout.addRow("", self.connection_status)

        self.tomoscan_prefix = QtWidgets.QLineEdit("32id:TomoScan:")
        ioc_layout.addRow("Tomoscan Prefix:", self.tomoscan_prefix)

        # camera_prefix → reads MaxSizeX_RBV/MaxSizeY_RBV to size new tiles
        self.camera_prefix = QtWidgets.QLineEdit("32idbSP1:cam1")
        ioc_layout.addRow("Camera Prefix:", self.camera_prefix)

        # crop_prefix → owns CropLeft/Right/Top/Bottom + Crop
        self.crop_prefix = QtWidgets.QLineEdit("32id:TXMOptics")
        ioc_layout.addRow("Crop Prefix:", self.crop_prefix)

        ioc_group.setLayout(ioc_layout)
        layout.addWidget(ioc_group)

        layout.addStretch()
        return w

    def _load_config(self):
        """Load settings from config file"""
        import json
        from pathlib import Path

        # Create config directory in user's home
        config_dir = Path.home() / '.pystream' / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / 'motor_scan_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load PVs
                if 'motor1_pv' in config:
                    self.motor1_pv.setText(config['motor1_pv'])
                if 'motor2_pv' in config:
                    self.motor2_pv.setText(config['motor2_pv'])
                for key in ('motor3_pv', 'rotation_pv'):
                    if key in config:
                        getattr(self, key).setText(config[key])
                for key in ('z_start', 'z_step', 'angle_a', 'angle_b'):
                    if key in config:
                        getattr(self, key).setValue(float(config[key]))
                if 'z_step_size' in config:
                    self.z_step_size.setValue(int(config['z_step_size']))
                for key in ('two_angle', 'reverse_b'):
                    if key in config:
                        getattr(self, key).setChecked(bool(config[key]))
                if 'tomoscan_path' in config:
                    self.tomoscan_path.setText(config['tomoscan_path'])
                if 'image_pv' in config:
                    self.image_pv.setText(config['image_pv'])
                if 'tomoscan_prefix' in config:
                    self.tomoscan_prefix.setText(config['tomoscan_prefix'])

                # Load scan parameters
                if 'pixel_size' in config:
                    self.pixel_size.setValue(config['pixel_size'])
                if 'x_start' in config:
                    self.x_start.setValue(config['x_start'])
                if 'x_step' in config:
                    self.x_step.setValue(config['x_step'])
                if 'x_step_size' in config:
                    self.x_step_size.setValue(config['x_step_size'])
                if 'y_start' in config:
                    self.y_start.setValue(config['y_start'])
                if 'y_step' in config:
                    self.y_step.setValue(config['y_step'])
                if 'y_step_size' in config:
                    self.y_step_size.setValue(config['y_step_size'])
                if 'settle_time' in config:
                    self.settle_time.setValue(config['settle_time'])
                if 'caput_timeout' in config:
                    self.caput_timeout.setValue(config['caput_timeout'])

                # Tile-runner extras
                if 'camera_prefix' in config:
                    self.camera_prefix.setText(config['camera_prefix'])
                if 'crop_prefix' in config:
                    self.crop_prefix.setText(config['crop_prefix'])
                if 'tile_timeout' in config:
                    self.tile_timeout.setValue(float(config['tile_timeout']))

                # Restore tiles (data + ROIs). Needs view_box already built,
                # which is true since _build_ui ran before _load_config.
                tile_state = config.get('tiles')
                if tile_state and self.tile_manager is not None:
                    try:
                        self.tile_manager.deserialize(tile_state)
                        self.tile_manager.restore_rois()
                        snap_b = config.get('tiles_b')
                        if snap_b:
                            self.tile_manager_b.deserialize({"snapshot": snap_b})
                            self.tile_manager_b.restore_rois()
                        if self.tile_manager.tiles:
                            self.add_tile_btn.setEnabled(True)
                            self.run_tiles_btn.setEnabled(True)
                            self._log(
                                f"✓ Restored {len(self.tile_manager.tiles)} "
                                f"tile(s) from config")
                    except Exception as e:
                        self._log(f"⚠ Could not restore tiles: {e}")

                self._log("✓ Loaded settings from config")
            except Exception as e:
                self._log(f"Could not load config: {e}")

    def _save_config(self):
        """Save current settings to config file"""
        import json
        from pathlib import Path

        # Create config directory in user's home
        config_dir = Path.home() / '.pystream' / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)

        config = {
            'motor1_pv': self.motor1_pv.text(),
            'motor2_pv': self.motor2_pv.text(),
            'motor3_pv': self.motor3_pv.text(),
            'rotation_pv': self.rotation_pv.text(),
            'z_start': self.z_start.value(),
            'z_step': self.z_step.value(),
            'z_step_size': self.z_step_size.value(),
            'two_angle': self.two_angle.isChecked(),
            'angle_a': self.angle_a.value(),
            'angle_b': self.angle_b.value(),
            'reverse_b': self.reverse_b.isChecked(),
            'tomoscan_path': self.tomoscan_path.text(),
            'image_pv': self.image_pv.text(),
            'tomoscan_prefix': self.tomoscan_prefix.text(),
            'pixel_size': self.pixel_size.value(),
            'x_start': self.x_start.value(),
            'x_step': self.x_step.value(),
            'x_step_size': self.x_step_size.value(),
            'y_start': self.y_start.value(),
            'y_step': self.y_step.value(),
            'y_step_size': self.y_step_size.value(),
            'settle_time': self.settle_time.value(),
            'caput_timeout': self.caput_timeout.value(),
            'camera_prefix': self.camera_prefix.text(),
            'crop_prefix': self.crop_prefix.text(),
            'tile_timeout': self.tile_timeout.value(),
            'tiles': (self.tile_manager.serialize()
                      if self.tile_manager is not None else None),
            'tiles_b': (self.tile_manager_b.serialize()["snapshot"]
                        if self.tile_manager_b is not None
                        and self.tile_manager_b.is_active() else None),
        }

        try:
            config_file = config_dir / 'motor_scan_config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self._log(f"✓ Settings saved to {config_file}")
        except Exception as e:
            self._log(f"Failed to save config: {e}")

    def _toggle_test_mode(self, checked: bool):
        """Toggle test/mock mode"""
        self.test_mode = checked
        if checked:
            self._log("Test Mode ENABLED - using mock PVs and tomoscan")
            self.connection_status.setText("Test Mode: Mock camera enabled")
        else:
            self._log("Test Mode DISABLED - using real PVs and tomoscan")
            self.connection_status.setText("Camera: checking…")

    def _on_preview_toggled(self, checked: bool):
        """Handle changes to the 'Enable Live Preview' checkbox."""
        if checked:
            self._log("Live preview ENABLED")
            if not self.preview_timer.isActive():
                self.preview_timer.start(500)
            # Force a refresh if we already have a stitched image
            self._refresh_stitched_preview()
        else:
            self._log("Live preview DISABLED")
            if self.preview_timer.isActive():
                self.preview_timer.stop()
            # Clear any image from the preview
            try:
                self.image_view.clear()
            except Exception:
                pass
            if hasattr(self, "_preview_initialized"):
                delattr(self, "_preview_initialized")

    def _generate_mock_image(self, position_index: int, total_positions: int):
        """Generate a mock image for testing"""
        img_h, img_w = 1024, 1280

        # Create base gradient
        y_grad = np.linspace(0, 1, img_h)[:, np.newaxis]
        x_grad = np.linspace(0, 1, img_w)[np.newaxis, :]

        # Position-dependent pattern
        pattern = (np.sin(x_grad * 10 + position_index) * 0.3 +
                   np.cos(y_grad * 8 + position_index * 0.5) * 0.3 +
                   0.4)

        # Add circular features
        center_y, center_x = img_h // 2, img_w // 2
        y_coords = np.arange(img_h)[:, np.newaxis] - center_y
        x_coords = np.arange(img_w)[np.newaxis, :] - center_x
        radius = np.sqrt(x_coords**2 + y_coords**2)
        circle_pattern = np.exp(-radius / 300) * 0.3

        # Combine patterns
        img = pattern + circle_pattern

        # Add noise
        noise = np.random.normal(0, 0.05, (img_h, img_w))
        img = img + noise

        # Normalize to uint16 range
        img = np.clip(img, 0, 1)
        img = (img * 40000 + 5000).astype(np.uint16)

        return img

    def _log(self, message: str):
        """Add message to log in a thread-safe way.

        - Can be called from any thread.
        - Only GUI thread touches QTextEdit.
        """
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"

        # Always send to Python logger / stdout
        if self.logger:
            self.logger.info(message)
        else:
            print(line)

        # Only the GUI thread may touch Qt widgets
        if QtCore.QThread.currentThread() is self.thread():
            try:
                self.log_output.append(line)
            except RuntimeError:
                # Widget might have been destroyed during shutdown
                pass

    def _refresh_stitched_preview(self):
        """Refresh the stitched mosaic display"""
        if not self.enable_preview.isChecked() or self._live_mode():
            return

        if self.graphics_view_b.isHidden() == self._has_view_b():
            self._update_view_titles()          # angle-B mosaic started / gone
        self.stitched_lock.lock()
        try:
            self._render_mosaic(self.stitched_image, self.image_view,
                                self.view_box, "_preview_initialized")
            self._render_mosaic(self.stitched_image_b, self.image_view_b,
                                self.view_box_b, "_preview_initialized_b")
        except Exception as e:
            self._log(f"Preview error: {e}")
        finally:
            self.stitched_lock.unlock()

    def _render_mosaic(self, stitched, image_item, view_box, init_attr):
        """Show one stitched canvas (caller holds stitched_lock)."""
        if stitched is None:
            image_item.clear()
            return
        # Flip vertically to correct display orientation
        img = np.flipud(stitched.copy())

        # Use percentile-based contrast on actual data
        if self.auto_contrast.isChecked():
            # Exclude zeros (unfilled canvas) and extreme values (frame borders)
            # Use a more conservative threshold to exclude background
            nz = img[img > 100]  # Exclude near-zero background
            if nz.size > 100:  # Need enough pixels for good statistics
                # Use narrower percentile range for better contrast
                vmin, vmax = np.percentile(nz, [0.5, 99.5])
                # Ensure reasonable dynamic range
                if vmax - vmin < 1000:
                    # Expand range if too narrow
                    mean_val = np.mean(nz)
                    vmin = max(0, mean_val - 10000)
                    vmax = min(65535, mean_val + 10000)
            else:
                # Fallback for mostly empty canvas
                vmin, vmax = 0, 65535
            image_item.setImage(img, levels=[vmin, vmax])
        else:
            image_item.setImage(img)

        # Auto-range on first update
        if not hasattr(self, init_attr):
            view_box.autoRange()
            setattr(self, init_attr, True)

    # ── frame capture ───────────────────────────────────────────────────
    #
    # Frames come from the Image PV (pvAccess NTNDArray) so mosalign works
    # as its own window: the pystream button launches it as a separate
    # process, where there is no parent viewer to borrow frames from.
    # The parent-viewer path is kept only for when Image PV is left empty.

    _NTNDA_VALUE_KEYS = ('ushortValue', 'shortValue', 'intValue', 'uintValue',
                         'floatValue', 'doubleValue', 'ubyteValue', 'byteValue')

    def _pva_channel(self):
        pv = self.image_pv.text().strip()
        if not pv:
            return None
        ch = self._pva_channels.get(pv)
        if ch is None:
            import pvaccess as pva
            ch = pva.Channel(pv)
            self._pva_channels[pv] = ch
        return ch

    def _frame_uid(self) -> Optional[int]:
        """uniqueId of the newest frame on the Image PV, None if unreadable."""
        if self.test_mode:
            return None
        try:
            ch = self._pva_channel()
            return None if ch is None else int(ch.get('field(uniqueId)')['uniqueId'])
        except Exception:
            return None

    @classmethod
    def _ntnda_to_array(cls, st) -> np.ndarray:
        val = st['value'][0]
        flat = next((np.asarray(val[k]) for k in cls._NTNDA_VALUE_KEYS if k in val), None)
        if flat is None:
            raise RuntimeError("unsupported NTNDArray data type")
        dims = st['dimension']
        # areaDetector order: dimension[0] is the width, dimension[1] the height.
        nx, ny = int(dims[0]['size']), int(dims[1]['size'])
        return flat.reshape(ny, nx)

    def _get_image_from_pv(self, after_uid: Optional[int], timeout: float = 10.0):
        """Newest frame on the Image PV. With `after_uid`, wait for a frame
        newer than that one, so a frame captured after a motor move really
        shows the new position."""
        pv = self.image_pv.text().strip()
        deadline = time.monotonic() + timeout
        uid = None
        while True:
            uid = self._frame_uid()
            if uid is not None and (after_uid is None or uid != after_uid):
                try:
                    return self._ntnda_to_array(self._pva_channel().get(''))
                except Exception as e:
                    self._last_capture_error = f"could not decode frame from {pv}: {e}"
                    return None
            if time.monotonic() >= deadline:
                self._last_capture_error = (
                    f"no frame on {pv}" if uid is None else
                    f"no new frame on {pv} within {timeout:.0f}s — is the camera acquiring?")
                return None
            time.sleep(0.2)

    def _get_image_now(self, position_index: int = 1, total_positions: int = 1,
                       after_uid: Optional[int] = None):
        self._last_capture_error = ""
        if self.test_mode:
            return self._generate_mock_image(position_index, total_positions)
        if self.image_pv.text().strip():
            return self._get_image_from_pv(after_uid)

        try:
            parent = self.parent()

            if not parent:
                self._log(f"⚠ No parent viewer available")
                return None

            for attempt in range(3):
                try:
                    if not hasattr(parent, '_last_display_img'):
                        self._log(f"⚠ Parent viewer does not have _last_display_img attribute")
                        return None

                    img_ref = parent._last_display_img

                    if img_ref is None:
                        if attempt < 2:
                            time.sleep(0.05)
                            continue
                        else:
                            self._log(f"⚠ No image available from main viewer after retries")
                            self._log(f"   Please ensure the main viewer is connected and receiving images")
                            return None

                    img = np.array(img_ref, dtype=img_ref.dtype)

                    if img.size == 0:
                        self._log(f"⚠ Empty image from main viewer")
                        return None

                    self._log(f"✓ Got image from main viewer ({img.shape[1]}x{img.shape[0]})")
                    return img

                except (AttributeError, RuntimeError, ValueError) as e:
                    if attempt < 2:
                        self._log(f"⚠ Retry {attempt + 1}/3: {e}")
                        time.sleep(0.05)
                        continue
                    else:
                        self._log(f"⚠ Could not copy viewer image after retries: {e}")
                        return None

            return None

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self._log(f"⚠ Unexpected error getting image: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def _caput(self, pv: str, value: float, timeout: float):
        if self.test_mode:
            return

        try:
            subprocess.run(
                ['caput', '-w', str(timeout), pv, str(value)],
                capture_output=True,
                timeout=timeout + 5,
                check=True
            )
        except subprocess.TimeoutExpired:
            self._log(f"⚠ caput timeout for {pv}")
        except subprocess.CalledProcessError as e:
            self._log(f"⚠ caput error for {pv}: {e}")

    def _caget(self, pv: str, timeout: float = 2.0) -> Optional[str]:
        """Return PV value as a stripped string, or None on failure.

        No-op in test mode (returns None). Used by both `_wait_for_motor`
        and the tile runner's crop/override snapshots.
        """
        if self.test_mode:
            return None
        try:
            result = subprocess.run(
                ['caget', '-t', pv],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, ValueError) as e:
            self._log(f"  Warning: caget error for {pv}: {e}")
        return None

    def _wait_for_motor(self, pv: str, target: float, tolerance: float = 0.001, timeout: float = 30.0):
        if self.test_mode:
            time.sleep(0.05)
            return True

        readback_pv = f"{pv}.RBV"
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            raw = self._caget(readback_pv, timeout=2.0)
            if raw is not None:
                try:
                    if abs(float(raw) - target) <= tolerance:
                        return True
                except ValueError:
                    self._log(f"  Warning: non-numeric readback {readback_pv}={raw!r}")
            time.sleep(0.1)

        self._log(f"⚠ Motor {pv} timeout")
        return False

    def _motors_configured(self) -> bool:
        if self.test_mode:
            return True
        if self.motor1_pv.text().strip() and self.motor2_pv.text().strip():
            return True
        self._log("⚠ Set both X and Y motor PVs first.")
        return False

    def _second_view(self) -> Optional[dict]:
        """What the angle-B preview scans, or None when it is off.

        The motor that is horizontal on the camera at angle B is X when B is
        nearer to A ± 180° (or A) and Z when it is nearer to A ± 90°; `sign`
        is -1 when that motor moves the sample the other way on the camera
        than X does at angle A (X at 180°)."""
        if not self.two_angle.isChecked():
            return None
        d = math.radians(self.angle_b.value() - self.angle_a.value())
        c, sn = math.cos(d), math.sin(d)
        if abs(c) >= abs(sn):
            v = dict(h_attr="x_mm", motor="X", pv=self.motor1_pv.text().strip(),
                     start=self.x_start.value(), step=self.x_step.value(),
                     n=self.x_step_size.value(), sign=1 if c > 0 else -1)
        else:
            v = dict(h_attr="z_mm", motor="Z", pv=self.motor3_pv.text().strip(),
                     start=self.z_start.value(), step=self.z_step.value(),
                     n=self.z_step_size.value(), sign=1 if sn > 0 else -1)
        if self.reverse_b.isChecked():
            v["sign"] = -v["sign"]
        v["angle_a"] = self.angle_a.value()
        v["angle_b"] = self.angle_b.value()
        return v

    def _update_angle_info(self, *_):
        v = self._second_view()
        on = v is not None
        for w_ in (self.angle_a, self.angle_b, self.reverse_b):
            w_.setEnabled(on)
        uses_z = on and v["h_attr"] == "z_mm"
        for w_ in (self.z_label, self.z_start, self.z_step, self.z_step_size):
            w_.setEnabled(uses_z)
        if not on:
            self.angle_info.setText("")
            return
        if uses_z:
            txt = (f"At {v['angle_b']:g}°: Z × Y grid (Z row above, Motor 3 PV). "
                   f"X stays at the centre of its range")
        else:
            txt = f"At {v['angle_b']:g}°: the same X × Y grid"
        if v["sign"] < 0:
            txt += " — X/Z moves the image the other way, the mosaic is mirrored"
        self.angle_info.setText(txt + ".")

    def _second_view_ready(self) -> bool:
        v = self._second_view()
        if v is None or self.test_mode:
            return True
        if not self.rotation_pv.text().strip():
            self._log("⚠ Second angle: set the Rotation PV (PVs tab) first.")
            return False
        if not v["pv"]:
            self._log(f"⚠ Second angle at {v['angle_b']:g}° scans Z: set Motor 3 (Z) "
                      f"on the PVs tab first.")
            return False
        return True

    def _start_scan(self):
        """Start the 2D motor scan"""
        if not self._motors_configured() or not self._second_view_ready():
            return
        self.stitched_lock.lock()
        self.stitched_image_b = None
        self.stitched_lock.unlock()
        self._last_scan_geom_b = None
        self.tile_manager_b.x_start_mm = None     # old angle-B tiles go
        self.tile_manager_b.restore_rois()
        self._set_view_mode(live=False)      # show the mosaic being built
        self.scanning = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        # Reset preview initialization
        for attr in ("_preview_initialized",
                     "_placeholder_shown",
                     "_preview_error_logged",
                     "_scan_just_finished"):
            if hasattr(self, attr):
                delattr(self, attr)

        if self.enable_preview.isChecked():
            if not self.preview_timer.isActive():   # stopped by the last scan
                self.preview_timer.start(500)
            self._log("Preview timer active (500ms refresh)")
        else:
            self._log("Live preview disabled - no mosaic display")

        # Create and start scan thread
        self.scan_thread = ScanWorker(self)
        self.scan_thread.progress_signal.connect(self._update_progress)
        self.scan_thread.log_signal.connect(self._log)
        self.scan_thread.position_signal.connect(self._update_position)
        self.scan_thread.finished_signal.connect(self._scan_finished)
        self.scan_thread.start()

        self._log("=== Scan started ===")

    def _stop_scan(self):
        """Stop the running scan"""
        self.scanning = False
        self._log("⚠ Stopping scan...")
        self.stop_btn.setEnabled(False)
        # If a tile run is in flight, kill the active caput so its per-tile
        # and outer try/finally restoration blocks can run.
        if self.tile_run_worker is not None and self.tile_run_worker.isRunning():
            self.tile_run_worker.terminate_active()

    # ── interactive-tiles slots ─────────────────────────────────────────

    def _add_tile_at_center(self):
        if self.tile_manager is None:
            return
        self.tile_manager.add_tile_at_view_center()

    def _run_tiles(self):
        """Launch TileRunWorker to step through each tile + StartScan."""
        if self.tile_manager is None or not self.tile_manager.tiles:
            self._log("⚠ No tiles to run — finish a grid scan first.")
            return
        if self.tile_run_worker is not None and self.tile_run_worker.isRunning():
            self._log("⚠ Tile run already in progress.")
            return
        if not self._motors_configured():
            return

        self.scanning = True
        self.start_btn.setEnabled(False)
        self.add_tile_btn.setEnabled(False)
        self.run_tiles_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        self.tile_run_worker = TileRunWorker(self)
        self.tile_run_worker.progress_signal.connect(self._update_progress)
        self.tile_run_worker.log_signal.connect(self._log)
        self.tile_run_worker.finished_signal.connect(self._tile_run_finished)
        self.tile_run_worker.start()

    def _tile_run_finished(self):
        self.scanning = False
        self.start_btn.setEnabled(True)
        self.add_tile_btn.setEnabled(True)
        self.run_tiles_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _get_sensor_size(self, fallback_w: int, fallback_h: int):
        """Return (sensor_w, sensor_h) in raw camera pixels.

        Real mode reads `{camera_prefix}:MaxSizeX_RBV` / `MaxSizeY_RBV`;
        falls back to the captured-frame dims if those PVs are unreachable.
        Test mode skips caget and uses the captured-frame dims directly.
        """
        cam = self.camera_prefix.text().strip()
        if self.test_mode or not cam:
            return fallback_w, fallback_h
        sw_raw = self._caget(f"{cam}:MaxSizeX_RBV")
        sh_raw = self._caget(f"{cam}:MaxSizeY_RBV")
        try:
            sw = int(float(sw_raw)) if sw_raw is not None else fallback_w
        except ValueError:
            sw = fallback_w
        try:
            sh = int(float(sh_raw)) if sh_raw is not None else fallback_h
        except ValueError:
            sh = fallback_h
        return sw, sh

    def _scan_finished(self):
        """Called when scan completes"""
        self.scanning = False
        self._scan_just_finished = True  # Prevent preview from accessing camera data

        # Stop preview timer to avoid further camera access
        if self.preview_timer.isActive():
            self.preview_timer.stop()
            self._log("Preview timer stopped (scan finished)")

        if self.enable_preview.isChecked():
            try:
                self._refresh_stitched_preview()
            except Exception as e:
                self._log(f"Final preview refresh error: {e}")

        QtCore.QCoreApplication.processEvents()
        time.sleep(0.1)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log("=== Scan finished ===")

        # Seed the interactive-tiles overlay from the just-finished grid.
        # If the worker bailed before stashing geometry (e.g. tomoscan-mosaic
        # path, or test-image failure), skip silently.
        geom = self._last_scan_geom
        if geom and self.tile_manager is not None:
            sensor_w, sensor_h = self._get_sensor_size(
                geom["img_w"], geom["img_h"])
            try:
                self.tile_manager.seed_from_grid(
                    x_start=geom["x_start"],
                    y_start=geom["y_start"],
                    x_step=geom["x_step"],
                    y_step=geom["y_step"],
                    nx=geom["nx"], ny=geom["ny"],
                    pixel_size_um=geom["pixel_size_um"],
                    img_w=geom["img_w"], img_h=geom["img_h"],
                    out_h=geom["out_h"],
                    sensor_w=sensor_w, sensor_h=sensor_h,
                    z_mm=geom.get("z_mm"),
                )
                gb = self._last_scan_geom_b
                tmb = self.tile_manager_b
                if gb:
                    tmb.set_geometry(h_start=gb["h_start"], y_start=gb["y_start"],
                                     pixel_size_um=gb["pixel_size_um"],
                                     img_w=gb["img_w"], img_h=gb["img_h"],
                                     out_h=gb["out_h"], sensor_w=sensor_w,
                                     sensor_h=sensor_h, h_attr=gb["h_attr"],
                                     h_sign=gb["h_sign"])
                else:
                    tmb.x_start_mm = None          # no second-angle mosaic
                tmb.restore_rois()
                self.add_tile_btn.setEnabled(True)
                self.run_tiles_btn.setEnabled(True)
            except Exception as e:
                self._log(f"⚠ Could not seed tiles: {e}")
        self._update_view_titles()

    def _update_progress(self, current: int, total: int):
        """Update progress bar"""
        percentage = int((current / total) * 100)
        self.progress.setValue(percentage)

    def _update_position(self, position: int, total: int, x: float, y: float, z: float):
        """Update current position display"""
        self.position_label.setText(
            f"Position: {position}/{total} | X={x:.3f}, Y={y:.3f}"
        )

    def closeEvent(self, event):
        """Clean up when dialog is closed"""
        self.scanning = False

        # Save settings
        self._save_config()

        if self.preview_timer.isActive():
            self.preview_timer.stop()

        if self.live_thread is not None:
            self.live_thread.stop()
            self.live_thread.wait(3000)

        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait(1000)
            if self.scan_thread.isRunning():
                self.scan_thread.terminate()

        if self.tile_run_worker and self.tile_run_worker.isRunning():
            self.tile_run_worker.terminate_active()
            self.tile_run_worker.wait(2000)
            if self.tile_run_worker.isRunning():
                self.tile_run_worker.terminate()

        event.accept()


class ScanWorker(QtCore.QThread):
    """Worker thread for running the scan"""
    progress_signal = QtCore.pyqtSignal(int, int)
    log_signal = QtCore.pyqtSignal(str)
    position_signal = QtCore.pyqtSignal(int, int, float, float, float)
    finished_signal = QtCore.pyqtSignal()

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    def run(self):
        """Run the 2D X-Y scan or execute mosaic.sh script"""
        try:
            if self.dialog.run_tomoscan.isChecked():
                self._run_mosaic_script()
                return

            d = self.dialog
            motor1_pv = d.motor1_pv.text().strip()
            motor2_pv = d.motor2_pv.text().strip()
            x = (d.x_start.value(), d.x_step.value(), d.x_step_size.value())
            y = (d.y_start.value(), d.y_step.value(), d.y_step_size.value())
            sv = d._second_view()
            rot_pv = d.rotation_pv.text().strip()
            centre = lambda g: g[0] + (g[2] - 1) * g[1] / 2.0
            uses_z = sv is not None and sv["h_attr"] == "z_mm"
            z_mid = centre((sv["start"], sv["step"], sv["n"])) if uses_z else None

            if sv is not None:
                if not self._rotate(rot_pv, sv["angle_a"]):
                    return
                if uses_z:                     # Z out of the way: middle of its range
                    if not self._move(sv["pv"], z_mid, "Z"):
                        return

            geom = self._grid(motor1_pv, x, motor2_pv, y, sign=1, view="a")
            if geom is None:
                return
            geom.update(x_start=x[0], y_start=y[0], x_step=x[1], y_step=y[1],
                        nx=x[2], ny=y[2], z_mm=z_mid,
                        angle=sv["angle_a"] if sv else None)
            d._last_scan_geom = geom

            if sv is not None and d.scanning:
                self.log_signal.emit("─" * 60)
                if uses_z:                     # X is along the beam at 90°
                    if not self._move(motor1_pv, centre(x), "X"):
                        return
                if not self._rotate(rot_pv, sv["angle_b"]):
                    return
                gb = self._grid(sv["pv"], (sv["start"], sv["step"], sv["n"]),
                                motor2_pv, y, sign=sv["sign"], view="b",
                                h_name=sv["motor"])
                if gb is not None:
                    gb.update(h_attr=sv["h_attr"], h_sign=sv["sign"],
                              angle=sv["angle_b"])
                    d._last_scan_geom_b = gb
                if d.scanning:
                    self._rotate(rot_pv, sv["angle_a"])
                else:
                    self.log_signal.emit(f"⚠ stopped — rotation left at {sv['angle_b']:g}°")

        except Exception as e:
            self.log_signal.emit(f"✗ Scan failed: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()

    # ── moves ────────────────────────────────────────────────────────────

    def _move(self, pv: str, target: float, name: str) -> bool:
        d = self.dialog
        self.log_signal.emit(f"Move {name} ({pv}) → {target:.3f}")
        d._caput(pv, target, d.caput_timeout.value())
        if d._wait_for_motor(pv, target, d.motor_tolerance.value(),
                             d.caput_timeout.value()):
            return True
        self.log_signal.emit(f"✗ {name} did not reach {target:.3f} — scan stopped")
        return False

    ROT_TOLERANCE_DEG = 0.05

    def _rotate(self, pv: str, angle: float) -> bool:
        """Rotate the sample and wait until it is there (a 90° turn can take
        a while, so the timeout is at least 3 min)."""
        d = self.dialog
        timeout = max(180.0, d.caput_timeout.value())
        self.log_signal.emit(f"Rotate ({pv}) → {angle:g}°")
        d._caput(pv, angle, timeout)
        if d._wait_for_motor(pv, angle, self.ROT_TOLERANCE_DEG, timeout):
            time.sleep(0.1 if d.test_mode else d.settle_time.value())
            return True
        self.log_signal.emit(f"✗ rotation did not reach {angle:g}° — scan stopped")
        return False

    def _grid(self, h_pv, h, v_pv, v, sign, view, h_name="X") -> Optional[dict]:
        """Grid of frames into one mosaic: horizontal motor outer, Y inner.

        h, v: (start, step, steps). `sign` -1 places frames mirrored (the
        horizontal motor moves the sample the other way on the camera).
        Fills dialog.stitched_image (view "a") or stitched_image_b ("b")
        and returns the canvas geometry, or None if no frame could be read.
        """
        d = self.dialog
        settle_time = d.settle_time.value()
        motor_tolerance = d.motor_tolerance.value()
        timeout = d.caput_timeout.value()
        nh, nv = h[2], v[2]
        total_positions = nh * nv
        attr = "stitched_image" if view == "a" else "stitched_image_b"

        self.log_signal.emit(f"Moving to start: {h_name}={h[0]:.3f}, Y={v[0]:.3f}")
        d._caput(h_pv, h[0], timeout)
        time.sleep(0.1)
        d._caput(v_pv, v[0], timeout)
        d._wait_for_motor(h_pv, h[0], motor_tolerance, timeout)
        d._wait_for_motor(v_pv, v[0], motor_tolerance, timeout)
        time.sleep(settle_time if not d.test_mode else 0.1)

        # Test image to determine canvas size
        self.log_signal.emit("Getting test image...")
        test_img = d._get_image_now(0, total_positions)
        if test_img is None:
            err = d._last_capture_error
            self.log_signal.emit("⚠ Failed to get test image!" + (f" ({err})" if err else ""))
            return None
        img_h, img_w = test_img.shape
        pixel_size_um = d.pixel_size.value()
        mm_px = 1000.0 / pixel_size_um

        # Canvas left edge = lowest sign×motor value; top (before the display
        # flip) = lowest Y. Frames go where their motor position says.
        u_ends = (sign * h[0], sign * (h[0] + (nh - 1) * h[1]))
        v_ends = (v[0], v[0] + (nv - 1) * v[1])
        u_min, v_min = min(u_ends), min(v_ends)
        out_w = int(round(abs(u_ends[1] - u_ends[0]) * mm_px)) + img_w
        out_h = int(round(abs(v_ends[1] - v_ends[0]) * mm_px)) + img_h
        self.log_signal.emit(f"Canvas: {out_w}×{out_h}, Image: {img_w}×{img_h}, "
                             f"Step: {abs(h[1]) * mm_px:.0f}×{abs(v[1]) * mm_px:.0f}px")

        d.stitched_lock.lock()
        setattr(d, attr, np.zeros((out_h, out_w), dtype=np.uint16))
        d.stitched_lock.unlock()

        position = 0
        for i in range(nh):
            if not d.scanning:
                break
            for j in range(nv):
                if not d.scanning:
                    break
                h_pos = h[0] + i * h[1]
                v_pos = v[0] + j * v[1]
                position += 1
                self.log_signal.emit(
                    f"[{position}/{total_positions}] {h_name}={h_pos:.3f}, Y={v_pos:.3f}")

                # Remember the newest frame before moving, so the capture
                # below waits for a frame taken at the new position.
                uid_before = d._frame_uid()

                if not d.test_mode:
                    try:
                        subprocess.run(['caput', h_pv, str(h_pos)],
                                       capture_output=True, timeout=5)
                        subprocess.run(['caput', v_pv, str(v_pos)],
                                       capture_output=True, timeout=5)
                    except Exception as e:
                        self.log_signal.emit(f"  ⚠ Motor error: {e}")

                at_h = d._wait_for_motor(h_pv, h_pos, motor_tolerance, timeout)
                at_v = d._wait_for_motor(v_pv, v_pos, motor_tolerance, timeout)
                self.position_signal.emit(position, total_positions, h_pos, v_pos, 0)

                if not (at_h and at_v):
                    self.log_signal.emit("  ✗ motors did not reach target — frame not captured")
                    self.progress_signal.emit(position, total_positions)
                    continue

                time.sleep(0.1 if d.test_mode else settle_time)

                img = None
                # The PV path already waits up to 10 s for a fresh frame.
                max_retries = 1 if d.image_pv.text().strip() else 3
                for retry in range(max_retries):
                    img = d._get_image_now(position, total_positions, after_uid=uid_before)
                    if img is not None:
                        break
                    if retry < max_retries - 1:
                        self.log_signal.emit(f"  ⚠ No image yet, waiting... ({retry+1}/{max_retries})")
                        time.sleep(0.2)

                if img is not None:
                    col = int(round((sign * h_pos - u_min) * mm_px))
                    row = int(round((v_pos - v_min) * mm_px))
                    self._place_image_in_canvas(img, col, row, attr)
                    self.log_signal.emit("  ✓ Image captured and placed in mosaic")
                else:
                    err = d._last_capture_error
                    self.log_signal.emit("  ✗ No image available - " + (err or "check camera stream"))
                self.progress_signal.emit(position, total_positions)

        self.log_signal.emit(f"Mosaic done: {position} position(s)")
        return dict(h_start=u_min, y_start=v_min, pixel_size_um=pixel_size_um,
                    img_w=img_w, img_h=img_h, out_h=out_h)

    def _run_mosaic_script(self):
        """Run tomoscan mosaic command directly (integrated, no external shell script)"""
        try:
            h_steps = self.dialog.x_step_size.value()
            v_steps = self.dialog.y_step_size.value()
            h_step_size = self.dialog.x_step.value()
            v_step_size = self.dialog.y_step.value()
            h_start = self.dialog.x_start.value()
            v_start = self.dialog.y_start.value()
            tomoscan_prefix = self.dialog.tomoscan_prefix.text().strip()
            tomoscan_cmd = self.dialog.tomoscan_path.text().strip()

            if h_steps < 1 or v_steps < 1:
                self.log_signal.emit("✗ Error: Steps must be >= 1")
                return

            # Build tomoscan mosaic command
            cmd = [
                tomoscan_cmd, 'mosaic',
                '--scan-type', 'Mosaic',
                '--tomoscan-prefix', tomoscan_prefix,
                '--horizontal-steps', str(h_steps),
                '--vertical-steps', str(v_steps),
                '--horizontal-step-size', str(h_step_size),
                '--vertical-step-size', str(v_step_size),
                '--horizontal-start', str(h_start),
                '--vertical-start', str(v_start),
                '--verbose'
            ]

            self.log_signal.emit("═" * 60)
            self.log_signal.emit("TOMOSCAN MOSAIC MODE (Integrated)")
            self.log_signal.emit("═" * 60)
            self.log_signal.emit(f"Command: {tomoscan_cmd}")
            self.log_signal.emit(f"Grid: {h_steps}x{v_steps}")
            self.log_signal.emit(f"Start positions: X={h_start}mm, Y={v_start}mm")
            self.log_signal.emit(f"Step sizes: X={h_step_size}mm, Y={v_step_size}mm")
            self.log_signal.emit(f"Tomoscan prefix: {tomoscan_prefix}")
            self.log_signal.emit(f"Total scans: {h_steps * v_steps}")
            self.log_signal.emit("─" * 60)

            # Run with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output in real-time
            for line in process.stdout:
                if not self.dialog.scanning:
                    process.terminate()
                    self.log_signal.emit("⚠ Scan stopped by user")
                    break
                self.log_signal.emit(line.rstrip())

            # Wait for process to complete
            returncode = process.wait()

            self.log_signal.emit("─" * 60)
            if returncode == 0:
                self.log_signal.emit("✓ Tomoscan mosaic completed successfully")
            else:
                self.log_signal.emit(f"✗ Tomoscan mosaic failed with exit code {returncode}")
            self.log_signal.emit("═" * 60)

        except FileNotFoundError:
            self.log_signal.emit(f"✗ Error: Command '{tomoscan_cmd}' not found")
            self.log_signal.emit("   Make sure tomoscan is installed and in your PATH")
            self.log_signal.emit("   You may need to activate the tomoscan conda environment")
        except Exception as e:
            self.log_signal.emit(f"✗ Failed to run tomoscan mosaic: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())

    def _place_image_in_canvas(self, img, start_x, start_y, attr="stitched_image"):
        """Paste a frame into a stitched canvas at pixel (start_x, start_y)."""
        img_h, img_w = img.shape
        self.dialog.stitched_lock.lock()
        try:
            canvas = getattr(self.dialog, attr)
            out_h, out_w = canvas.shape
            end_x = min(start_x + img_w, out_w)
            end_y = min(start_y + img_h, out_h)
            canvas[start_y:end_y, start_x:end_x] = img[:end_y - start_y, :end_x - start_x]
        finally:
            self.dialog.stitched_lock.unlock()


def main():
    """Run mosalign as standalone application"""
    import sys

    app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)

    dialog = MotorScanDialog()
    dialog.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
