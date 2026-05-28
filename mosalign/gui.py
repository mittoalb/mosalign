#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosalign - Motor Scan Plugin for PyStream
------------------------------------------
Performs 2D motor scan with live stitched preview.

Features:
- X-Y motor grid scan with image stitching
- Tomoscan integration at each position
- Test mode for offline development
"""

import logging
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

        # For stitched preview
        self.stitched_image = None
        self.stitched_lock = QtCore.QMutex()

        # Interactive tiles (created after view_box exists in _build_ui)
        self.tile_manager: Optional[TileManager] = None
        self.tile_run_worker: Optional[TileRunWorker] = None
        # Last grid-scan geometry, stashed by ScanWorker for seed_from_grid
        self._last_scan_geom: Optional[dict] = None

        # Test/Mock mode
        self.test_mode = False

        # Timer to refresh preview during scan
        self.preview_timer = QtCore.QTimer()
        self.preview_timer.timeout.connect(self._refresh_stitched_preview)

        self.setWindowTitle("Mosalign")
        self.resize(1200, 800)

        self._build_ui()
        self._load_config()

        self._on_preview_toggled(self.enable_preview.isChecked())

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # ── Left panel ─────────────────────────────────────────────────
        # QTabWidget at top (Scan = operational, PVs = IOC config)
        # + always-visible action area at bottom (Start/Stop/Progress).
        left_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(left_panel)

        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self._build_scan_tab(), "Scan")
        self.left_tabs.addTab(self._build_pv_tab(),   "PVs")
        layout.addWidget(self.left_tabs, 1)

        # Action buttons (kept outside the tabs so they're always visible)
        button_layout = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start Scan")
        self.start_btn.setAutoDefault(False)
        self.start_btn.setDefault(False)
        self.start_btn.clicked.connect(self._start_scan)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setAutoDefault(False)
        self.stop_btn.setDefault(False)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_scan)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        self.position_label = QtWidgets.QLabel("Position: -")
        layout.addWidget(self.position_label)

        main_layout.addWidget(left_panel, 1)

        # Right panel - Image preview & log
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Image view
        right_layout.addWidget(QtWidgets.QLabel("Live Preview (Stitched Mosaic)"))

        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setMinimumSize(400, 400)  # Ensure preview is visible
        self.view_box = self.graphics_view.addViewBox()
        self.view_box.setAspectLocked(True)
        self.image_view = pg.ImageItem()
        self.view_box.addItem(self.image_view)

        # Tile toolbar — sits above the preview. Buttons stay disabled until
        # a grid scan completes (and _scan_finished seeds the tile manager).
        tile_btns = QtWidgets.QHBoxLayout()
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
        tile_btns.addStretch()
        right_layout.addLayout(tile_btns)

        # Tile manager owns ROIs on view_box; created here so it's ready for
        # config-load-time restore_rois() calls.
        self.tile_manager = TileManager(self.view_box, self)

        # Canvas + side panel in a resizable horizontal splitter.
        # Side panel shows the currently-selected tile's motor X/Y +
        # NumAngles / ExposureTime overrides; double-click a tile to edit.
        self.tile_side_panel = TileSidePanel(self.tile_manager, parent=self)
        canvas_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        canvas_splitter.addWidget(self.graphics_view)
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

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

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

        self.motor1_pv = QtWidgets.QLineEdit("2bmb:m17")
        self.motor2_pv = QtWidgets.QLineEdit("2bmHXP:m3")
        self.tomoscan_path = QtWidgets.QLineEdit("tomoscan")

        pv_layout.addRow("Motor 1 (X):",   self.motor1_pv)
        pv_layout.addRow("Motor 2 (Y):",   self.motor2_pv)
        pv_layout.addRow("Tomoscan Path:", self.tomoscan_path)

        pv_group.setLayout(pv_layout)
        layout.addWidget(pv_group)

        # IOC / Camera prefixes + image PV
        ioc_group = QtWidgets.QGroupBox("IOC / Camera")
        ioc_layout = QtWidgets.QFormLayout()

        self.image_pv = QtWidgets.QLineEdit("2bmbSP1:Pva1:Image")
        ioc_layout.addRow("Image PV:", self.image_pv)

        self.connection_status = QtWidgets.QLabel("Status: Not connected")
        ioc_layout.addRow("", self.connection_status)

        self.tomoscan_prefix = QtWidgets.QLineEdit("2bmb:TomoScan:")
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
            self.connection_status.setText("Status: Not connected")

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
        if not self.enable_preview.isChecked():
            return

        self.stitched_lock.lock()
        try:
            if self.stitched_image is not None:
                # Flip vertically to correct display orientation
                img = np.flipud(self.stitched_image.copy())

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
                    self.image_view.setImage(img, levels=[vmin, vmax])
                else:
                    self.image_view.setImage(img)

                # Auto-range on first update
                if not hasattr(self, '_preview_initialized'):
                    self.view_box.autoRange()
                    self._preview_initialized = True
        except Exception as e:
            self._log(f"Preview error: {e}")
        finally:
            self.stitched_lock.unlock()

    def _get_image_now(self, position_index: int = 1, total_positions: int = 1):
        if self.test_mode:
            return self._generate_mock_image(position_index, total_positions)

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

    def _start_scan(self):
        """Start the 2D motor scan"""
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
                )
                self.add_tile_btn.setEnabled(True)
                self.run_tiles_btn.setEnabled(True)
            except Exception as e:
                self._log(f"⚠ Could not seed tiles: {e}")

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

            motor1_pv = self.dialog.motor1_pv.text().strip()
            motor2_pv = self.dialog.motor2_pv.text().strip()

            x_start = self.dialog.x_start.value()
            x_step = self.dialog.x_step.value()
            x_step_size = self.dialog.x_step_size.value()

            y_start = self.dialog.y_start.value()
            y_step = self.dialog.y_step.value()
            y_step_size = self.dialog.y_step_size.value()

            settle_time = self.dialog.settle_time.value()
            motor_tolerance = self.dialog.motor_tolerance.value()
            timeout = self.dialog.caput_timeout.value()

            total_positions = x_step_size * y_step_size
            position = 0

            self.log_signal.emit(f"Moving to start: X={x_start:.3f}, Y={y_start:.3f}")
            self.dialog._caput(motor1_pv, x_start, timeout)
            time.sleep(0.1)
            self.dialog._caput(motor2_pv, y_start, timeout)
            time.sleep(settle_time if not self.dialog.test_mode else 0.1)

            # Get test image to determine canvas size
            self.log_signal.emit("Getting test image...")
            test_img = self.dialog._get_image_now(0, total_positions)

            if test_img is not None:
                img_h, img_w = test_img.shape

                pixel_size_um = self.dialog.pixel_size.value()
                x_step_um = x_step * 1000
                y_step_um = y_step * 1000

                eff_w = int(x_step_um / pixel_size_um)
                eff_h = int(y_step_um / pixel_size_um)

                out_w = eff_w * x_step_size + (img_w - eff_w)
                out_h = eff_h * y_step_size + (img_h - eff_h)

                self.log_signal.emit(
                    f"Canvas: {out_w}×{out_h}, Image: {img_w}×{img_h}, Step: {eff_w}×{eff_h}px"
                )
            else:
                self.log_signal.emit("⚠ Failed to get test image!")
                return

            self.dialog.stitched_lock.lock()
            self.dialog.stitched_image = np.zeros((out_h, out_w), dtype=np.uint16)
            self.dialog.stitched_lock.unlock()

            # Stash geometry so _scan_finished can seed the tile manager.
            # img_w/img_h are the per-frame size; out_h is the full canvas.
            self.dialog._last_scan_geom = dict(
                x_start=x_start, y_start=y_start,
                x_step=x_step, y_step=y_step,
                nx=x_step_size, ny=y_step_size,
                pixel_size_um=pixel_size_um,
                img_w=img_w, img_h=img_h,
                out_h=out_h,
            )

            for i in range(x_step_size):
                if not self.dialog.scanning:
                    break

                for j in range(y_step_size):
                    if not self.dialog.scanning:
                        break

                    x_pos = x_start + (i * x_step)
                    y_pos = y_start + (j * y_step)

                    position += 1

                    self.log_signal.emit(
                        f"[{position}/{total_positions}] X={x_pos:.3f}, Y={y_pos:.3f}"
                    )

                    if not self.dialog.test_mode:
                        try:
                            subprocess.run(['caput', motor1_pv, str(x_pos)],
                                           capture_output=True, timeout=5)
                            subprocess.run(['caput', motor2_pv, str(y_pos)],
                                           capture_output=True, timeout=5)
                        except Exception as e:
                            self.log_signal.emit(f"  ⚠ Motor error: {e}")

                    self.dialog._wait_for_motor(motor1_pv, x_pos, motor_tolerance, timeout)
                    self.dialog._wait_for_motor(motor2_pv, y_pos, motor_tolerance, timeout)

                    self.position_signal.emit(position, total_positions, x_pos, y_pos, 0)

                    time.sleep(0.1 if self.dialog.test_mode else settle_time)

                    img = None
                    max_retries = 3

                    for retry in range(max_retries):
                        img = self.dialog._get_image_now(position, total_positions)
                        if img is not None:
                            break
                        if retry < max_retries - 1:
                            self.log_signal.emit(f"  ⚠ No image yet, waiting... ({retry+1}/{max_retries})")
                            time.sleep(0.2)

                    if img is not None:
                        self._place_image_in_canvas(img, i, j, eff_w, eff_h, out_w, out_h)
                        self.log_signal.emit(f"  ✓ Image captured and placed in mosaic")
                    else:
                        self.log_signal.emit(f"  ✗ No image available - check camera stream")

                    self.progress_signal.emit(position, total_positions)

            self.log_signal.emit(f"Scan complete! Total positions: {position}")

        except Exception as e:
            self.log_signal.emit(f"✗ Scan failed: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()

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

    def _place_image_in_canvas(self, img, i, j, eff_w, eff_h, out_w, out_h):
        """Place image in stitched canvas"""
        img_h, img_w = img.shape

        start_x = i * eff_w
        start_y = j * eff_h
        end_x = min(start_x + img_w, out_w)
        end_y = min(start_y + img_h, out_h)

        sub = img[:end_y - start_y, :end_x - start_x]

        self.dialog.stitched_lock.lock()
        try:
            self.dialog.stitched_image[start_y:end_y, start_x:end_x] = sub
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
