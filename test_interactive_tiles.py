#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosalign — interactive tiles smoke test (no beamline required).

Pops the dialog in TEST MODE, paints a recognizable fake stitched image
into the preview, and seeds a 2×3 tile grid. Everything works offline:
- drag tiles → motor coords update in the log
- resize tiles → crop math is computed (no real PVs touched)
- right-click → Edit Overrides / Reset Size / Delete
- "+ Tile at Center" → adds a new tile centered on the current view
- "Run Tiles" → walks every tile and prints the per-tile sequence
  (crop apply → motor move → override apply → StartScan → restore)

Run:
    python test_interactive_tiles.py
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# Make sure the in-repo mosalign package is found, not any installed copy
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosalign.gui import MotorScanDialog


# ── fake-image generator ────────────────────────────────────────────────

def make_fake_mosaic(nx: int, ny: int,
                     frame_w: int, frame_h: int,
                     overlap_px: int = 0) -> np.ndarray:
    """Build a recognizable fake stitched canvas: per-tile color block,
    gradient, big tile number, and a centered ring."""
    eff_w = frame_w - overlap_px
    eff_h = frame_h - overlap_px
    out_w = eff_w * nx + overlap_px
    out_h = eff_h * ny + overlap_px

    canvas = np.zeros((out_h, out_w), dtype=np.uint16)

    rng = np.random.default_rng(7)
    for i in range(nx):
        for j in range(ny):
            x0 = i * eff_w
            y0 = j * eff_h

            # Gradient base, different brightness per cell
            y_grad = np.linspace(0.3, 0.9, frame_h)[:, None]
            x_grad = np.linspace(0.4, 0.8, frame_w)[None, :]
            base = (x_grad + y_grad) * 0.5

            # Centered ring — different radius per tile
            yy, xx = np.indices((frame_h, frame_w))
            cy = frame_h // 2
            cx = frame_w // 2
            r = np.hypot(xx - cx, yy - cy)
            ring_r = 80 + 40 * ((i + 2 * j) % 4)
            ring = np.exp(-((r - ring_r) ** 2) / (2 * 12.0 ** 2)) * 0.6

            cell = base + ring + rng.normal(0, 0.03, base.shape)
            cell = np.clip(cell, 0, 1)
            cell16 = (cell * 50_000 + 5_000).astype(np.uint16)

            canvas[y0:y0 + frame_h, x0:x0 + frame_w] = cell16

    return canvas


# ── entry point ─────────────────────────────────────────────────────────

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Dark Fusion palette (matches the production GUI)
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window,          QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText,      QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base,            QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase,   QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Text,            QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button,          QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText,      QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight,       QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)

    dlg = MotorScanDialog()

    # ── force test mode so no real caput/caget runs ────────────────────
    dlg.test_mode_checkbox.setChecked(True)

    # ── grid geometry (matches what the scan-finished hook would see) ──
    NX, NY        = 2, 3
    FRAME_W, FRAME_H = 1280, 1024            # camera native pixels
    PIXEL_UM      = 0.766                    # 32-ID TXM pixel size
    X_START, Y_START = -0.16, 0.0            # mm
    X_STEP,  Y_STEP  = (FRAME_W * PIXEL_UM / 1000.0,
                        FRAME_H * PIXEL_UM / 1000.0)  # adjacent, no overlap

    # ── paint the fake stitched canvas into the preview ────────────────
    img = make_fake_mosaic(NX, NY, FRAME_W, FRAME_H)
    out_h, out_w = img.shape

    dlg.stitched_lock.lock()
    dlg.stitched_image = img
    dlg.stitched_lock.unlock()
    dlg._refresh_stitched_preview()

    # ── seed the tile overlay & enable the tile buttons ────────────────
    dlg.tile_manager.seed_from_grid(
        x_start=X_START, y_start=Y_START,
        x_step=X_STEP,   y_step=Y_STEP,
        nx=NX, ny=NY,
        pixel_size_um=PIXEL_UM,
        img_w=FRAME_W, img_h=FRAME_H,
        out_h=out_h,
        sensor_w=FRAME_W, sensor_h=FRAME_H,
    )
    dlg.add_tile_btn.setEnabled(True)
    dlg.run_tiles_btn.setEnabled(True)

    # Make sure spinboxes match the painted canvas (so save_config writes
    # consistent values; not strictly required for the test).
    dlg.x_start.setValue(X_START)
    dlg.y_start.setValue(Y_START)
    dlg.x_step.setValue(X_STEP)
    dlg.y_step.setValue(Y_STEP)
    dlg.x_step_size.setValue(NX)
    dlg.y_step_size.setValue(NY)
    dlg.pixel_size.setValue(PIXEL_UM)

    dlg._log("─" * 60)
    dlg._log(f"TEST HARNESS: fake mosaic loaded ({NX}×{NY} of "
             f"{FRAME_W}×{FRAME_H} px, {out_w}×{out_h} canvas).")
    dlg._log("→ Double-click a tile to edit it in the side panel on the right.")
    dlg._log("  • drag the tile: panel X/Y track live")
    dlg._log("  • edit X/Y in the panel: tile moves on the canvas")
    dlg._log("  • set # Projections / Integration to override tomoscan PVs")
    dlg._log("    for that one tile (0 = use IOC default).")
    dlg._log("→ + Tile at Center: adds a tile at the current view center.")
    dlg._log("→ Run Tiles: walks every tile and runs the per-tile sequence.")
    dlg._log("─" * 60)

    dlg.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
