#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosalign — Interactive tile overlay + per-tile tomoscan runner.

The stitched preview built by `ScanWorker` is treated as a static backdrop.
After that scan finishes, `TileManager.seed_from_grid()` creates one
`Tile` per grid cell with a matching `pg.RectROI` overlaid on the canvas.
The user can drag, resize (= crop), add, delete, and per-tile-override any
tomoscan PV. `TileRunWorker` walks the tile list and runs one
`{tomoscan_prefix}StartScan` per tile, saving/restoring crop and override
PV values around each.

Coordinate convention
---------------------
The preview applies `np.flipud(stitched_image)` at display time
(gui.py:_refresh_stitched_preview). We keep that flip and account for it
in `TileManager.mm_to_canvas_px` by inverting Y. If the display flip is
ever removed, fix the mapping in `mm_to_canvas_px` / `px_to_mm` only.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets


class _TileROI(pg.RectROI):
    """RectROI that:
      * always shows its (custom) context menu on right-click
        (pg.RectROI gates `raiseContextMenu` on `removable`),
      * emits `sigDoubleClicked` on left-button double-click so the
        dialog can populate its side panel.

    pyqtgraph's `MouseClickEvent` carries a `.double()` flag — that's
    where ROI double-clicks actually land. The Qt-standard
    `mouseDoubleClickEvent` rarely fires on pg ROIs because pyqtgraph's
    GraphicsScene swallows the press/release stream into its own
    MouseClickEvent dispatch. We override `mouseClickEvent` (pyqtgraph's
    method, not Qt's) and intercept `ev.double()`; we keep
    `mouseDoubleClickEvent` as a defensive fallback.
    """
    sigDoubleClicked = QtCore.pyqtSignal()

    def contextMenuEnabled(self):
        return True

    def mouseClickEvent(self, ev):
        if (ev.button() == QtCore.Qt.LeftButton
                and hasattr(ev, "double") and ev.double()):
            self.sigDoubleClicked.emit()
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        # Fallback for the (rare) path where Qt's double-click actually
        # reaches us instead of being routed through MouseClickEvent.
        if ev.button() == QtCore.Qt.LeftButton:
            self.sigDoubleClicked.emit()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)


# ────────────────────────────────────────────────────────────────────────
# Tile data model
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Tile:
    """A single tile = motor position + camera-pixel size + optional overrides.

    `x_mm`/`y_mm` are the authoritative motor coordinates; the on-screen
    `pg.RectROI` is a *view* of this dataclass. Per-tile `overrides` maps a
    tomoscan PV **suffix** (e.g. "NumAngles") to a string value; the runner
    prepends `tomoscan_prefix` at execution time.
    """
    id: int
    x_mm: float
    y_mm: float
    width_px: int
    height_px: int
    overrides: Dict[str, str] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────
# Overrides editor
# ────────────────────────────────────────────────────────────────────────

class TileOverridesDialog(QtWidgets.QDialog):
    """Two-column (PV suffix, value) table editor for a single tile.

    Empty table = "use the GUI defaults for this tile." PV suffix is just
    the tomoscan PV name without the prefix (e.g. "NumAngles",
    "ExposureTime").
    """

    def __init__(self, tile: Tile, tomoscan_prefix: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tile #{tile.id} — Tomoscan Overrides")
        self.resize(420, 320)
        self._tile = tile

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            f"Overrides for tile #{tile.id}. PV name is the suffix appended "
            f"to <b>{tomoscan_prefix}</b>. Empty rows are ignored."))

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["PV suffix", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        # Populate from current tile overrides
        for k, val in tile.overrides.items():
            self._append_row(k, val)

        row_btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Row")
        add_btn.clicked.connect(lambda: self._append_row("", ""))
        rm_btn = QtWidgets.QPushButton("− Selected")
        rm_btn.clicked.connect(self._remove_selected)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(rm_btn)
        row_btns.addStretch()
        v.addLayout(row_btns)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _append_row(self, key: str, value: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(value))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _accept(self):
        out: Dict[str, str] = {}
        for r in range(self.table.rowCount()):
            k_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            k = (k_item.text() if k_item else "").strip()
            v = (v_item.text() if v_item else "").strip()
            if k:
                out[k] = v
        self._tile.overrides = out
        self.accept()


# ────────────────────────────────────────────────────────────────────────
# Tile manager (ROIs + dataclass list + canvas coord mapping)
# ────────────────────────────────────────────────────────────────────────

class TileManager(QtCore.QObject):
    """Owns the tile list and the on-screen ROIs.

    The `pg.ViewBox` and the parent dialog (for logging + tomoscan prefix
    on the right-click menu) are passed in. The scan snapshot
    (x_start_mm, y_start_mm, pixel_size_um, img_w/h, out_h, sensor_w/h)
    is captured once in `seed_from_grid` and held for the session — later
    spinbox edits in the main GUI do not move existing tiles.

    Signals
    -------
    selectionChanged(int)   — emitted with new selected tile id, or -1
    tileChanged(int)        — emitted while the user drags/resizes a tile
                              (live, not debounced) AND on the final commit.
                              The side panel listens to keep its X/Y/size
                              fields in sync with the canvas.
    """

    MIN_SIZE_PX = 16

    # Pens for the unselected vs selected ROIs (visual cue for which tile
    # the side panel is currently editing).
    _PEN_NORMAL = ('y', 2)                  # yellow, 2 px
    _PEN_SELECTED = ((80, 220, 255), 3)     # cyan, 3 px

    selectionChanged = QtCore.pyqtSignal(int)
    tileChanged = QtCore.pyqtSignal(int)

    def __init__(self, view_box: pg.ViewBox, dialog, logger=None):
        super().__init__()
        self.view_box = view_box
        self.dialog = dialog
        self.logger = logger

        self.tiles: List[Tile] = []
        self._rois: Dict[int, pg.RectROI] = {}
        self._next_id: int = 1
        self._selected_tid: int = -1

        # Snapshot captured at seed_from_grid time; None when no session active
        self.x_start_mm: Optional[float] = None
        self.y_start_mm: Optional[float] = None
        self.pixel_size_um: Optional[float] = None
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.out_h: Optional[int] = None
        self.sensor_w: Optional[int] = None
        self.sensor_h: Optional[int] = None

    # ── logging passthrough ─────────────────────────────────────────────

    def _log(self, msg: str):
        try:
            self.dialog._log(msg)
        except Exception:
            pass

    # ── coord mapping (motor mm ↔ canvas px) ────────────────────────────

    def is_active(self) -> bool:
        return self.x_start_mm is not None and self.pixel_size_um

    def mm_to_canvas_px(self, x_mm: float, y_mm: float):
        """Top-left of a default-sized (img_w × img_h) tile at (x_mm, y_mm).

        Y is inverted to match the display `np.flipud` in
        gui._refresh_stitched_preview.
        """
        cx = (x_mm - self.x_start_mm) * 1000.0 / self.pixel_size_um
        cy = (self.out_h - self.img_h
              - (y_mm - self.y_start_mm) * 1000.0 / self.pixel_size_um)
        return cx, cy

    def canvas_px_to_mm(self, cx: float, cy: float):
        """Inverse of `mm_to_canvas_px` on the tile's top-left."""
        x_mm = self.x_start_mm + cx * self.pixel_size_um / 1000.0
        y_mm = (self.y_start_mm
                + (self.out_h - self.img_h - cy)
                * self.pixel_size_um / 1000.0)
        return x_mm, y_mm

    # ── seeding + lifecycle ─────────────────────────────────────────────

    def seed_from_grid(self, *, x_start: float, y_start: float,
                       x_step: float, y_step: float,
                       nx: int, ny: int,
                       pixel_size_um: float,
                       img_w: int, img_h: int, out_h: int,
                       sensor_w: int, sensor_h: int):
        """Wipe existing tiles and create one per grid cell."""
        self.clear_all()

        self.x_start_mm = float(x_start)
        self.y_start_mm = float(y_start)
        self.pixel_size_um = float(pixel_size_um)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.out_h = int(out_h)
        self.sensor_w = int(sensor_w)
        self.sensor_h = int(sensor_h)

        for i in range(nx):
            for j in range(ny):
                t = Tile(
                    id=self._next_id,
                    x_mm=self.x_start_mm + i * float(x_step),
                    y_mm=self.y_start_mm + j * float(y_step),
                    width_px=self.sensor_w,
                    height_px=self.sensor_h,
                )
                self._next_id += 1
                self.tiles.append(t)
                self._make_roi(t)

        self._log(f"[tiles] seeded {len(self.tiles)} tiles from {nx}×{ny} grid "
                  f"(sensor {sensor_w}×{sensor_h})")

    def clear_all(self):
        for tid in list(self._rois.keys()):
            self._remove_roi(tid)
        self.tiles = []

    def add_tile_at_view_center(self):
        if not self.is_active():
            self._log("[tiles] no scan snapshot yet — run a grid scan first")
            return
        # Centre of current view in scene coords; map to a tile top-left
        # (top-left = centre − half sensor)
        vr = self.view_box.viewRect()
        cx = vr.center().x() - self.sensor_w / 2.0
        cy = vr.center().y() - self.sensor_h / 2.0
        x_mm, y_mm = self.canvas_px_to_mm(cx, cy)
        t = Tile(
            id=self._next_id,
            x_mm=x_mm, y_mm=y_mm,
            width_px=self.sensor_w, height_px=self.sensor_h,
        )
        self._next_id += 1
        self.tiles.append(t)
        self._make_roi(t)
        self._log(f"[tiles] +tile #{t.id} at "
                  f"X={x_mm:.3f} Y={y_mm:.3f}")

    def delete_tile(self, tid: int):
        tile = self._tile_by_id(tid)
        if tile is None:
            return
        if tid == self._selected_tid:
            self.clear_selection()
        self.tiles.remove(tile)
        self._remove_roi(tid)
        self._log(f"[tiles] −tile #{tid}")

    def reset_size(self, tid: int):
        tile = self._tile_by_id(tid)
        if tile is None or not self.is_active():
            return
        tile.width_px = self.sensor_w
        tile.height_px = self.sensor_h
        self._sync_roi_from_tile(tile)
        self._log(f"[tiles] tile #{tid} size reset to "
                  f"{self.sensor_w}×{self.sensor_h}")

    def edit_overrides(self, tid: int):
        tile = self._tile_by_id(tid)
        if tile is None:
            return
        prefix = getattr(self.dialog, "tomoscan_prefix", None)
        prefix_text = prefix.text().strip() if prefix is not None else ""
        dlg = TileOverridesDialog(tile, prefix_text, parent=self.dialog)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            n = len(tile.overrides)
            self._log(f"[tiles] tile #{tid} overrides: {n} entries")

    # ── ROI plumbing ────────────────────────────────────────────────────

    def _tile_by_id(self, tid: int) -> Optional[Tile]:
        for t in self.tiles:
            if t.id == tid:
                return t
        return None

    def _make_roi(self, tile: Tile):
        cx, cy = self.mm_to_canvas_px(tile.x_mm, tile.y_mm)
        pen_color, pen_w = self._PEN_NORMAL
        pen = pg.mkPen(pen_color, width=pen_w)
        hover_pen = pg.mkPen((255, 255, 100), width=3)
        roi = _TileROI(
            [cx, cy], [tile.width_px, tile.height_px],
            pen=pen, hoverPen=hover_pen,
            movable=True, resizable=True, removable=False,
        )
        roi.setZValue(1000)
        # Add the corner handles (built-in RectROI gives one; add the rest)
        roi.addScaleHandle([0, 0], [1, 1])  # top-left
        roi.addScaleHandle([1, 0], [0, 1])  # top-right
        roi.addScaleHandle([0, 1], [1, 0])  # bottom-left
        roi.addScaleHandle([0.5, 0],   [0.5, 1])
        roi.addScaleHandle([0.5, 1],   [0.5, 0])
        roi.addScaleHandle([0,   0.5], [1,   0.5])
        roi.addScaleHandle([1,   0.5], [0,   0.5])

        # Tile-id label (small text item at top-left of ROI, parented to ROI)
        label = pg.TextItem(text=f"#{tile.id}", color=(255, 255, 0),
                            anchor=(0, 1))
        label.setParentItem(roi)
        label.setPos(0, 0)

        # Right-click menu — override to insert our actions
        self._install_menu(roi, tile.id)

        # Live drag: update Tile (no clamp/log) and emit so the side panel
        # tracks the cursor. Final commit (clamp + log) on Finished.
        roi.sigRegionChanged.connect(
            lambda _roi, tid=tile.id: self._on_roi_dragging(tid))
        roi.sigRegionChangeFinished.connect(
            lambda _roi, tid=tile.id: self._on_roi_finished(tid))

        # Double-click to select (populates the side panel)
        roi.sigDoubleClicked.connect(
            lambda tid=tile.id: self.select_tile(tid))

        self.view_box.addItem(roi)
        self._rois[tile.id] = roi

    def _remove_roi(self, tid: int):
        roi = self._rois.pop(tid, None)
        if roi is not None:
            try:
                self.view_box.removeItem(roi)
            except Exception:
                pass

    def _install_menu(self, roi: pg.RectROI, tid: int):
        menu = QtWidgets.QMenu()
        a_edit = menu.addAction("Edit Overrides…")
        a_reset = menu.addAction("Reset Size (no crop)")
        menu.addSeparator()
        a_del = menu.addAction("Delete Tile")

        a_edit.triggered.connect(lambda: self.edit_overrides(tid))
        a_reset.triggered.connect(lambda: self.reset_size(tid))
        a_del.triggered.connect(lambda: self.delete_tile(tid))

        # pyqtgraph ROI exposes its menu via `getMenu`/`menu` and shows it
        # on right-click via `raiseContextMenu`. Easiest: shadow the
        # method so our menu is returned.
        roi.getMenu = lambda: menu  # type: ignore[assignment]

    def _on_roi_dragging(self, tid: int):
        """Live update while the user is moving/resizing a tile.

        Cheap: updates the Tile dataclass (no clamp / no log) and emits
        `tileChanged` so the side panel can mirror the cursor. The real
        clamp + log happens in `_on_roi_finished`.
        """
        tile = self._tile_by_id(tid)
        roi = self._rois.get(tid)
        if tile is None or roi is None or not self.is_active():
            return
        pos = roi.pos()
        size = roi.size()
        tile.x_mm, tile.y_mm = self.canvas_px_to_mm(pos.x(), pos.y())
        tile.width_px = int(round(size.x()))
        tile.height_px = int(round(size.y()))
        self.tileChanged.emit(tid)

    def _on_roi_finished(self, tid: int):
        tile = self._tile_by_id(tid)
        roi = self._rois.get(tid)
        if tile is None or roi is None or not self.is_active():
            return

        pos = roi.pos()
        size = roi.size()
        new_w = max(self.MIN_SIZE_PX, min(self.sensor_w, int(round(size.x()))))
        new_h = max(self.MIN_SIZE_PX, min(self.sensor_h, int(round(size.y()))))
        new_x_mm, new_y_mm = self.canvas_px_to_mm(pos.x(), pos.y())

        clamped = (new_w != int(round(size.x()))
                   or new_h != int(round(size.y())))

        tile.x_mm = new_x_mm
        tile.y_mm = new_y_mm
        tile.width_px = new_w
        tile.height_px = new_h

        if clamped:
            # Reflect the clamp back to the ROI so the user sees it
            self._sync_roi_from_tile(tile)

        self._log(f"[tiles] tile #{tid} → X={new_x_mm:.3f} Y={new_y_mm:.3f}  "
                  f"size {new_w}×{new_h}")
        self.tileChanged.emit(tid)

    # ── selection ───────────────────────────────────────────────────────

    @property
    def selected_tid(self) -> int:
        return self._selected_tid

    def select_tile(self, tid: int):
        """Highlight tile `tid` (or -1 to clear) and emit selectionChanged."""
        if tid == self._selected_tid:
            return
        # Restore previous selection's pen
        if self._selected_tid in self._rois:
            c, w = self._PEN_NORMAL
            self._rois[self._selected_tid].setPen(pg.mkPen(c, width=w))
        # Highlight the new one
        if tid in self._rois:
            c, w = self._PEN_SELECTED
            self._rois[tid].setPen(pg.mkPen(c, width=w))
            self._selected_tid = tid
        else:
            self._selected_tid = -1
        self.selectionChanged.emit(self._selected_tid)

    def clear_selection(self):
        self.select_tile(-1)

    # ── setters callable from the side panel ────────────────────────────

    def set_tile_position(self, tid: int, x_mm: float, y_mm: float):
        """Move tile `tid` to (x_mm, y_mm). Side panel calls this when the
        user edits the X/Y spinboxes; we update the Tile + sync the ROI
        without re-emitting tileChanged (the panel is the originator)."""
        tile = self._tile_by_id(tid)
        if tile is None or not self.is_active():
            return
        tile.x_mm = float(x_mm)
        tile.y_mm = float(y_mm)
        self._sync_roi_from_tile(tile)
        self._log(f"[tiles] tile #{tid} → X={x_mm:.3f} Y={y_mm:.3f} "
                  f"(from panel)")

    def set_tile_override(self, tid: int, pv_suffix: str,
                          value: Optional[str]):
        """Set / clear a single override on tile `tid`. `value=None` or
        empty string deletes the entry (= use IOC default for this tile)."""
        tile = self._tile_by_id(tid)
        if tile is None:
            return
        if value is None or value == "":
            tile.overrides.pop(pv_suffix, None)
        else:
            tile.overrides[pv_suffix] = str(value)

    def _sync_roi_from_tile(self, tile: Tile):
        roi = self._rois.get(tile.id)
        if roi is None:
            return
        cx, cy = self.mm_to_canvas_px(tile.x_mm, tile.y_mm)
        roi.blockSignals(True)
        try:
            roi.setPos([cx, cy], update=False)
            roi.setSize([tile.width_px, tile.height_px], update=True)
        finally:
            roi.blockSignals(False)

    # ── (de)serialization for config save/load ──────────────────────────

    def serialize(self) -> dict:
        """Snapshot of tiles + canvas coord state, for JSON config."""
        return {
            "snapshot": {
                "x_start_mm": self.x_start_mm,
                "y_start_mm": self.y_start_mm,
                "pixel_size_um": self.pixel_size_um,
                "img_w": self.img_w,
                "img_h": self.img_h,
                "out_h": self.out_h,
                "sensor_w": self.sensor_w,
                "sensor_h": self.sensor_h,
            },
            "tiles": [asdict(t) for t in self.tiles],
            "next_id": self._next_id,
        }

    def deserialize(self, data: dict):
        """Restore from `serialize()`. ROIs are NOT created here — call
        `restore_rois()` after the canvas is up if you want them visible."""
        snap = data.get("snapshot") or {}
        self.x_start_mm = snap.get("x_start_mm")
        self.y_start_mm = snap.get("y_start_mm")
        self.pixel_size_um = snap.get("pixel_size_um")
        self.img_w = snap.get("img_w")
        self.img_h = snap.get("img_h")
        self.out_h = snap.get("out_h")
        self.sensor_w = snap.get("sensor_w")
        self.sensor_h = snap.get("sensor_h")

        self.tiles = []
        for d in data.get("tiles", []):
            self.tiles.append(Tile(
                id=int(d["id"]),
                x_mm=float(d["x_mm"]),
                y_mm=float(d["y_mm"]),
                width_px=int(d["width_px"]),
                height_px=int(d["height_px"]),
                overrides=dict(d.get("overrides") or {}),
            ))
        ids = [t.id for t in self.tiles]
        self._next_id = max(data.get("next_id", 1),
                            (max(ids) + 1) if ids else 1)

    def restore_rois(self):
        """Re-create on-screen ROIs from the current tile list. Use after
        `deserialize()` once the canvas + snapshot are valid."""
        for tid in list(self._rois.keys()):
            self._remove_roi(tid)
        if not self.is_active():
            return
        for t in self.tiles:
            self._make_roi(t)


# ────────────────────────────────────────────────────────────────────────
# Side panel — selected-tile editor
# ────────────────────────────────────────────────────────────────────────

class TileSidePanel(QtWidgets.QWidget):
    """Docked editor for the currently-selected tile.

    Shows tile id + X/Y motor positions + the two most-common overrides
    (NumAngles = number of projections, ExposureTime = integration time
    in seconds). Bidirectionally bound to the TileManager:

      * double-click a tile on the canvas → fields populate
      * drag the tile → X/Y fields update live (read from the manager's
        `tileChanged` signal)
      * type into a field → tile moves / override is written

    Empty NumAngles / ExposureTime fields mean "no override" — the
    tomoscan IOC's current value is used for that tile.
    """

    # Override PV suffixes (without the tomoscan prefix). Matches the
    # tomoscan IOC convention.
    PV_NUM_ANGLES = "NumAngles"
    PV_EXPOSURE   = "ExposureTime"

    def __init__(self, tile_manager: TileManager, parent=None):
        super().__init__(parent)
        self.tm = tile_manager
        self.dialog = tile_manager.dialog   # for live PV-name lookup
        self._suspend_writeback = False     # guard against signal loops

        # PV-name labels can be long ("32idcSOFT:m17:") so give the panel
        # enough horizontal room that QFormLayout doesn't squeeze the label
        # column to zero. The splitter on the right pane sets the initial
        # width; this just guarantees a sensible minimum.
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        self.header = QtWidgets.QLabel("<b>No tile selected</b>")
        self.header.setStyleSheet("color: #d0d0d0;")
        v.addWidget(self.header)

        v.addWidget(QtWidgets.QLabel(
            "<i>Double-click a tile on the canvas to edit it.</i>"))

        # Editable fields
        self.box = QtWidgets.QGroupBox("Selected Tile")
        form = QtWidgets.QFormLayout(self.box)
        # Long PV labels need their own space — let the field column grow
        # into whatever's left rather than the form sizing both columns to
        # the field's preferred width.
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # If a PV label is wider than the field column, wrap onto two lines
        # rather than clipping the label.
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)

        # X / Y row labels mirror the motor-PV fields on the left panel
        # (motor1_pv / motor2_pv) and refresh live when those are edited.
        self.x_label = QtWidgets.QLabel("Motor X:")
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setDecimals(4)
        self.x_spin.setSuffix(" mm")
        self.x_spin.setKeyboardTracking(False)
        form.addRow(self.x_label, self.x_spin)

        self.y_label = QtWidgets.QLabel("Motor Y:")
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setDecimals(4)
        self.y_spin.setSuffix(" mm")
        self.y_spin.setKeyboardTracking(False)
        form.addRow(self.y_label, self.y_spin)

        # Use double spin with special-value at minimum = "(default)"
        self.proj_spin = QtWidgets.QSpinBox()
        self.proj_spin.setRange(0, 100000)
        self.proj_spin.setSpecialValueText("(default)")  # shows at value 0
        self.proj_spin.setKeyboardTracking(False)
        form.addRow("# Projections:", self.proj_spin)

        self.expo_spin = QtWidgets.QDoubleSpinBox()
        self.expo_spin.setRange(0.0, 3600.0)
        self.expo_spin.setDecimals(4)
        self.expo_spin.setSingleStep(0.001)
        self.expo_spin.setSuffix(" s")
        self.expo_spin.setSpecialValueText("(default)")  # value 0 = unset
        self.expo_spin.setKeyboardTracking(False)
        form.addRow("Integration:", self.expo_spin)

        self.size_label = QtWidgets.QLabel("—")
        form.addRow("Crop size:", self.size_label)

        v.addWidget(self.box)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset Size")
        self.reset_btn.setToolTip("Restore tile to full sensor size (no crop)")
        self.reset_btn.clicked.connect(self._on_reset_size)
        btns.addWidget(self.reset_btn)

        self.del_btn = QtWidgets.QPushButton("Delete Tile")
        self.del_btn.clicked.connect(self._on_delete)
        btns.addWidget(self.del_btn)
        v.addLayout(btns)

        self.adv_btn = QtWidgets.QPushButton("Other overrides…")
        self.adv_btn.setToolTip(
            "Edit additional tomoscan PV overrides for this tile")
        self.adv_btn.clicked.connect(self._on_advanced)
        v.addWidget(self.adv_btn)

        v.addStretch()

        # Wire signals
        self.x_spin.valueChanged.connect(self._on_xy_edited)
        self.y_spin.valueChanged.connect(self._on_xy_edited)
        self.proj_spin.valueChanged.connect(self._on_proj_edited)
        self.expo_spin.valueChanged.connect(self._on_expo_edited)

        self.tm.selectionChanged.connect(self._on_selection_changed)
        self.tm.tileChanged.connect(self._on_tile_changed)

        # Keep the X/Y labels in sync with the editable motor-PV fields
        # on the left panel. Tolerate missing attrs (e.g. in tests).
        if hasattr(self.dialog, "motor1_pv"):
            self.dialog.motor1_pv.textChanged.connect(
                lambda _t: self._refresh_pv_labels())
        if hasattr(self.dialog, "motor2_pv"):
            self.dialog.motor2_pv.textChanged.connect(
                lambda _t: self._refresh_pv_labels())
        self._refresh_pv_labels()

        self._set_enabled(False)

    # ── motor-PV label sync ─────────────────────────────────────────────

    def _refresh_pv_labels(self):
        """Mirror the motor-PV fields on the left panel into the X/Y
        row labels. Empty PV → fall back to the generic 'Motor X/Y:'."""
        pv1 = (self.dialog.motor1_pv.text().strip()
               if hasattr(self.dialog, "motor1_pv") else "")
        pv2 = (self.dialog.motor2_pv.text().strip()
               if hasattr(self.dialog, "motor2_pv") else "")
        self.x_label.setText(f"{pv1}:" if pv1 else "Motor X:")
        self.y_label.setText(f"{pv2}:" if pv2 else "Motor Y:")

    # ── enable/disable based on whether a tile is selected ──────────────

    def _set_enabled(self, on: bool):
        for w in (self.x_spin, self.y_spin, self.proj_spin, self.expo_spin,
                  self.reset_btn, self.del_btn, self.adv_btn):
            w.setEnabled(on)

    # ── signals from TileManager ────────────────────────────────────────

    def _on_selection_changed(self, tid: int):
        if tid < 0:
            self.header.setText("<b>No tile selected</b>")
            self.size_label.setText("—")
            self._set_enabled(False)
            return
        tile = self.tm._tile_by_id(tid)
        if tile is None:
            return
        self.header.setText(f"<b>Tile #{tid}</b>")
        self._set_enabled(True)
        self._populate_from(tile)

    def _on_tile_changed(self, tid: int):
        if tid != self.tm.selected_tid:
            return
        tile = self.tm._tile_by_id(tid)
        if tile is None:
            return
        # Only refresh fields the user is not currently editing —
        # blockSignals avoids re-triggering our valueChanged handlers.
        self._populate_from(tile, fields=('xy', 'size'))

    def _populate_from(self, tile: Tile, fields=('xy', 'size', 'overrides')):
        self._suspend_writeback = True
        try:
            if 'xy' in fields:
                self.x_spin.blockSignals(True)
                self.x_spin.setValue(tile.x_mm)
                self.x_spin.blockSignals(False)
                self.y_spin.blockSignals(True)
                self.y_spin.setValue(tile.y_mm)
                self.y_spin.blockSignals(False)
            if 'size' in fields:
                self.size_label.setText(
                    f"{tile.width_px} × {tile.height_px} px")
            if 'overrides' in fields:
                self.proj_spin.blockSignals(True)
                try:
                    self.proj_spin.setValue(
                        int(float(tile.overrides.get(self.PV_NUM_ANGLES, 0))))
                except ValueError:
                    self.proj_spin.setValue(0)
                self.proj_spin.blockSignals(False)

                self.expo_spin.blockSignals(True)
                try:
                    self.expo_spin.setValue(
                        float(tile.overrides.get(self.PV_EXPOSURE, 0.0)))
                except ValueError:
                    self.expo_spin.setValue(0.0)
                self.expo_spin.blockSignals(False)
        finally:
            self._suspend_writeback = False

    # ── signals from the editable fields ────────────────────────────────

    def _on_xy_edited(self, *_):
        if self._suspend_writeback or self.tm.selected_tid < 0:
            return
        self.tm.set_tile_position(self.tm.selected_tid,
                                  self.x_spin.value(),
                                  self.y_spin.value())

    def _on_proj_edited(self, val: int):
        if self._suspend_writeback or self.tm.selected_tid < 0:
            return
        # value 0 == special "(default)" text → clear the override
        if val <= 0:
            self.tm.set_tile_override(
                self.tm.selected_tid, self.PV_NUM_ANGLES, None)
        else:
            self.tm.set_tile_override(
                self.tm.selected_tid, self.PV_NUM_ANGLES, str(val))

    def _on_expo_edited(self, val: float):
        if self._suspend_writeback or self.tm.selected_tid < 0:
            return
        if val <= 0.0:
            self.tm.set_tile_override(
                self.tm.selected_tid, self.PV_EXPOSURE, None)
        else:
            self.tm.set_tile_override(
                self.tm.selected_tid, self.PV_EXPOSURE, f"{val:g}")

    # ── button slots ────────────────────────────────────────────────────

    def _on_reset_size(self):
        tid = self.tm.selected_tid
        if tid >= 0:
            self.tm.reset_size(tid)

    def _on_delete(self):
        tid = self.tm.selected_tid
        if tid >= 0:
            self.tm.delete_tile(tid)

    def _on_advanced(self):
        tid = self.tm.selected_tid
        if tid >= 0:
            self.tm.edit_overrides(tid)
            # Re-populate to catch any changes the dialog made
            tile = self.tm._tile_by_id(tid)
            if tile is not None:
                self._populate_from(tile, fields=('overrides',))


# ────────────────────────────────────────────────────────────────────────
# Per-tile execution worker
# ────────────────────────────────────────────────────────────────────────

class TileRunWorker(QtCore.QThread):
    """Runs one `tomoscan StartScan` per tile, saving/restoring crop +
    per-tile override PVs around each tile.

    Mirrors `ScanWorker`'s signal shape so the dialog can wire it the same
    way. Reuses `dialog._caput`, `dialog._caget`, and
    `dialog._wait_for_motor`.
    """

    progress_signal = QtCore.pyqtSignal(int, int)
    log_signal      = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self._active_proc: Optional[subprocess.Popen] = None

    # ── public helpers ──────────────────────────────────────────────────

    def terminate_active(self):
        """Called by the dialog's Stop button. Kill the in-flight caput so
        the worker's per-tile `finally` block can run restoration."""
        if self._active_proc and self._active_proc.poll() is None:
            try:
                self._active_proc.terminate()
            except Exception:
                pass

    # ── crop math ──────────────────────────────────────────────────────

    def _apply_crop(self, crop_prefix: str, tile: Tile):
        tm = self.dialog.tile_manager
        sw, sh = tm.sensor_w, tm.sensor_h
        if sw is None or sh is None:
            self.log_signal.emit("  ⚠ no sensor size cached, skipping crop")
            return

        extra_w = max(0, sw - tile.width_px)
        extra_h = max(0, sh - tile.height_px)
        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top

        # Resize to native sensor = "no crop" — write zeros and trigger
        # apply so any prior crop is cleared.
        for pv, val in [
            (f"{crop_prefix}:CropLeft",   left),
            (f"{crop_prefix}:CropRight",  right),
            (f"{crop_prefix}:CropTop",    top),
            (f"{crop_prefix}:CropBottom", bottom),
        ]:
            self.dialog._caput(pv, val, timeout=5)
        self.dialog._caput(f"{crop_prefix}:Crop", 1, timeout=5)
        self.log_signal.emit(
            f"  crop L={left} R={right} T={top} B={bottom} "
            f"(tile {tile.width_px}×{tile.height_px} of {sw}×{sh})")

    # ── main loop ──────────────────────────────────────────────────────

    def run(self):
        try:
            tiles: List[Tile] = list(self.dialog.tile_manager.tiles)
            if not tiles:
                self.log_signal.emit("⚠ No tiles to run.")
                return

            crop_prefix   = self.dialog.crop_prefix.text().strip()
            tomo_prefix   = self.dialog.tomoscan_prefix.text().strip()
            motor1_pv     = self.dialog.motor1_pv.text().strip()
            motor2_pv     = self.dialog.motor2_pv.text().strip()
            tile_timeout  = self.dialog.tile_timeout.value()
            motor_tol     = self.dialog.motor_tolerance.value()
            caput_timeout = self.dialog.caput_timeout.value()
            settle        = self.dialog.settle_time.value()

            crop_pvs = (f"{crop_prefix}:CropLeft",
                        f"{crop_prefix}:CropRight",
                        f"{crop_prefix}:CropTop",
                        f"{crop_prefix}:CropBottom")

            self.log_signal.emit("═" * 60)
            self.log_signal.emit(f"RUN TILES — {len(tiles)} tile(s)")
            self.log_signal.emit("═" * 60)

            # Snapshot crop state once (it's the same four PVs every iter).
            crop_backup: Dict[str, str] = {}
            for pv in crop_pvs:
                v = self.dialog._caget(pv)
                if v is not None:
                    crop_backup[pv] = v
            self.log_signal.emit(f"saved crop state: {crop_backup or '(none)'}")

            try:
                for idx, tile in enumerate(tiles, 1):
                    if not self.dialog.scanning:
                        self.log_signal.emit("⚠ stopped by user")
                        break

                    self.log_signal.emit("─" * 60)
                    self.log_signal.emit(
                        f"[tile {idx}/{len(tiles)}] #{tile.id} "
                        f"X={tile.x_mm:.3f} Y={tile.y_mm:.3f} "
                        f"{tile.width_px}×{tile.height_px} "
                        f"overrides={len(tile.overrides)}")

                    # Per-tile override snapshot
                    override_backup: Dict[str, Optional[str]] = {}
                    for k in tile.overrides:
                        override_backup[k] = self.dialog._caget(
                            f"{tomo_prefix}{k}")

                    try:
                        self._apply_crop(crop_prefix, tile)

                        self.log_signal.emit(
                            f"  move {motor1_pv}→{tile.x_mm:.3f}, "
                            f"{motor2_pv}→{tile.y_mm:.3f}")
                        self.dialog._caput(motor1_pv, tile.x_mm,
                                           caput_timeout)
                        self.dialog._caput(motor2_pv, tile.y_mm,
                                           caput_timeout)
                        self.dialog._wait_for_motor(
                            motor1_pv, tile.x_mm, motor_tol, caput_timeout)
                        self.dialog._wait_for_motor(
                            motor2_pv, tile.y_mm, motor_tol, caput_timeout)
                        time.sleep(0.1 if self.dialog.test_mode else settle)

                        for k, v in tile.overrides.items():
                            self.log_signal.emit(
                                f"  override {tomo_prefix}{k} = {v}")
                            self.dialog._caput(f"{tomo_prefix}{k}", v,
                                               timeout=5)

                        start_pv = f"{tomo_prefix}StartScan"
                        self.log_signal.emit(
                            f"  trigger {start_pv} (wait, timeout="
                            f"{tile_timeout:.0f}s)")
                        if self.dialog.test_mode:
                            time.sleep(0.2)
                            self.log_signal.emit(
                                "  [test] StartScan skipped")
                        else:
                            self._active_proc = subprocess.Popen(
                                ['caput', '-w', str(tile_timeout),
                                 start_pv, '1'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )
                            rc = self._active_proc.wait()
                            self._active_proc = None
                            if rc != 0:
                                self.log_signal.emit(
                                    f"  ⚠ StartScan exit code {rc}")
                            else:
                                self.log_signal.emit("  ✓ StartScan done")
                    finally:
                        # Per-tile override restore
                        for pv_suffix, val in override_backup.items():
                            if val is None:
                                continue
                            self.dialog._caput(
                                f"{tomo_prefix}{pv_suffix}", val, timeout=5)
                        if override_backup:
                            self.log_signal.emit(
                                f"  restored {len(override_backup)} override(s)")

                    self.progress_signal.emit(idx, len(tiles))
            finally:
                # Whole-run crop restore (always runs, even on Stop / crash)
                for pv, val in crop_backup.items():
                    self.dialog._caput(pv, val, timeout=5)
                if crop_backup:
                    self.dialog._caput(f"{crop_prefix}:Crop", 1, timeout=5)
                    self.log_signal.emit("restored crop state")

            self.log_signal.emit("═" * 60)
            self.log_signal.emit("RUN TILES finished")
            self.log_signal.emit("═" * 60)

        except Exception as e:
            self.log_signal.emit(f"✗ Tile run failed: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()
