# Mosalign — Motor Scan, Stitched Preview, and Per-Tile Tomoscan

A PyQt5 GUI for 2D motor scanning at the APS 32-ID TXM beamline. Drives a
rigid grid scan, stitches the captured frames into a live mosaic, then
turns that mosaic into an **editable tile overlay**: drag, resize, add, or
delete tiles to choose exactly which positions tomoscan will visit.

## What it does

1. **Grid scan** — steps two motors over an N×M grid and snapshots a
   camera frame at each position into a stitched preview.
2. **Interactive tiles** — when the grid scan finishes, each cell becomes
   a draggable `RectROI` on the preview. Tiles are the source of truth
   for the next step.
3. **Run Tiles** — for each tile, applies a crop (if the tile is smaller
   than the sensor), moves the motors to the tile's position, optionally
   overrides tomoscan parameters (NumAngles, ExposureTime, …), triggers
   `tomoscan StartScan`, and restores the original PV state on completion.

## Features

- **2D motor grid scanning** with live stitched mosaic preview.
- **Editable tile overlay** — drag-to-reposition, resize-to-crop,
  right-click for context menu, double-click for the side panel.
- **Per-tile side panel** — shows the selected tile's X/Y motor positions
  (labels live-bound to the configured motor-PV names), number of
  projections, integration time, crop size. Edits write back to the tile;
  dragging the tile updates the panel.
- **Per-tile tomoscan overrides** — set `NumAngles` and `ExposureTime` in
  the side panel, or any other PV suffix via the "Other overrides…"
  dialog. Empty = use the IOC's current value.
- **Crop integration** — resizing a tile below sensor size writes
  `Crop*` PVs (`CropLeft/Right/Top/Bottom`, then `Crop=1`) before the
  scan, restores them after the full run.
- **Test mode** — no real `caput`/`caget` calls; the per-tile sequence is
  printed to the log so the UI can be exercised offline.
- **Configuration persistence** — scan parameters, IOC prefixes, and the
  full tile list (positions, sizes, overrides) are saved between sessions.

## Installation

```bash
cd /home/beams0/AMITTONE/Software/mosalign
pip install -e .
```

## Usage

### Standalone

```bash
mosalign
```

### From Python

```python
from mosalign import main
main()
```

### Offline smoke test (no beamline required)

Paints a recognizable fake 2×3 mosaic into the preview, seeds the tile
overlay, enables the tile buttons, and forces test mode so no real PVs
are touched:

```bash
python test_interactive_tiles.py
```

Use it to exercise: drag → motor coords update, resize → crop math,
double-click → side panel edits, `+ Tile at Center`, right-click menu,
Run Tiles → per-tile sequence printed to the log.

## UI layout

- **Left panel (tabbed)**
  - **Scan tab** — grid parameters (X/Y start/step/count), runtime
    settings (test mode, pixel size, motor tolerance, caput/tile
    timeouts, run-tomoscan toggle), preview options.
  - **PVs tab** — motor PVs, tomoscan path, image PV, connection status,
    `tomoscan_prefix`, `camera_prefix`, `crop_prefix`.
  - **Below the tabs (always visible)** — Start / Stop / progress bar /
    current position.
- **Right panel**
  - Tile actions row: `+ Tile at Center`, `Run Tiles`.
  - Splitter: stitched canvas (left) | tile side panel (right).
  - Log output (terminal styling).

## Per-tile execution sequence

For each tile in order, the runner:

1. Snapshots the four `Crop*` PVs (only once, at run start).
2. Snapshots the override PVs the tile sets (so they can be restored).
3. Applies the crop (`CropLeft/Right/Top/Bottom` from sensor minus tile
   size, then `Crop=1`).
4. Moves both motors to the tile's `(x_mm, y_mm)`; waits until the
   motors are within tolerance.
5. Applies the per-tile overrides via `caput`.
6. Fires `caput -w {tile_timeout} {tomo_prefix}StartScan 1` and waits.
7. Restores the per-tile override PVs.

After the loop, restores the four `Crop*` PVs to the run-start snapshot
and writes `Crop=1` to commit. The Stop button terminates the active
`caput -w` subprocess; the `finally` blocks still restore PV state.

## Configuration

Settings (including the tile list) are saved to
`~/.pystream/config/motor_scan_config.json` and restored on launch.
Tile IDs persist across sessions; new tiles get fresh IDs even after a
reload.

## Coordinate / flip handling

The preview applies `np.flipud(stitched_image)` at display time. The tile
coordinate mapping inverts Y to account for this:

```
cx = (x_mm - x_start_mm) * 1000 / pixel_size_um
cy = out_h - img_h - (y_mm - y_start_mm) * 1000 / pixel_size_um
```

If a future refactor removes the `flipud`, fix
`TileManager.mm_to_canvas_px` in the same change.

## Requirements

- Python ≥ 3.8
- PyQt5 ≥ 5.15
- numpy ≥ 1.20
- pyqtgraph ≥ 0.12
- EPICS command-line tools (`caput`, `caget`) — not required in test
  mode.

## License

MIT License — Copyright (c) 2025 Alberto Mittone
