# Mosalign - Motor Scan and Alignment Tool

A PyQt5-based GUI for 2D motor scanning with live stitched preview and tomoscan integration.

## Features

- **2D Motor Grid Scanning**: Configure X-Y motor scans with customizable start positions, step sizes, and number of steps
- **Live Stitched Preview**: Real-time mosaic display of captured images during scan
- **Integrated Tomoscan**: Direct Python integration with tomoscan mosaic mode (no external shell scripts)
- **Real-Time Console**: Terminal-style console with live output streaming from tomoscan
- **Test Mode**: Mock PV and camera mode for offline development and testing
- **Auto-Contrast**: Automatic contrast adjustment for optimal image display
- **Configuration Persistence**: Saves and loads scan parameters between sessions

## Installation

```bash
cd /home/beams0/AMITTONE/Software/mosalign
pip install -e .
```

## Usage

### Standalone Application

```bash
mosalign
```

### From Python

```python
from mosalign import main
main()
```

## Configuration

Settings are automatically saved to `~/.pystream/config/motor_scan_config.json` and restored on launch.

## Requirements

- Python >= 3.8
- PyQt5 >= 5.15.0
- numpy >= 1.20.0
- pyqtgraph >= 0.12.0
- EPICS command-line tools (`caput`, `caget`)

## License

MIT License - Copyright (c) 2025 Alberto Mittone
