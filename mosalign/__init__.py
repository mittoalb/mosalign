"""
Mosalign - Motor Scan and Alignment Tool
==========================================

A PyQt5-based GUI for 2D motor scanning with live stitched preview and tomoscan integration.
"""

__version__ = "1.0.0"

from .gui import MotorScanDialog, main

__all__ = ['MotorScanDialog', 'main']
