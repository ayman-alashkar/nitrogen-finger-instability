"""
Nitrogen-Water Interface Finger Instability Analysis

A Python package for detecting and quantifying finger instabilities
at the nitrogen-water interface using spectral analysis.
"""

from .circle_detection import detect_circle, detect_circle_from_path
from .intensity_analysis import extract_intensity_grid
from .spectral_analysis import compute_psd, analyze_averaged_psd

__version__ = "1.0.0"

__all__ = [
    "detect_circle",
    "detect_circle_from_path",
    "extract_intensity_grid",
    "compute_psd",
    "analyze_averaged_psd",
]
