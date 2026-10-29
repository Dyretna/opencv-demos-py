"""
Processing submodule - Image processing operations.

Examples:
    from cv_utils.processing import (
        EdgeDetector, ROIMask,
        SmoothingFilter, HistogramEqualizers
    )
"""

from .edges import EdgeDetector
from .filters import SmoothingFilter, smoothing
from .histogram_utils import HistogramEqualizers
from .masking import ROIMask

__all__ = [
    "EdgeDetector",
    "HistogramEqualizers",
    "ROIMask",
    "SmoothingFilter",
    "smoothing",
]