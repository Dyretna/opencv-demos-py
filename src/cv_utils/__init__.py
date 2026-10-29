"""
cv_utils - Computer Vision Utilities Package

Import Structure:
- Level 1 (Root): Only modules directly in cv_utils/ (color, geometry, io)
- Level 2 (Submodules): Everything else via submodule imports

Examples:
    # Level 1 - Root imports
    from cv_utils import Color, Palette, ImageGeometry, ImageLoader

    # Level 2 - Submodule imports
    from cv_utils.points_shapes import Circle, Point, Rectangle
    from cv_utils.detection import LineDetector, LaneDetectionPipeline
    from cv_utils.processing import EdgeDetector, ROIMask
    from cv_utils.drawing import display, display_grid
    from cv_utils.video import VideoProcessor
"""

# Level 1: Only direct modules (color, geometry, io)
from .color import Color, Palette
from .image_utils import ImageGeometry
from .image_processor import ImageProcessor
from .loaders import ImageLoader, XMLLoader
from .visualisation import display, display_grid

__all__ = [
    "Color",
    "display",
    "display_grid",
    "Palette",
    "ImageGeometry",
    "ImageProcessor",
    "ImageLoader",
    "XMLLoader",
]