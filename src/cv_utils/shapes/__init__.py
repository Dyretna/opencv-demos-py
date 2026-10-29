"""
Points and Shapes submodule - Geometric primitives.

Examples:
    from cv_utils.points_shapes import (
        Point, Circle, Rectangle, Polygon,
        RegularHexagon, RegularOctagon
    )
"""

from .circle import Circle
from .lines import Line, Polyline
from .point import Point
from .polygon import Polygon
from .rectangle import Rectangle
from .regular_shapes import (
    RegularHeptagon,
    RegularHexagon,
    RegularOctagon,
    RegularPentagon,
    RegularShape,
    RegularSquare,
    RegularTriangle
)

__all__ = [
    "Circle",
    "Line",
    "Point",
    "Polygon",
    "Polyline",
    "Rectangle",
    "RegularHeptagon",
    "RegularHexagon",
    "RegularOctagon",
    "RegularPentagon",
    "RegularShape",
    "RegularSquare",
    "RegularTriangle",
]