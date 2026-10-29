"""
Drawing submodule - Visualization and drawing utilities.

Examples:
    from cv_utils.drawing import (
        display, display_grid,
        draw_filled_shapes, draw_labeled_box
    )
"""

from .drawing import (draw_filled_shapes,
                      draw_img_center_point,
                      draw_labeled_box,
                      draw_wireframe_shapes)

from .shape_generators import random_regulars


__all__ = [
    "draw_filled_shapes",
    "draw_img_center_point",
    "draw_wireframe_shapes",
    "draw_labeled_box",
    "random_regulars",
]