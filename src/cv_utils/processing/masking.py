"""
Image masking operations.

Provides ROI (Region of Interest) masking functionality.
"""

import cv2
import numpy as np

from ..shapes import Polygon


class ROIMask:
    """
    Applies a region of interest mask to images.

    Useful for focusing processing on specific image regions.

    Example:
        >>> from cv_utils.points_shapes import Polygon
        >>> roi = Polygon([(100, 0), (400, 0), (400, 300), (100, 300)])
        >>> mask = ROIMask(polygon=roi)
        >>> masked_image = mask.apply(image)
    """

    def __init__(self, polygon: Polygon):
        """
        Initialize ROI mask.

        Args:
            polygon: Polygon defining the ROI
        """
        self.polygon = polygon

        # Build vertices exactly like legacy code
        self.vertices = np.array(
            [[[int(p.x), int(p.y)] for p in polygon.vertices]], np.int32
        )
        # Cache for mask (created on first apply)
        self._cached_mask = None
        self._cached_shape = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply ROI mask to image."""

        # Create or reuse cached mask
        if self._cached_mask is None or self._cached_shape != image.shape:
            mask = np.zeros_like(image)
            fill_color = (255, 255, 255) if len(image.shape) == 3 else 255
            cv2.fillPoly(mask, self.vertices, fill_color)
            self._cached_mask = mask
            self._cached_shape = image.shape
        else:
            mask = self._cached_mask

        # Apply mask
        masked = cv2.bitwise_and(image, mask)

        return masked

    def scale(self, scale: float) -> None:
        """
        Scale ROI mask by factor.

        Scales the polygon and rebuilds vertices for OpenCV.
        Invalidates cache since dimensions changed.

        Args:
            scale: Scale factor (e.g., 0.5 for half size, 2.0 for double)

        Example:
            >>> roi_mask = ROIMask(polygon)
            >>> roi_mask.scale(0.5)  # Scale down to half size
        """
        # Scale the polygon (in-place)
        self.polygon.scale(scale)

        # Rebuild vertices array for OpenCV
        self.vertices = np.array(
            [[[int(p.x), int(p.y)] for p in self.polygon.vertices]], np.int32
        )

        # Invalidate cache since polygon changed
        self._cached_mask = None
        self._cached_shape = None

    def __str__(self) -> str:
        """Return detailed ROI mask configuration."""
        info = "--- ROI Masking ---\n"
        info += f"Vertices: {len(self.polygon.vertices)}\n"
        info += f"ROI type: {len(self.polygon.vertices)}-corner polygon\n"
        return info

    def __repr__(self) -> str:
        return f"ROIMask(vertices={len(self.polygon.vertices)})"
