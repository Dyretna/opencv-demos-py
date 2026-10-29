"""
Line detection operations.

Provides line detection using Hough Transform.

Note: Returns raw numpy arrays from cv2.HoughLinesP for performance.
      Use cv_utils.points_shapes.Line for geometric line operations.
"""

from typing import Optional

import cv2
import numpy as np

from .base import Detector


class LineDetector(Detector):
    """
    Detects lines in edge images using Hough Line Transform.

    Uses HoughLinesP (Probabilistic Hough Transform) which is more
    efficient and returns line segments directly as numpy arrays.

    Example:
        >>> detector = LineDetector(
        ...     threshold=150,
        ...     min_line_length=40,
        ...     max_line_gap=20
        ... )
        >>> lines = detector.detect(edge_image)
        >>> if lines is not None:
        ...     for line in lines:
        ...         x1, y1, x2, y2 = line[0]
        ...         cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    """

    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 100,
        min_line_length: int = 20,
        max_line_gap: int = 500,
    ):
        """
        Initialize line detector.

        Args:
            rho: Distance resolution in pixels
            theta: Angle resolution in radians (default: 1 degree)
            threshold: Minimum votes to detect a line
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap between line segments to connect them

        Raises:
            ValueError: If parameters are invalid
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")

        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")

        if min_line_length < 0:
            raise ValueError(f"min_line_length must be >= 0, got {min_line_length}")

        if max_line_gap < 0:
            raise ValueError(f"max_line_gap must be >= 0, got {max_line_gap}")

        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines in an edge image.

        Args:
            edges: Binary edge image (single channel, typically from Canny)

        Returns:
            Numpy array of shape (N, 1, 4) where each line is [x1, y1, x2, y2],
            or None if no lines found. Format is directly from cv2.HoughLinesP,
            ready for drawing without conversion overhead.

        Example:
            >>> lines = detector.detect(edges)
            >>> if lines is not None:
            ...     for line in lines:
            ...         x1, y1, x2, y2 = line[0]
            ...         cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        """
        return cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

    def __str__(self) -> str:
        """Return detailed line detector configuration."""
        info = "--- Line Detection (Hough Transform) ---\n"
        info += f"Rho: {self.rho}\n"
        info += f"Theta: {self.theta:.4f} rad\n"
        info += f"Threshold: {self.threshold}\n"
        info += f"Min line length: {self.min_line_length}\n"
        info += f"Max line gap: {self.max_line_gap}\n"
        return info

    def __repr__(self) -> str:
        return (
            f"LineDetector("
            f"threshold={self.threshold}, "
            f"min_len={self.min_line_length}, "
            f"max_gap={self.max_line_gap})"
        )
