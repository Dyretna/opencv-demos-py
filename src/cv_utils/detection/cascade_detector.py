"""
Cascade detector for object detection using Haar Cascades.

Pure detection logic - no drawing, no I/O.
"""

from pathlib import Path

import cv2
import numpy as np

from ..loaders import XMLLoader
from .base import Detector


class CascadeDetector(Detector):
    """
    Pure object detection using OpenCV Haar Cascades.

    Returns raw detection coordinates without any visualization.
    Implements the Detector interface for composability.

    Example:
        >>> import cv2
        >>> from cv_utils.detection.cascade import CascadeDetector
        >>>
        >>> xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        >>> detector = CascadeDetector(xml_path=xml)
        >>> detections = detector.detect(image)  # Returns [(x, y, w, h), ...]
    """

    def __init__(
        self,
        xml_path: str | Path,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        flags: int | None = None,
        min_size: tuple[int, int] | None = None,
        max_size: tuple[int, int] | None = None,
    ):
        """
        Initialize cascade detector.

        Args:
            xml_path: Path to Haar Cascade XML file
            scale_factor: How much image size is reduced at each scale (default: 1.1, must be > 1)
            min_neighbors: How many neighbors each candidate needs (default: 5)
            min_size: Minimum object size (width, height) in pixels
            max_size: Maximum object size (width, height) in pixels
        """
        # Validate and load XML path using XMLLoader
        self.xml_path = XMLLoader.load(xml_path)
        self.cascade = cv2.CascadeClassifier(str(self.xml_path))

        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from: {self.xml_path}")

        # Validate scale_factor
        if scale_factor <= 1.0:
            raise ValueError(f"scale_factor must be > 1.0, got {scale_factor}")

        # Detection parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.flags = flags
        self.min_size = min_size
        self.max_size = max_size

        # interface for pipeline
        self.kwargs = {
            "scale_factor": scale_factor,
            "min_neighbors": min_neighbors,
            "flags": flags,
            "min_size": min_size,
            "max_size": max_size,
        }

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects in image.

        Args:
            image: Input image in BGR or RGB format

        Returns:
            Array of detections with shape (N, 4) where each row is [x, y, width, height]
            Returns empty array if no objects detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect objects
        objects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=self.flags,
            minSize=self.min_size,
            maxSize=self.max_size,
        )

        return objects

    def __str__(self) -> str:
        """Return detector configuration as string."""
        info = "--- Cascade Detector ---\n"
        info += f"XML: {self.xml_path.name}\n"
        info += f"Scale factor: {self.scale_factor}\n"
        info += f"Min neighbors: {self.min_neighbors}\n"

        if self.min_size:
            info += f"Min object size: {self.min_size[0]}x{self.min_size[1]}\n"
        else:
            info += "Min object size: None\n"

        if self.max_size:
            info += f"Max object size: {self.max_size[0]}x{self.max_size[1]}\n"
        else:
            info += "Max object size: None\n"

        return info

    def __repr__(self) -> str:
        return (
            f"CascadeDetector("
            f"xml_path={self.xml_path}, "
            f"scale_factor={self.scale_factor}, "
            f"min_neighbors={self.min_neighbors})"
        )
