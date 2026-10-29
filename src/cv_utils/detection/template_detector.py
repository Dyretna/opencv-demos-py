"""
Template detector for finding patterns in images.

Pure detection logic - no drawing, no I/O.
"""

from pathlib import Path

import numpy as np

from ..loaders import ImageLoader
from .base import Detector
from .matchers import TemplateMatcher


class TemplateDetector(Detector):
    """
    Pure template matching detector using OpenCV matchTemplate.

    Returns match location without any visualization.
    Implements the Detector interface for composability.

    Example:
        >>> from cv_utils.detection.template import TemplateDetector
        >>>
        >>> detector = TemplateDetector(
        ...     template="logo.png",
        ...     method="cv2.TM_CCOEFF_NORMED",
        ...     threshold=0.8
        ... )
        >>> match = detector.detect(image)  # Returns (top_left, bottom_right) or None
    """

    def __init__(
        self,
        template: np.ndarray | str | Path,
        method: str = "cv2.TM_CCOEFF_NORMED",
        threshold: float | None = None,
        scale: float = 1.0,
    ):
        """
        Initialize template detector.

        Args:
            template: Template image as numpy array or path to image file
            method: OpenCV template matching method. Options:
                - "cv2.TM_CCOEFF"
                - "cv2.TM_CCOEFF_NORMED" (default)
                - "cv2.TM_CCORR"
                - "cv2.TM_CCORR_NORMED"
                - "cv2.TM_SQDIFF"
                - "cv2.TM_SQDIFF_NORMED"
            threshold: Match quality threshold. If None, uses method-specific default
            scale: Scale factor for template (default: 1.0). Use same scale as video/image.
        """
        # Load template if path given
        if isinstance(template, (str, Path)):
            self.template_path = (
                Path(template) if isinstance(template, str) else template
            )
            self.template = ImageLoader.load(self.template_path, "rgb")
        else:
            self.template_path = None
            self.template = template

        # Apply scaling to template if needed
        self.scale = scale
        if scale != 1.0:
            import cv2

            new_width = int(self.template.shape[1] * scale)
            new_height = int(self.template.shape[0] * scale)
            self.template = cv2.resize(self.template, (new_width, new_height))

        # Initialize matcher
        self.matcher = TemplateMatcher()
        self.method = method
        self.threshold = threshold

        # Validate method
        if method not in self.matcher.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.matcher.SUPPORTED_METHODS}")

    def detect(
        self, image: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Detect template in image. Returns coordinates if match."""
        # Match template
        self.matcher.match(image, self.template, self.method)

        # Check if match is good
        if self.matcher.is_good_match(self.threshold):
            return (self.matcher.top_left, self.matcher.bottom_right)

        return None

    def get_match_score(self) -> float:
        """Get the match quality score from last detection."""
        return self.matcher.get_match_score()

    def __str__(self) -> str:
        """Return detector configuration as string."""
        info = "--- Template Detector ---\n"

        if self.template_path:
            info += f"Template: {self.template_path.name}\n"

        h, w = self.template.shape[:2]
        info += f"Template size: {w}x{h}\n"

        if self.scale != 1.0:
            info += f"Scale: {self.scale}x (scaled to {w}x{h})\n"

        method_short = self.method.replace("cv2.", "")
        info += f"Method: {method_short}\n"
        info += f"Threshold: {self.threshold}\n"

        return info

    def __repr__(self) -> str:
        template_info = (
            f"template_path={self.template_path}"
            if self.template_path
            else "template=<array>"
        )
        return (
            f"TemplateDetector("
            f"{template_info}, "
            f"method={self.method}, "
            f"threshold={self.threshold})"
        )
