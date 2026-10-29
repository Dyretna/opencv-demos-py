"""
Template matching pipeline for finding patterns in images.

Combines TemplateDetector with visualization logic.
"""

from pathlib import Path

import numpy as np

from ...drawing import draw_labeled_box
from ..base import Pipeline
from ..template_detector import TemplateDetector


class TemplateMatchingPipeline(Pipeline):
    """
    Template matching pipeline using OpenCV matchTemplate.

    Combines TemplateDetector with drawing logic for complete pipeline.
    This pipeline finds occurrences of a template image within a larger image
    and draws rectangles around matches that exceed a quality threshold.

    Example:
        >>> from cv_utils import ImageProcessor
        >>> from cv_utils.detection.template import TemplateDetector, TemplateMatchingPipeline
        >>>
        >>> # Create detector
        >>> detector = TemplateDetector(
        ...     template="logo.png",
        ...     method="cv2.TM_CCOEFF_NORMED",
        ...     threshold=0.8
        ... )
        >>>
        >>> # Create pipeline with detector
        >>> pipeline = TemplateMatchingPipeline(
        ...     detector=detector,
        ...     drawing_params={'color': (0, 255, 0)}
        ... )
        >>>
        >>> # Use with ImageProcessor
        >>> processor = ImageProcessor(img_src="screenshot.jpg", pipeline=pipeline)
        >>> result = processor.process()
        >>>
        >>> # Or with VideoProcessor
        >>> from cv_utils.video import VideoProcessor
        >>> video_proc = VideoProcessor(
        ...     pipeline=pipeline,
        ...     video_source="demo.mp4"
        ... )
        >>> video_proc.process()
    """

    def __init__(
        self,
        detector: TemplateDetector | None = None,
        template: np.ndarray | str | Path | None = None,
        drawing_params: dict | None = None,
        show_score: bool = False,
        **detector_kwargs,
    ):
        """
        Initialize template matching pipeline.

        Args:
            detector: TemplateDetector instance. If None, creates one from template
            template: Template image or path (only used if detector is None)
            drawing_params: Drawing configuration dict with keys:
                - 'color': BGR tuple (default: (0, 255, 0))
                - 'thickness': Line thickness (default: 2)
                - 'text_color': Text color BGR tuple (default: (255, 255, 255))
                - 'font_scale': Text size (default: 0.6)
            show_score: Whether to show match score on image (default: False)
            **detector_kwargs: Additional args passed to TemplateDetector if creating one
                (method, threshold)
        """
        # Create or use provided detector
        if detector is None:
            if template is None:
                raise ValueError("Must provide either 'detector' or 'template'")
            self.detector = TemplateDetector(template=template, **detector_kwargs)
        else:
            self.detector = detector

        self.show_score = show_score

        # Drawing parameters with defaults
        drawing_defaults = {
            "color": (0, 255, 0),  # Green in BGR
            "thickness": 2,
            "text_color": (255, 255, 255),  # White
            "font_scale": 0.6,
        }
        self.drawing_params = {**drawing_defaults, **(drawing_params or {})}

        # Metadata storage (accessible via metadata property)
        self._match_found = False
        self._match_score = None
        self._match_location = None

    def get_config(self):
        """
        Return the current configuration of the pipeline as a dictionary.
        """
        return {
            "template_path": getattr(self.detector, "template_path", None),
            "match_method": getattr(self.detector, "method", "cv2.TM_CCOEFF_NORMED"),
            "threshold": getattr(self.detector, "threshold", 0.8),
        }

    @classmethod
    def from_config(cls, kwargs):
        return cls(kwargs)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run template matching and draw rectangle if match found."""
        # 1. Detect template (delegated to detector)
        match = self.detector.detect(image)

        # Store metadata
        self._match_found = match is not None
        self._match_score = self.detector.get_match_score()
        self._match_location = match

        # 2. Draw on copy of original (pipeline's responsibility)
        result = image.copy()

        if match is not None:
            top_left, bottom_right = match
            # Optionally show match score
            if self.show_score:
                score_text = f"Score: {self._match_score:.3f}"
            else:
                score_text = ""

            draw_labeled_box(
                img=result,
                tl=top_left,
                br=bottom_right,
                text=score_text,
                color=self.drawing_params["color"],
            )

        return result

    @property
    def metadata(self) -> dict[str, bool | float | tuple | None]:
        """
        Get detection metadata from last processed frame.

        Returns:
            Dictionary with keys:
            - 'match_found': Whether a match was found (bool)
            - 'score': Match quality score (0-1) or None
            - 'location': Tuple of (top_left, bottom_right) or None
        """
        return {
            "match_found": self._match_found,
            "score": self._match_score,
            "location": self._match_location,
        }

    def __str__(self) -> str:
        """Return pipeline configuration as string."""
        info = "--- Template Matching ---\n"

        # Template info
        if self.detector.template_path:
            info += f"Template: {self.detector.template_path.name}\n"
            h, w = self.detector.template.shape[:2]
            info += f"Size: {w}x{h}"
            if self.detector.scale != 1.0:
                info += f" (scaled {self.detector.scale}x)"
            info += "\n"

        # Method info
        method_short = self.detector.method.replace("cv2.", "")
        info += f"Method: {method_short}\n"
        info += f"Threshold: {self.detector.threshold}\n"

        # Status section
        info += "\n--- Detection Status ---\n"
        if self._match_score is not None:
            info += (
                f"Last result: {'Match found' if self._match_found else 'No match'}\n"
            )
            info += f"Match score: {self._match_score:.3f}\n"
            if self._match_location:
                top_left, bottom_right = self._match_location
                info += f"Match location: {top_left} to {bottom_right}\n"
        else:
            info += "Not yet run\n"

        # Visualization section
        info += "\n--- Visualization ---\n"
        info += f"Show score on image: {'Yes' if self.show_score else 'No'}\n"
        color_name = (
            "Green" if self.drawing_params["color"] == (0, 255, 0) else "Custom"
        )
        info += f"Box color: {color_name} {self.drawing_params['color']}\n"
        info += f"Box thickness: {self.drawing_params['thickness']}\n"

        return info

    def __repr__(self) -> str:
        return f"TemplateMatchingPipeline(detector={repr(self.detector)})"
