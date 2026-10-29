"""
Cascade detection pipeline for object detection using Haar Cascades.

Combines CascadeDetector with visualization logic.
"""

from pathlib import Path

import cv2
import numpy as np

from ..base import Pipeline
from ..cascade_detector import CascadeDetector


class CascadeDetectionPipeline(Pipeline):
    """
    Object detection pipeline using OpenCV Haar Cascades.

    Combines CascadeDetector with drawing logic for complete pipeline.
    This pipeline detects objects (faces, eyes, cats, etc.) and draws
    rectangles around them.

    Example:
        >>> import cv2
        >>> from cv_utils import ImageProcessor
        >>> from cv_utils.detection.cascade import CascadeDetector, CascadeDetectionPipeline
        >>>
        >>> # Create detector
        >>> xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        >>> detector = CascadeDetector(xml_path=xml, min_neighbors=5)
        >>>
        >>> # Create pipeline with detector
        >>> pipeline = CascadeDetectionPipeline(
        ...     detector=detector,
        ...     drawing_params={'color': (0, 255, 0), 'thickness': 2}
        ... )
        >>>
        >>> # Use with ImageProcessor
        >>> processor = ImageProcessor(img_src="faces.jpg", pipeline=pipeline)
        >>> result = processor.process()
        >>>
        >>> # Or with VideoProcessor
        >>> from cv_utils.video import VideoProcessor
        >>> video_proc = VideoProcessor(
        ...     pipeline=pipeline,
        ...     video_source="meeting.mp4"
        ... )
        >>> video_proc.process()
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        drawing_params: dict | None = None,
        show_count: bool = True,
        **cascade_kwargs,
    ):
        """
        Initialize cascade detection pipeline.

        Args:
            detector: CascadeDetector instance. If None, creates one from xml_path
            xml_path: Path to Haar Cascade XML (only used if detector is None)
            drawing_params: Drawing configuration dict with keys:
                - 'color': BGR tuple (default: (255, 255, 0))
                - 'thickness': Line thickness (default: 3)

                - 'text_color': Text color BGR tuple (default: (0, 200, 100))
                - 'font_scale': Text size (default: 1.0)
            show_count: Whether to show object count on image (default: True)
            **detector_kwargs: Additional args passed to CascadeDetector if creating one
                (scale_factor, min_neighbors, min_size, max_size)
        """
        # Create or use provided detector

        self.detector = CascadeDetector(xml_path=xml_path, **cascade_kwargs)

        # Drawing parameters with defaults
        drawing_defaults = {
            "color": (255, 255, 0),  # Yellow in BGR
            "thickness": 3,
            "text_color": (0, 200, 100),
            "font_scale": 1.0,
        }
        self.drawing_params = {**drawing_defaults, **(drawing_params or {})}
        self.show_count = show_count

        # Metadata storage (accessible via metadata property)
        self._objects_count = 0
        self._last_detections = []

    @classmethod
    def from_config(cls, config):
        """
        Create a new CascadeDetectionPipeline from config dict.
        """
        return cls(
            xml_path=config.get("xml_path"),
            drawing_params=config.get("drawing_params", {}),
            show_count=config.get("show_count", True),
            **config.get("cascade_kwargs", {}),
        )

    def get_config(self):
        """
        Return the current configuration of the pipeline as a dictionary.
        """
        return {
            "xml_path": self.detector.xml_path,
            "drawing_params": self.drawing_params,
            "show_count": self.show_count,
            "cascade_kwargs": self.detector.kwargs,
        }

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run cascade detection and draw rectangles.

        Metadata available after processing via .metadata property:
            - count: Number of objects detected
            - detections: List of (x, y, w, h) tuples

        Returns:
            Processed image with detections drawn
        """
        # 1. Detect objects
        objects = self.detector.detect(image)

        # Store metadata
        self._objects_count = len(objects)
        self._last_detections = objects

        # 2. Draw on copy of original (pipeline's responsibility)
        result = image.copy()

        # Draw rectangles around detected objects
        for x, y, w, h in objects:
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                self.drawing_params["color"],
                self.drawing_params["thickness"],
            )
        # Optionally show count on image
        if self.show_count:
            count_text = f"Objects: {self._objects_count}"
            cv2.putText(
                result,
                count_text,
                (10, 40),  # Top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                self.drawing_params["font_scale"],
                self.drawing_params["text_color"],
                2,
                cv2.LINE_AA,
            )

        return result

    @property
    def metadata(self) -> dict[str, int | list]:
        """
        Get detection metadata from last processed frame.

        Returns:
            Dictionary with keys:
            - 'count': Number of objects detected
            - 'detections': List of (x, y, w, h) bounding boxes
        """
        return {
            "count": self._objects_count,
            "detections": self._last_detections,
        }

    def __str__(self) -> str:
        """Return pipeline configuration as string."""
        info = "--- Cascade Detection ---\n"

        # Detector info
        info += f"Cascade: {self.detector.xml_path.name}\n"
        info += f"Scale factor: {self.detector.scale_factor}\n"
        info += f"Min neighbors: {self.detector.min_neighbors}\n"

        if self.detector.min_size:
            info += (
                f"Min size: {self.detector.min_size[0]}x{self.detector.min_size[1]}\n"
            )
        else:
            info += "Min size: None\n"

        if self.detector.max_size:
            info += (
                f"Max size: {self.detector.max_size[0]}x{self.detector.max_size[1]}\n"
            )
        else:
            info += "Max size: None\n"

        # Status section
        info += "\n--- Detection Status ---\n"
        if self._objects_count > 0:
            info += f"Last result: {self._objects_count} object(s) detected\n"
        else:
            info += "Last result: No objects detected\n"

        # Visualization section
        info += "\n--- Visualization ---\n"
        info += f"Show count on image: {'Yes' if self.show_count else 'No'}\n"
        info += f"Box color: {self.drawing_params['color']}\n"
        info += f"Box thickness: {self.drawing_params['thickness']}\n"

        return info
