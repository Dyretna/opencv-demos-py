"""
Lane detection pipeline.

This module combines ROI masking, edge detection, and line detection
into a complete lane detection pipeline.
"""

import cv2
import numpy as np

from ...image_utils import ImageGeometry
from ...processing.edges import EdgeDetector
from ...processing.masking import ROIMask
from ..base import Pipeline
from ..line_detector import LineDetector


class LaneDetectionPipeline(Pipeline):
    """
    Complete lane detection pipeline.

    Orchestrates:
    1. ROI masking (focus on road area)
    2. Edge detection (find edges)
    3. Line detection (find lane lines)
    4. Visualization (draw results)

    Example:
        >>> pipeline = LaneDetectionPipeline(
        ...     roi_mask=ROIMask(...),
        ...     edge_detector=EdgeDetector(...),
        ...     line_detector=LineDetector(...)
        ... )
        >>> result = pipeline.process(frame)
    """

    def __init__(
        self,
        roi_mask: ROIMask = None,
        edge_detector: EdgeDetector = None,
        line_detector: LineDetector = None,
        drawing_params: dict = None,
        blend_params: dict = None,
    ):
        """
        Initialize lane detection pipeline.

        Args:
            roi_mask: ROI masking component (if None, no masking)
            edge_detector: Edge detection component
            line_detector: Line detection component
            drawing_params: Drawing configuration. Keys: 'color' (BGR), 'thickness'
            blend_params: Blending configuration. Keys: 'frame_weight', 'line_weight', 'gamma'
        """
        self.roi_mask = roi_mask
        # detectors can be instances,
        # or be initialized with defaults
        self.edge_detector = edge_detector or EdgeDetector()
        self.line_detector = line_detector or LineDetector()

        # Default drawing parameters
        drawing_defaults = {
            "color": (255, 0, 255),
            "thickness": 3,
        }
        self.drawing_params = {**drawing_defaults, **(drawing_params or {})}
        self._validate_drawing_params()

        # Default blend parameters
        blend_defaults = {
            "frame_weight": 0.7,
            "line_weight": 0.3,
            "gamma": 0.0,
        }
        self.blend_params = {**blend_defaults, **(blend_params or {})}
        self._validate_blend_params()

        # Metadata storage (accessible via metadata property)
        self._lines_count = 0
        self._last_lines = None

    @classmethod
    def from_config(cls, **kwargs):
        """
        Create a new LaneDetectionPipeline from config dict.
        """
        return cls(kwargs)

    def get_config(self):
        """
        Return the current configuration of the pipeline as a dictionary.
        """
        return {
            "param1": getattr(self, "param1", None),
            "param2": getattr(self, "param2", None),
            "param3": getattr(self, "param3", None),
        }

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run full lane detection pipeline.

        Processes frame by frame. The most performance expensive process is
        the edge detector. therefore we have designed a optimization.
        1. Save the bottom_half of current frame - bellow the horizon.
        2. Convert to gray scale.
        3. Apply an already initialized ROI mask:
            five-point polygon adjusted for right side traffic.
        4. Process with edge detector (Canny with optional blur).
        5. Line detection (Hough Lines)
        6. Drawing the Lane-lines

        Returns:
            Processed frame with lanes drawn
        """
        # Step 1: Extract bottom_half and calculate offset
        height = frame.shape[0]
        y_offset = height // 2
        bottom_half = ImageGeometry.get_bottom_half(frame)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply ROI mask to bottom-half grayscale
        if self.roi_mask is not None:
            masked_gray = self.roi_mask.apply(gray)
        else:
            masked_gray = gray

        # Step 4: Detect edges on masked half grayscale
        edges = self.edge_detector.detect(masked_gray)

        # Step 5: Remove ROI boundary artifacts (before Line Detection)
        if self.roi_mask is not None:
            cv2.polylines(
                edges, self.roi_mask.vertices, isClosed=True, color=0, thickness=2
            )

        # Step 6: Detect lines
        lines = self.line_detector.detect(edges)

        # Store metadata
        self._last_lines = lines
        self._lines_count = len(lines) if lines is not None else 0

        # Step 7: Draw lines on FULL frame with y_offset
        processed_frame = self._draw_lines(frame, lines, y_offset)

        return processed_frame

    def _draw_lines(
        self, frame: np.ndarray, lines: np.ndarray | None, y_offset: int
    ) -> np.ndarray:
        """Draw detected lines on frame with blending."""
        # Create blank image for lines
        line_image = np.zeros_like(frame, dtype=np.uint8)

        # Draw each line with y_offset
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    line_image,
                    (x1, y1 + y_offset),
                    (x2, y2 + y_offset),
                    self.drawing_params["color"],
                    self.drawing_params["thickness"],
                )

        # Blend with original frame
        result = cv2.addWeighted(
            frame,
            self.blend_params["frame_weight"],
            line_image,
            self.blend_params["line_weight"],
            self.blend_params["gamma"],
        )

        return result

    @property
    def metadata(self) -> dict[str, int | np.ndarray | None]:
        """
        Get detection metadata from last processed frame.

        Returns:
            Dictionary with keys:
            - 'count': Number of lane lines detected
            - 'lines': Raw line data as numpy array (x1, y1, x2, y2) or None
        """
        return {
            "count": self._lines_count,
            "lines": self._last_lines,
        }

    def __str__(self) -> str:
        """Return detailed pipeline configuration."""
        info = "\n"

        # ROI Configuration
        if self.roi_mask is not None:
            info += str(self.roi_mask)
        else:
            info += "--- ROI Masking ---\n"
            info += "ROI: disabled (full frame)\n"

        info += "\n"

        # Edge Detection
        info += str(self.edge_detector)
        info += "\n"

        # Line Detection
        info += str(self.line_detector)
        info += "\n"

        # Drawing Parameters
        info += "--- Drawing Parameters ---\n"
        info += f"Line color (BGR): {self.drawing_params['color']}\n"
        info += f"Line thickness: {self.drawing_params['thickness']}\n"
        info += "\n"

        # Blending Parameters
        info += "--- Blending Parameters ---\n"
        info += f"Frame blend weight: {self.blend_params['frame_weight']}\n"
        info += f"Line blend weight: {self.blend_params['line_weight']}\n"
        info += f"Gamma: {self.blend_params['gamma']}\n"

        return info

    def __repr__(self) -> str:
        return (
            f"LaneDetectionPipeline(\n"
            f"  roi={self.roi_mask},\n"
            f"  edge={self.edge_detector},\n"
            f"  line={self.line_detector}\n"
            f")"
        )

    def _validate_drawing_params(self):
        """Validate drawing parameters."""
        color = self.drawing_params.get("color")
        if color is not None:
            if not isinstance(color, (tuple, list)) or len(color) != 3:
                raise ValueError(f"Color must be (B, G, R) tuple, got {color}")
            if not all(0 <= c <= 255 for c in color):
                raise ValueError(f"Color values must be 0-255, got {color}")

        thickness = self.drawing_params.get("thickness")
        if thickness is not None and thickness < 1:
            raise ValueError(f"Thickness must be >= 1, got {thickness}")

    def _validate_blend_params(self):
        """Validate blending parameters."""
        frame_weight = self.blend_params.get("frame_weight")
        if frame_weight is not None and not (0 <= frame_weight <= 1):
            raise ValueError(f"Frame weight must be 0-1, got {frame_weight}")

        line_weight = self.blend_params.get("line_weight")
        if line_weight is not None and not (0 <= line_weight <= 1):
            raise ValueError(f"Line weight must be 0-1, got {line_weight}")
