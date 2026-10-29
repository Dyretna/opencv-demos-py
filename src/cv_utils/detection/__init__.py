"""
Detection submodule - All detection-related functionality.

Exposes all detection classes including specialized pipelines.

Examples:
    from cv_utils.detection import (
        LineDetector,
        TemplateMatcher,
        CascadeDetector,
        TemplateDetector,
    )
    from cv_utils.detection.pipelines import (
        CascadeDetectionPipeline,
        TemplateMatchingPipeline,
        LaneDetectionPipeline,
    )
"""

from .base import Detector, Pipeline
from .line_detector import LineDetector
from .matchers import TemplateMatcher
from .cascade_detector import CascadeDetector
from .template_detector import TemplateDetector

# Import pipelines for convenience
from .pipelines import (
    CascadeDetectionPipeline,
    TemplateMatchingPipeline,
    LaneDetectionPipeline,
    YOLOPipeline,
)

__all__ = [
    # Base classes
    "Detector",
    "Pipeline",
    # Detectors
    "LineDetector",
    "TemplateMatcher",
    "CascadeDetector",
    "TemplateDetector",
    # Pipelines
    "CascadeDetectionPipeline",
    "TemplateMatchingPipeline",
    "LaneDetectionPipeline",
    "YOLOpipeline",
]