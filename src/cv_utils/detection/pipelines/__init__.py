"""
Detection pipelines.

Pipelines combine detectors with visualization and orchestration logic.
"""

from .cascade_pipeline import CascadeDetectionPipeline
from .template_pipeline import TemplateMatchingPipeline
from .lane_pipeline import LaneDetectionPipeline
from .yolo_pipeline import YOLOPipeline

__all__ = [
    "CascadeDetectionPipeline",
    "LaneDetectionPipeline",
    "TemplateMatchingPipeline",
    "YOLOPipeline",
]
