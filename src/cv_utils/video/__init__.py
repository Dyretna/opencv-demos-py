"""
Video submodule - Video processing utilities.

Examples:
    from cv_utils.video import (
        VideoProcessor,
        VideoCaptureHandler, VideoWriterHandler
    )
"""

from .video_helpers import VideoCaptureHandler, VideoWriterHandler
from .video_processor import VideoProcessor

__all__ = [
    "ImageProcessor",
    "VideoCaptureHandler",
    "VideoProcessor",
    "VideoWriterHandler",
]