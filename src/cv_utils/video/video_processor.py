"""
Generic video processor.

Provides generic processing framework that works with any detection pipeline.
"""

import datetime as dt
import platform
from pathlib import Path

import cv2

from ..detection.base import Pipeline
from .video_helpers import VideoCaptureHandler, VideoWriterHandler


class VideoProcessor:
    """
    Generic video processor that works with any Pipeline.

    Handles:
    - Video input (file or camera)
    - Frame-by-frame processing through pipeline
    - Optional video output
    - Optional frame callbacks
    - Display with OpenCV

    Example:
        >>> from cv_utils.detection.lane_detection import LaneDetectionPipeline
        >>> pipeline = LaneDetectionPipeline(...)
        >>> processor = VideoProcessor(
        ...     pipeline=pipeline,
        ...     video_source="road.mp4",
        ...     output_title="output"
        ... )
        >>> processor.process()
    """

    def __init__(
        self,
        pipeline: Pipeline,
        video_source: str | Path | int,
        scale: float = 1.0,
        skip_frames: int = 1,
        output_title: str = None,
        output_format: str = ".mp4",
        output_is_color: bool = True,
        display_window: str = "Video Processing",
        show_frame_nr: bool = True,
        show_time: bool = False,
    ):
        """
        Initialize video processor.

        Args:
            pipeline: Detection pipeline to apply to each frame
            video_source: Video file path or camera index
            scale: Scale factor for video processing
            skip_frames: Process every Nth frame (1 = all frames)
            output_title: If provided, save processed video to file
            output_format: Output video format
            output_is_color: Whether output is color or grayscale
            display_window: Window name for display
        """
        self.pipeline = pipeline
        self.video_source = video_source
        self.scale = scale
        self.skip_frames = skip_frames
        self.display_window = display_window

        # platform, versions and time
        self.os = platform.system()
        self.python_version = platform.python_version()
        self.cv2_version = cv2.__version__
        self.timestamp = dt.datetime.now().strftime("%H:%M:%S").replace(":", "_")

        # Setup video capture
        self.cap = VideoCaptureHandler(video_source, scale=scale)
        self.width = self.cap.width
        self.height = self.cap.height
        self.fps = self.cap.fps

        # Setup video writer if output requested
        self.output_title = output_title
        self.writer = None
        if output_title is not None:
            self.writer = VideoWriterHandler(
                title=output_title,
                format=output_format,
                fps=self.fps,
                width=self.width,
                height=self.height,
                isColor=output_is_color,
            )

        # Initialize counters
        self.show_frame_nr = show_frame_nr
        self.show_time = show_time
        self.frames_saved = 0
        self.frame_number = 0
        self.detected_frames = []

    def process(self, show_display: bool = True) -> None:
        """Process video through pipeline, with optional display."""

        try:
            while self.cap.capture.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Video complete. Processed {self.frame_number} frames.")
                    break

                # Frame counting and skipping
                self.frame_number += 1
                if self.frame_number % self.skip_frames != 0:
                    continue

                # Process through pipeline
                result = self.pipeline.process(frame)

                # Check metadata to determine if detection occurred
                metadata = self.pipeline.metadata
                has_detection = False

                # Check based on pipeline type
                if "match_found" in metadata:
                    # TemplateMatchingPipeline
                    has_detection = metadata["match_found"]
                elif "count" in metadata:
                    # CascadeDetectionPipeline or LaneDetectionPipeline
                    has_detection = metadata["count"] > 0

                if has_detection:
                    # Print detection info
                    score = metadata.get("score")
                    if score is not None:
                        print(
                            f"Match found in frame #{self.frame_number}, score: {score:.3f}"
                        )
                    else:
                        count = metadata.get("count", 0)
                        print(
                            f"Detection in frame #{self.frame_number}, count: {count}"
                        )

                # Display if requested
                if self.show_time:
                    self._show_time(result)
                if self.show_frame_nr:
                    self._show_frame(result)
                if show_display:
                    cv2.imshow(self.display_window, result)
                    if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
                        print(f"User interrupted at frame {self.frame_number}")
                        break

                # Write to output file if configured
                if self.writer is not None:
                    self.writer.write(result)
                    self.frames_saved += 1
        finally:
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()

    def _show_time(self, result):
        # Get current time for each frame
        current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time.replace(":", "_")
        cv2.putText(
            img=result,
            text=f"Time: {current_time}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 100, 255),
            thickness=2,
        )

    def _show_frame(self, result):
        cv2.putText(
            img=result,
            text=f"Frame #{self.frame_number}",
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 100, 255),
            thickness=2,
        )

    def __str__(self) -> str:
        """Return detailed video processor configuration."""
        header = "=" * 50 + "\n"
        header += "VIDEO PROCESSOR\n"
        header += "=" * 50 + "\n"

        # System info
        info = "\n--- System ---\n"
        info += f"OS: {self.os} | Python: {self.python_version} | OpenCV: {self.cv2_version}\n"

        # Video settings
        info += "\n--- Video Settings ---\n"
        info += f"Source: {self.video_source}\n"
        info += f"Format: {self.width}x{self.height}@{self.fps}fps | Backend: {self.cap.backend_name}\n"
        info += f"Scale: {self.scale}x | Skip: every {self.skip_frames} frame(s)\n"

        # Pipeline info (delegates to pipeline's __str__)
        info += "\n--- Pipeline ---\n"
        info += str(self.pipeline)

        # Writer info
        if self.output_title is not None:
            info += "\n--- Output ---\n"
            info += f"Saving to: {self.writer.title}\n"
            info += f"Format: {self.writer.fourcc} | Color: {'Yes' if self.writer.isColor else 'No'}\n"

        footer = "\n" + "=" * 50

        return header + info + footer

    def __repr__(self) -> str:
        """Return simple processor representation."""
        return (
            f"VideoProcessor("
            f"source={self.video_source}, "
            f"{self.width}x{self.height}@{self.fps}fps, "
            f"pipeline={self.pipeline.__class__.__name__})"
        )
