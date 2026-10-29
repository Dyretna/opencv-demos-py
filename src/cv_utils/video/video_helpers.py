# std
import datetime as dt
import platform
from pathlib import Path

# 3rd party
import cv2


class VideoCaptureHandler:
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(self, source: str | Path | int, scale: float = 1.0):
        """
        Initialize video capture from file or camera.

        Args:
            source: Either a file path (str/Path) or camera index (int)
                   - Path/str: video file path
                   - int: camera index (0 for default camera)
            scale: Scale factor for frames (default 1.0 = no scaling)
        """
        if isinstance(source, int):
            # Camera capture
            self.capture = cv2.VideoCapture(source)
            self.source_type = "camera"
        else:
            # File capture - validate then load
            validated_path = self._validate_video_path(source)
            self.capture = cv2.VideoCapture(str(validated_path))
            self.source_type = "file"

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self.backend_name = self.capture.getBackendName()
        self.original_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        # Scale settings
        self.scale = scale
        self.width = int(self.original_width * scale)
        self.height = int(self.original_height * scale)

    def read(self):
        """
        Read the next frame from the video, scaled if scale != 1.0.

        Returns:
            tuple: (ret, frame) where:
                - ret (bool):
                    True if frame was successfully read, False otherwise
                - frame (numpy.ndarray):
                    The frame image (scaled if needed), or None if ret is False
        """
        ret, frame = self.capture.read()
        if ret and self.scale != 1.0:
            frame = cv2.resize(frame, (self.width, self.height))
        return ret, frame

    def release(self):
        self.capture.release()

    @staticmethod
    def _validate_video_path(path):
        """Validate video file path and format."""
        file_path = Path(path) if isinstance(path, str) else path

        if not isinstance(file_path, Path):
            raise TypeError("Path must be a string or pathlib.Path object")

        if not file_path.exists():
            raise ValueError(f"Video file does not exist: {path}")

        if not file_path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path}")

        if file_path.suffix.lower() not in VideoCaptureHandler.SUPPORTED_VIDEO_FORMATS:
            raise TypeError(
                f"Unsupported video format: {file_path.suffix}\n"
                f"Supported formats: {VideoCaptureHandler.SUPPORTED_VIDEO_FORMATS}"
            )

        return file_path


class VideoWriterHandler:
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(self, title, format, fps, width, height, isColor=False):
        """
        Create a video writer with specified parameters.

        Args:
            name: Base title (without extension)
            format: File extension (e.g., '.avi', '.mp4')
            fps: Frames per second
            width: Video width in pixels
            height: Video height in pixels
        """
        # Validate format parameter
        if not format.startswith("."):
            raise ValueError(
                f"Format must start with a dot (e.g., '.mp4', '.avi'). Got: '{format}'"
            )

        if format.lower() not in VideoWriterHandler.SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: {format}\n"
                f"Supported formats: {VideoWriterHandler.SUPPORTED_VIDEO_FORMATS}"
            )

        # dynamically set the codec
        self.os = platform.system()
        if self.os == "Windows":
            self.fourcc = "VIDX"
        else:
            self.fourcc = "XVID"

        # title with timestamp
        timestamp = dt.datetime.now().strftime("%H:%M:%S").replace(":", "_")
        self.title = f"{title}_{timestamp}{format}"

        # store video properties
        self.fps = fps
        self.width = width
        self.height = height
        self.isColor = isColor

        # create writer
        self.writer = cv2.VideoWriter(
            self.title,
            cv2.VideoWriter_fourcc(*self.fourcc),
            fps,
            (width, height),
            isColor=self.isColor,
        )

    def write(self, frame):
        """Write a frame to the video"""
        self.writer.write(frame)

    def release(self):
        """Release the video writer"""
        self.writer.release()
