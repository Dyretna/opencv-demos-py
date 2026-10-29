from abc import ABC, abstractmethod
from pathlib import Path

from cv2 import COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_BGR2RGB, cvtColor, imread


class Loader(ABC):
    SUPPORTED_TYPES = []  # Must be overridden in subclasses

    @staticmethod
    def _validate_path(path):
        file_path = Path(path) if isinstance(path, str) else path
        if not isinstance(file_path, Path):
            raise TypeError("Loader accepts instances of 'pathlib.Path' or 'str'")

        if not file_path.exists():
            raise ValueError("Path to image is not a valid path")
        if not file_path.is_file():
            raise FileNotFoundError("File not found - check path")

        return file_path

    @staticmethod
    def _validate_type(path, supported_types):
        """Shared validation logic for file type checking."""
        if path.suffix.lower() not in supported_types:
            raise TypeError(
                f"Unsupported file format: {path.suffix}\n"
                f"Supported formats: {supported_types}"
            )

    @staticmethod
    @abstractmethod
    def load(path, **kwargs):
        """Must be implemented in subclass."""
        pass


class ImageLoader(Loader):
    SUPPORTED_TYPES = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

    @staticmethod
    def load(path: Path, cmap=None):
        """Validates path and image, then loads the image."""
        try:
            file_path = Loader._validate_path(path)
            Loader._validate_type(file_path, ImageLoader.SUPPORTED_TYPES)
            bgr_img = imread(str(file_path))
        except Exception as e:
            print(f"caught an exception: \n{e}")
            raise
        match cmap:
            case None:
                return bgr_img
            case "gray" | "grey" | "gs":
                return cvtColor(bgr_img, COLOR_BGR2GRAY)
            case "hsv":
                return cvtColor(bgr_img, COLOR_BGR2HSV)
            case "rgb":
                return cvtColor(bgr_img, COLOR_BGR2RGB)
            case _:
                raise ValueError("cmap not available")

    @staticmethod
    def load_bgr_gs_hsv_rgb(path: Path):
        """Validates path and image, then loads the image."""
        try:
            file_path = Loader._validate_path(path)
        except Exception as e:
            print(f"caught an exception: \n{e}")
            raise

        bgr = ImageLoader.load(file_path, None)
        gs = ImageLoader.load(file_path, "gs")
        hsv = ImageLoader.load(file_path, "hsv")
        rgb = ImageLoader.load(file_path, "rgb")
        return [bgr, gs, hsv, rgb]


class XMLLoader(Loader):
    SUPPORTED_TYPES = [".xml"]

    @staticmethod
    def load(path):
        """Validates path and XML, then returns the path."""
        try:
            file_path = Loader._validate_path(path)
            Loader._validate_type(file_path, XMLLoader.SUPPORTED_TYPES)
            return file_path
        except Exception as e:
            print(f"caught an exception: \n{e}")
