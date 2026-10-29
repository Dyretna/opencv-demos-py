"""
Generic image processor.

Provides generic processing framework that works with any detection pipeline.
"""

from pathlib import Path

import cv2
import numpy as np

from .detection.base import Pipeline
from .drawing import draw_wireframe_shapes
from .image_utils import ImageGeometry

# local
from .loaders import ImageLoader
from .shapes import Rectangle
from .visualisation import display, display_grid


class ImageProcessor:
    """
    Generic image processor that works with any Pipeline.

    Useful for:
    - Testing pipelines on static images
    - Parameter tuning before video processing
    - Batch processing of images
    - Creating visualizations

    Example:
        >>> from cv_utils.detection.lane_detection import LaneDetectionPipeline
        >>> pipeline = LaneDetectionPipeline(...)
        >>> processor = ImageProcessor(pipeline=pipeline, image_source="road.jpg")
        >>> result = processor.process()
    """

    def __init__(
        self,
        img_src: str | Path,
        pipeline: Pipeline | None = None,
    ):
        """
        Initialize image processor.

        Args:
            pipeline: Detection pipeline to apply
            img_src: Path to image file
            scale: Scale factor for image (1.0 = no scaling)
        """
        self._validate_img_src(img_src)

        # Store path only
        self.img_path = Path(img_src) if isinstance(img_src, str) else img_src
        self.pipeline = pipeline
        self.result = None

    # Metadata property
    @property
    def metadata(self):
        """Get metadata from pipeline (available after process())."""
        if self.pipeline is None:
            return {}
        return self.pipeline.metadata

    def print_metadata(self) -> None:
        """Pretty print metadata from pipeline."""
        meta = self.metadata

        if not meta:
            print(
                "No metadata available (pipeline not processed yet or no pipeline set)"
            )
            return

        print("--- PIPELINE METADATA --- ")
        for key, value in meta.items():
            # Format key nicely
            formatted_key = key.replace("_", " ").title()

            # Handle different value types
            if isinstance(value, (list, np.ndarray)) and value is not None:
                if isinstance(value, np.ndarray):
                    print(f"{formatted_key}: {len(value)} items")
                else:
                    print(f"{formatted_key}: {len(value)} items")
            elif value is None:
                print(f"{formatted_key}: None")
            else:
                print(f"{formatted_key}: {value}")

    # --- "convert" with loaders ---

    def bgr(self) -> np.ndarray:
        """Load image as BGR and return as BGR."""
        return ImageLoader.load(self.img_path)

    def bgr_2_rgb(self) -> np.ndarray:
        """Load image as BGR and convert to RGB."""
        return ImageLoader.load(self.img_path, "rgb")

    def bgr_2_gray(self) -> np.ndarray:
        """Load image as BGR and convert to grayscale."""
        return ImageLoader.load(self.img_path, "gray")

    def bgr_2_hsv(self) -> np.ndarray:
        """Load image as BGR and convert to HSV."""
        return ImageLoader.load(self.img_path, "hsv")

    def get_bgr_gs_hsv_rgb(self):
        "Loads and returns BGR, GS, HSV, and RGB versions"
        return ImageLoader.load_bgr_gs_hsv_rgb(self.img_path)

    # --- misc image helpers ---

    def crop(
        self,
        topleft: tuple,
        width: int,
        height: int,
        output_path: Path | str | None = None,
    ):
        """croping image using slicing."""
        # Load the image
        img = ImageLoader.load(self.img_path, cmap=None)

        # Define the region of interest (ROI)
        x_start, y_start = topleft
        x_end, y_end = topleft[0] + width, topleft[1] + height

        # Crop the image using slicing
        cropped_img = img[y_start:y_end, x_start:x_end]

        if output_path is not None:
            # TODO: add validation
            # imwrite always handles BGR, and converts to RGB
            cv2.imwrite(output_path, img)

        # return RGB version
        return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    def find_center(self, output_path: Path | str | None = None) -> np.ndarray:
        """Get the center point of an image. Optional output"""
        img = ImageLoader.load(self.img_path, cmap=None)
        center = ImageGeometry.get_center(img)

        cv2.circle(
            img,
            center,
            radius=5,
            color=(255, 0, 0),  # red
            thickness=-1,
        )
        if output_path is not None:
            # TODO: add validation
            # imwrite always handles BGR, and converts to RGB
            cv2.imwrite(output_path, img)

        # return RGB version
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def frame_it(
        self, color=(255, 0, 0), thickness=20, output_path: Path | str | None = None
    ) -> np.ndarray:
        """Returns frame around RGB version of image. Optional output"""
        img = self.bgr_2_rgb()
        tl, _, br, _ = ImageGeometry.get_corners(img)
        rect = Rectangle.create_from_corners(tl, br)
        img = draw_wireframe_shapes(
            img,
            shapes=[rect],
            colors=[color],
            thickness=thickness * 2,  # half is outside!
        )
        # optional save
        if output_path is not None:
            # TODO: add validation
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr_img)
        return img

    def resize_to_rgb(self, scale=0.5) -> np.ndarray:
        """Load image, convert to RGB and resize. default scale = 0.5"""
        rgb_img = self.bgr_2_rgb()
        new_size = (int(rgb_img.shape[1] * scale), int(rgb_img.shape[0] * scale))
        return cv2.resize(rgb_img, new_size)

    # --- display and output ---

    def show(self, cmap=None) -> None:
        """Display rgb image."""
        img = self.bgr_2_rgb()
        display(img, cmap)

    def show_bgr_gs_hsv_rgb(self) -> None:
        """displays BGR, GS, HSV and RGB"""
        img_list = ImageLoader.load_bgr_gs_hsv_rgb(self.img_path)
        display_grid(img_list, "gray")

    def original_writer(self, out_path: str | Path):
        """saves the original image in rgb."""
        # imwrite always handles BGR, and converts to RGB
        cv2.imwrite(out_path, ImageLoader.load(self.img_path))

    def result_writer(self, out_path: str | Path):
        """saves the processing result in rgb."""
        cv2.imwrite(out_path, self.result)

    # +----------------------+
    # |   Process pipeline   |
    # +----------------------+

    def process(
        self,
        scale: float = 1.0,
        show_visualization: bool = True,
    ) -> np.ndarray:
        """Process image through pipeline."""

        # Load image for processing
        image = ImageLoader.load(self.img_path, "rgb")

        # Apply optional scaling
        if scale != 1.0:
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        result = self.pipeline.process(image)
        if show_visualization:
            self._visualize(image, result)
        self.result = result
        return result

    def _visualize(self, original: np.ndarray, processed: np.ndarray) -> None:
        """Create and display visualization grid."""
        images = [original, processed]
        titles = ["Original", "Processed"]
        display_grid(images, titles=titles, max_cols=2)

    def to_pipeline(self, new_config: dict) -> None:
        """Sends a new config to the pipeline.

        Any number of parameters can be sent, since
        the new config will be merged with the current config.

        current config is accessed through any pipeline.get_config()
        method. The merged is sent to pipeline.from_config() class method
        and replaces the instance in the Processor.
        """
        merged = {**self.pipeline.get_config(), **new_config}
        self.pipeline = self.pipeline.from_config(merged)

    # --- str and repr ---

    def __str__(self) -> str:
        """Return detailed image processor configuration."""
        import platform

        header = "=" * 50 + "\n"
        header += "IMAGE PROCESSOR\n"
        header += "=" * 50 + "\n"

        # System info
        info = "\n--- System ---\n"
        info += f"OS: {platform.system()} | "
        info += f"| Python: {platform.python_version()} "
        info += f"| OpenCV: {cv2.__version__}\n"

        # Image settings
        info += "\n--- Image Settings ---\n"
        info += f"Source: {self.img_path.name}\n"
        info += f"Path: {self.img_path.parent}\n"

        # Pipeline info
        if self.pipeline is not None:
            info += "\n--- Pipeline ---\n"
            info += str(self.pipeline)
        else:
            info += "\n--- Pipeline ---\n"
            info += "No pipeline configured\n"

        footer = "\n" + "=" * 50

        return header + info + footer

    def __repr__(self) -> str:
        return (
            f"ImageProcessor(source={self.img_path}, "
            f"pipeline={self.pipeline.__class__.__name__ if self.pipeline else None})"
        )

    # --- validation ---

    def _validate_img_src(self, img_src):
        if not isinstance(img_src, (str, Path)):
            raise TypeError("Image source must be str or Path.")
