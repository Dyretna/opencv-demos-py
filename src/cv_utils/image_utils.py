import numpy as np


class ImageGeometry:
    """Utility class for image geometry operations."""

    @staticmethod
    def get_center(img: np.ndarray):
        """Get the center point of an image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Tuple of (center_width, center_height)
        """
        height, width = img.shape[:2]
        center_height = int(height / 2)
        center_width = int(width / 2)
        return (center_width, center_height)

    @staticmethod
    def get_top_center(img: np.ndarray):
        """Get the top center point of an image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Tuple of (center_width, 0)
        """
        height, width = img.shape[:2]
        center_width = int(width / 2)
        return (center_width, 0)

    @staticmethod
    def get_top_left(img: np.ndarray):
        """Get the top left corner point of an image.

        Returns:
            Tuple of (0, 0)
        """
        return (0, 0)

    @staticmethod
    def get_top_right(img: np.ndarray):
        """Get the top right corner point of an image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Tuple of (width - 1, 0)
        """
        height, width = img.shape[:2]
        return (width - 1, 0)

    @staticmethod
    def get_bottom_left(img: np.ndarray):
        """Get the bottom left corner point of an image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Tuple of (0, height - 1)
        """
        height, width = img.shape[:2]
        return (0, height - 1)

    @staticmethod
    def get_bottom_right(img: np.ndarray):
        """Get the bottom right corner point of an image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Tuple of (width - 1, height - 1)
        """
        height, width = img.shape[:2]
        return (width - 1, height - 1)

    @staticmethod
    def get_corners(img: np.ndarray) -> tuple[tuple,]:
        """Returns tuple of image corners in clockwise order:

        - top left
        - top right
        - bottom right
        - bottom left
        """
        return (
            ImageGeometry.get_top_left(img),
            ImageGeometry.get_top_right(img),
            ImageGeometry.get_bottom_right(img),
            ImageGeometry.get_bottom_left(img),
        )

    @staticmethod
    def get_top_half(img: np.ndarray):
        """Return the top half of the image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Top half of the image
        """
        height = img.shape[0]
        if len(img.shape) == 3:  # Color image
            return img[: int(height / 2), :, :]
        else:  # Grayscale image
            return img[: int(height / 2), :]

    @staticmethod
    def get_bottom_half(img: np.ndarray):
        """Return the bottom half of the image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Bottom half of the image
        """
        height = img.shape[0]
        if len(img.shape) == 3:  # Color image
            return img[int(height / 2) :, :, :]
        else:  # Grayscale image
            return img[int(height / 2) :, :]

    @staticmethod
    def get_left_half(img: np.ndarray):
        """Return the left half of the image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Left half of the image
        """
        width = img.shape[1]
        if len(img.shape) == 3:  # Color image
            return img[:, : int(width / 2), :]
        else:  # Grayscale image
            return img[:, : int(width / 2)]

    @staticmethod
    def get_right_half(img: np.ndarray):
        """Return the right half of the image.

        Args:
            img: Input image (color or grayscale)

        Returns:
            Right half of the image
        """
        width = img.shape[1]
        if len(img.shape) == 3:  # Color image
            return img[:, int(width / 2) :, :]
        else:  # Grayscale image
            return img[:, int(width / 2) :]
