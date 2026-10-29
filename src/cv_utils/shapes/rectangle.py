from typing import Optional, Tuple, Union

from .point import Point
from .polygon import Polygon

Pair = Tuple[float, float]
Point_or_Tuple = Union[Point, Pair]


class Rectangle(Polygon):
    """
    Represents a rectangle defined by center, width, and height.
    Unlike RegularSquare, a Rectangle can have different width and height.
    """

    def __init__(self, vertices: Optional[list[Point_or_Tuple]] = None):
        """
        Initialize a rectangle with four vertices.

        Args:
            vertices: List of four points defining the rectangle corners
        """
        super().__init__(vertices)
        if vertices and len(vertices) != 4:
            raise ValueError("Rectangle must have exactly 4 vertices")

    @classmethod
    def create_from_center(cls, center: Point_or_Tuple, width: int, height: int):
        """
        Create a rectangle with given center, width, and height.

        Args:
            center: Center point of the rectangle
            width: Width of the rectangle
            height: Height of the rectangle

        Returns:
            Rectangle instance
        """
        if not isinstance(center, Point):
            center_point = cls.make_point(center)
        else:
            center_point = center

        half_width = width / 2
        half_height = height / 2

        # Create vertices starting from top-left, going clockwise
        vertices = [
            Point(
                int(center_point.x - half_width), int(center_point.y - half_height)
            ),  # Top-left
            Point(
                int(center_point.x + half_width), int(center_point.y - half_height)
            ),  # Top-right
            Point(
                int(center_point.x + half_width), int(center_point.y + half_height)
            ),  # Bottom-right
            Point(
                int(center_point.x - half_width), int(center_point.y + half_height)
            ),  # Bottom-left
        ]
        return cls(vertices)

    @classmethod
    def create_from_topleft(cls, top_left: Point_or_Tuple, width: int, height: int):
        """
        Create a rectangle from top-left corner with given width and height.
        Common format in OpenCV.

        Args:
            top_left: Top-left corner point
            width: Width of the rectangle
            height: Height of the rectangle

        Returns:
            Rectangle instance
        """
        if not isinstance(top_left, Point):
            tl = cls.make_point(top_left)
        else:
            tl = top_left

        vertices = [
            Point(int(tl.x), int(tl.y)),  # Top-left
            Point(int(tl.x + width), int(tl.y)),  # Top-right
            Point(int(tl.x + width), int(tl.y + height)),  # Bottom-right
            Point(int(tl.x), int(tl.y + height)),  # Bottom-left
        ]
        return cls(vertices)

    @classmethod
    def create_from_corners(
        cls, top_left: Point_or_Tuple, bottom_right: Point_or_Tuple
    ):
        """
        Create a rectangle from two opposite corners.
        Common format in OpenCV: cv2.rectangle(img, pt1, pt2, color).

        Args:
            top_left: Top-left corner point
            bottom_right: Bottom-right corner point

        Returns:
            Rectangle instance
        """
        if not isinstance(top_left, Point):
            tl = cls.make_point(top_left)
        else:
            tl = top_left

        if not isinstance(bottom_right, Point):
            br = cls.make_point(bottom_right)
        else:
            br = bottom_right

        vertices = [
            Point(int(tl.x), int(tl.y)),  # Top-left
            Point(int(br.x), int(tl.y)),  # Top-right
            Point(int(br.x), int(br.y)),  # Bottom-right
            Point(int(tl.x), int(br.y)),  # Bottom-left
        ]
        return cls(vertices)

    @property
    def top_left(self) -> Point:
        """Get the top-left corner of the rectangle."""
        return self.vertices[0]

    @property
    def top_right(self) -> Point:
        """Get the top-left corner of the rectangle."""
        return self.vertices[1]

    @property
    def bottom_right(self) -> Point:
        """Get the bottom-right corner of the rectangle."""
        return self.vertices[2]

    @property
    def bottom_left(self) -> Point:
        """Get the bottom-right corner of the rectangle."""
        return self.vertices[3]

    @property
    def corners(self) -> tuple[Point, Point]:
        """Get top-left and bottom-right corners as tuple.

        Returns:
            Tuple of (top_left, bottom_right) points
        """
        return (self.top_left, self.bottom_right)

    @property
    def width(self) -> float:
        """Calculate the width of the rectangle."""
        return int(self.top_left.calculate_euclidean_distance(self.top_right))

    @property
    def height(self) -> float:
        """Calculate the height of the rectangle."""
        return int(self.top_left.calculate_euclidean_distance(self.bottom_left))

    @property
    def area(self) -> float:
        """Calculate the area of the rectangle."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio (width / height)."""
        if self.height != 0:
            return self.width / self.height
        return 0.0

    @property
    def diagonal(self) -> float:
        """Calculate the diagonal length of the rectangle."""
        if len(self.vertices) >= 3:
            return self.vertices[0].calculate_euclidean_distance(self.vertices[2])
        return 0.0

    def __str__(self):
        return (
            f"\nRectangle:\n"
            f"  Center: {self.center}\n"
            f"  Width: {self.width:.2f}\n"
            f"  Height: {self.height:.2f}\n"
            f"  Area: {self.area:.2f}\n"
            f"  Aspect Ratio: {self.aspect_ratio:.2f}"
        )

    @staticmethod
    def calculate_area_from_corners(
        top_left: Point_or_Tuple, bottom_right: Point_or_Tuple
    ) -> int:
        """Calculate rectangle area from two opposite corners"""
        # Handle both Point and tuple inputs
        if isinstance(top_left, Point):
            tl_x, tl_y = top_left.x, top_left.y
        else:
            tl_x, tl_y = top_left

        if isinstance(bottom_right, Point):
            br_x, br_y = bottom_right.x, bottom_right.y
        else:
            br_x, br_y = bottom_right

        # Calculate dimensions
        width = abs(br_x - tl_x)
        height = abs(br_y - tl_y)

        return int(width * height)
