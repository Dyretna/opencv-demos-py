from typing import Iterable, Optional, Tuple, Union

from .point import Point
from .polygon import Polygon

Pair = Tuple[float, float]
Point_or_Tuple = Union[Point, Pair]


class Line(Polygon):
    """
    Represents a straight line segment defined by exactly two points.
    Corresponds to cv2.line() functionality.
    """

    def __init__(
        self,
        start: Optional[Point_or_Tuple] = None,
        end: Optional[Point_or_Tuple] = None,
    ):
        """
        Initialize a line with two points.

        Args:
            start: Starting point of the line
            end: Ending point of the line
        """
        if start is not None and end is not None:
            super().__init__([start, end])
        else:
            super().__init__()

        if len(self.vertices) > 2:
            raise ValueError("Line must have exactly 2 vertices")

    @property
    def start(self) -> Point:
        """Get the starting point of the line."""
        return self.vertices[0] if len(self.vertices) > 0 else None

    @start.setter
    def start(self, point: Point_or_Tuple) -> None:
        """Set the starting point of the line."""
        if len(self.vertices) > 0:
            self.vertices[0] = self.make_point(point)
        else:
            self.vertices.append(self.make_point(point))

    @property
    def end(self) -> Point:
        """Get the ending point of the line."""
        return self.vertices[1] if len(self.vertices) > 1 else None

    @end.setter
    def end(self, point: Point_or_Tuple) -> None:
        """Set the ending point of the line."""
        if len(self.vertices) > 1:
            self.vertices[1] = self.make_point(point)
        elif len(self.vertices) == 1:
            self.vertices.append(self.make_point(point))
        else:
            # Need start point first
            raise ValueError("Cannot set end point before start point")

    @property
    def length(self) -> float:
        """Calculate the length of the line."""
        if len(self.vertices) >= 2:
            return self.start.calculate_euclidean_distance(self.end)
        return 0.0

    @property
    def slope(self) -> float:
        """Calculate the slope of the line."""
        if len(self.vertices) >= 2:
            dx = self.end.x - self.start.x
            if dx == 0:
                return float("inf")
            return (self.end.y - self.start.y) / dx
        return 0.0

    @classmethod
    def create(cls, start: Point_or_Tuple, end: Point_or_Tuple):
        """Create a line with two points."""
        return cls(start, end)

    def __str__(self):
        return f"\nLine: {self.start} -> {self.end}\nLength: {self.length:.2f}"


class Polyline(Polygon):
    """
    Represents a series of connected line segments defined by 2 or more points.

    Corresponds to cv2.polylines() functionality.
    Useful for paths, curves, and lane detection
    where multiple connected segments are needed.
    """

    def __init__(self, vertices: Optional[Iterable[Point_or_Tuple]] = None):
        """
        Initialize a polyline with multiple vertices.

        Args:
            vertices: List of points defining the polyline path
        """
        super().__init__(vertices)

        if vertices and len(self.vertices) < 2:
            raise ValueError("Polyline must have at least 2 vertices")

    @property
    def start(self) -> Point:
        """Get the first point of the polyline."""
        return self.vertices[0] if len(self.vertices) > 0 else None

    @property
    def end(self) -> Point:
        """Get the last point of the polyline."""
        return self.vertices[-1] if len(self.vertices) > 0 else None

    @property
    def segments(self) -> int:
        """Get the number of line segments in the polyline."""
        return max(0, len(self.vertices) - 1)

    @property
    def total_length(self) -> float:
        """Calculate the total length of all segments."""
        if len(self.vertices) < 2:
            return 0.0

        length = 0.0
        for i in range(len(self.vertices) - 1):
            length += self.vertices[i].calculate_euclidean_distance(
                self.vertices[i + 1]
            )
        return length

    def segment_length(self, segment_index: int) -> float:
        """Calculate the length of a specific segment."""
        if segment_index < 0 or segment_index >= self.segments:
            raise IndexError(f"Segment index {segment_index} out of range")

        return self.vertices[segment_index].calculate_euclidean_distance(
            self.vertices[segment_index + 1]
        )

    @classmethod
    def create(cls, vertices: Iterable[Point_or_Tuple]):
        """Create a polyline with multiple points."""
        return cls(vertices)

    @classmethod
    def from_hough_lines(cls, hough_lines):
        """
        Create a Polyline from HoughLinesP output format.

        Args:
            hough_lines: Output from cv2.HoughLinesP, shape (N, 1, 4)
                        where each line is [x1, y1, x2, y2]

        Returns:
            List of Line instances
        """
        lines = []
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            lines.append(Line(Point(x1, y1), Point(x2, y2)))
        return lines

    def __str__(self):
        return (
            f"\nPolyline:\n"
            f"  Vertices: {len(self.vertices)}\n"
            f"  Segments: {self.segments}\n"
            f"  Total Length: {self.total_length:.2f}\n"
            f"  Start: {self.start}\n"
            f"  End: {self.end}"
        )
