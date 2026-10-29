# cred to Lott & Phillips (2021) Python Object Oriented Programing


import math


class Point:
    """
    Represents a point in two-dimensional geometric coordinates

    >>> p_0 = Point()
    >>> p_1 = Point(3, 4)
    >>> p_0.calculate_distance(p_1)
    5.0
    """

    def __init__(self, x: float = 0, y: float = 0) -> None:
        """Initialize the position of a new point.

        The x and y coordinates can be specified. If they are not,
        the point defaults to the origin.
        """
        self.move(x, y)

    def move(self, x: float, y: float) -> None:
        """
        Move the point to anew location in 2D space.
        """
        self.x = x
        self.y = y

    def calculate_manhattan_distances(self, other: "Point") -> tuple[int, int]:
        """
        Calculate the x and y distances from this point
        to a second point passed as a parameter.
        """
        return (abs(self.x - other.x), abs(self.y - other.y))

    def calculate_euclidean_distance(self, other: "Point") -> float:
        """
        Calculate the Euclidean distance from this point
        to a second point passed as a parameter.
        """
        return math.hypot(self.x - other.x, self.y - other.y)

    def make_tuple(self) -> tuple[int, int]:
        """return tuple from Point instance"""
        return (self.x, self.y)

    def make_list(self) -> list[int, int]:
        """return list from Point instance"""
        return [self.x, self.y]

    @staticmethod
    def make_point(tuple_x_y: tuple[int, int]) -> tuple:
        """make Point instance from tuple"""
        return Point(tuple_x_y[0], tuple_x_y[1])

    # should probably later be moved to a "shapes" class
    @staticmethod
    def square_pts_from_center(center: "Point", size: int) -> tuple["Point", "Point"]:
        top_left = (int(center.x - (size / 2)), int(center.y - (size / 2)))
        bottom_right = (int(center.x + (size / 2)), int(center.y + (size / 2)))

        return (Point.make_point(top_left), Point.make_point(bottom_right))

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
