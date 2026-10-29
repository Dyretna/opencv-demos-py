# cred to Lott & Phillips (2021) Python Object Oriented Programing


from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .point import Point

Pair = Tuple[float, float]
Point_or_Tuple = Union[Point, Pair]


class Polygon:
    def __init__(self, vertices: Optional[Iterable[Point_or_Tuple]] = None) -> None:
        """
        initializes the polygon with optional points.

        If no arguments passed, sets vertices as empty list.
        """
        self.vertices: List[Point] = []
        if vertices:
            for point_or_tuple in vertices:
                self.vertices.append(self.make_point(point_or_tuple))

    @property
    def center(self) -> Point:
        x = int(sum(p.x for p in self.vertices) / len(self.vertices))
        y = int(sum(p.y for p in self.vertices) / len(self.vertices))
        return self.make_point((x, y))

    @center.setter
    def center(self, new_center: Point_or_Tuple) -> None:
        new_center = self.make_point(new_center)
        distance = (new_center.x - self.center.x, new_center.y - self.center.y)

        [p.move(int(p.x + distance[0]), int(p.y + distance[1])) for p in self.vertices]

    # --- measures: perimeter, area, etc ---

    @property
    def perimeter(self) -> float:
        pairs = zip(self.vertices, self.vertices[1:] + self.vertices[:1])
        return round(sum(p1.calculate_euclidean_distance(p2) for p1, p2 in pairs), 2)

    @property
    def area(self) -> float:
        """Calculates the area of any polygon using the Shoelace Formula

        vertices has to be in order!
        https://stackoverflow.com/questions/41077185/fastest-way-to-shoelace-formula
        https://en.wikipedia.org/wiki/Shoelace_formula
        """
        coords = np.array([(p.x, p.y) for p in self.vertices])

        x = coords[:, 0]
        y = coords[:, 1]

        # roll back creates next points in polygon
        # we are then calculating the diagonal sums
        s1 = np.sum(x * np.roll(y, -1))
        s2 = np.sum(y * np.roll(x, -1))
        area = 0.5 * np.absolute(s1 - s2)

        return area

    # --- manipulations ---
    def move(self, coordinate: Point_or_Tuple) -> None:
        """Moves the polygons center, to the assigned coordinate"""
        self.center = coordinate

    def scale(self, scale: float) -> None:
        """Scales the Polygon.

        Scales the difference between polygon center and point,
        then adds the center coordinates.
        """
        center = self.center
        [
            p.move(
                int((p.x - center.x) * scale) + center.x,
                int((p.y - center.y) * scale) + center.y,
            )
            for p in self.vertices
        ]

    def rotate(self, deg: float) -> None:
        """rotates the polygon around its center.

        tweaked version of:
        https://stackoverflow.com/questions/45508202/how-to-rotate-a-polygon
        """
        theta = np.radians(deg)
        # cos and sin describes point in radians from origo
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        center = self.center
        for p in self.vertices:
            # get distance between point and center - as a line
            tx = p.x - center.x
            ty = p.y - center.y

            p.move(  # rotate line from axis
                int((tx * cos_theta - ty * sin_theta) + center.x),
                int((tx * sin_theta + ty * cos_theta) + center.y),
            )

    # --- making and adding points ---

    @staticmethod
    def make_point(item: Point_or_Tuple) -> Point:
        return item if isinstance(item, Point) else Point(*item)

    def add_point(self, point: Point) -> None:
        self.vertices.append(point)

    def add_points(self, points: Iterable[Point,]) -> None:
        self.vertices.extend(points)

    # --- exports ---

    def bounding_box(self) -> tuple[tuple, tuple]:
        min_x = min(self.vertices, key=lambda p: p.x)
        min_y = min(self.vertices, key=lambda p: p.y)
        max_x = max(self.vertices, key=lambda p: p.x)
        max_y = max(self.vertices, key=lambda p: p.y)

        tl = (min_x.x, min_y.y)
        br = (max_x.x, max_y.y)
        return (tl, br)

    def to_array(self):
        return np.array([[p.make_list()] for p in self.vertices])

    def __str__(self):
        return f"\nPolygon: {self.vertices}\nCenter: {self.center}"
