from typing import Tuple, Union

import numpy as np

from .point import Point

Pair = Tuple[float, float]
Point_or_Tuple = Union[Point, Pair]


class Circle:
    def __init__(self, center: Point_or_Tuple, size: int):
        self._center = (
            self.make_point(center) if not isinstance(center, Point) else center
        )
        self._radius = int(size / 2)

    @property
    def center(self):
        return (self._center.x, self._center.y)

    @center.setter
    def center(self, coordinates: Point):
        self._center = coordinates

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius: Tuple):
        self._radius = radius

    @property
    def area(self) -> float:
        return np.pi * (self._radius**2)

    # --- manipulations ---

    def scale(self, scale: float):
        self._radius = self._radius * scale

    # --- exports ---

    def bounding_box(self) -> Tuple[Pair, Pair]:
        cx, cy = self.center
        r = self._radius
        topleft = (cx - r, cy - r)
        bottomright = (cx + r, cy + r)

        return topleft, bottomright

    def to_array(self):
        return np.array(np.sin(self.radius), np.cos(self.radius))

    @staticmethod
    def make_point(item: Point_or_Tuple) -> Point:
        return item if isinstance(item, Point) else Point(*item)

    @classmethod
    def create(cls, center: Point_or_Tuple, size: int):
        return cls(center, size)
