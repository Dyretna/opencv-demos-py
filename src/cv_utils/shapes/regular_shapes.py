import math
from abc import ABC

from .point import Point
from .polygon import Polygon


class RegularShape(Polygon, ABC):
    class_vertices = 0

    @classmethod
    def create(cls, center: Point, size: int):
        """
        Create a regular polygon with given center and size.

        https://stackoverflow.com/questions/62734379/\
            how-to-create-a-triangle-with-the-center-\
                coordinates-and-the-coordinates-of-one
        """
        center = cls.make_point(center)
        angle_between_vertices = 2 * math.pi / cls.class_vertices
        radius = size / 2

        # Generate vertices around the center
        vertices = []
        for i in range(cls.class_vertices):
            angle = i * angle_between_vertices - math.pi / 2
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            vertices.append(Point(int(x), int(y)))

        return cls(vertices)


class RegularTriangle(RegularShape):
    class_vertices = 3

    def __init__(self, vertices=None):
        super().__init__(vertices)


class RegularSquare(RegularShape):
    class_vertices = 4

    def __init__(self, vertices=None):
        super().__init__(vertices)


class RegularPentagon(RegularShape):
    class_vertices = 5

    def __init__(self, vertices=None):
        super().__init__(vertices)


class RegularHexagon(RegularShape):
    class_vertices = 6

    def __init__(self, vertices=None):
        super().__init__(vertices)


class RegularHeptagon(RegularShape):
    class_vertices = 7

    def __init__(self, vertices=None):
        super().__init__(vertices)


class RegularOctagon(RegularShape):
    class_vertices = 8

    def __init__(self, vertices=None):
        super().__init__(vertices)
