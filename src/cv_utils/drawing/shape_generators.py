import random as rd

import numpy as np

from ..shapes import (
    Circle,
    Point,
    RegularHeptagon,
    RegularHexagon,
    RegularOctagon,
    RegularPentagon,
    RegularSquare,
    RegularTriangle,
)


def random_regulars(
    img_size: tuple = (900, 900),
    n: int = 10,
    shape_max_size: int | None = None,
    shape_min_size: int | None = None,
):
    if not shape_max_size:
        shape_max_size = int(np.sqrt(img_size[0] * img_size[1]) / 5)
    if not shape_min_size:
        shape_min_size = int(np.sqrt(img_size[0] * img_size[1]) / 10)

    # randomization loop
    rand_shapes = []

    for _ in range(n):
        # include padding to img border
        rand_center_x = rd.randint(50, img_size[0] - 50)
        rand_center_y = rd.randint(50, img_size[1] - 50)
        rand_center = Point(rand_center_x, rand_center_y)

        rand_shape_size = rd.randint(shape_min_size, shape_max_size)

        ### debug:
        # print("rand_center: ", rand_center, "rand_size: ", rand_shape_size)

        shapes = [
            Circle.create(rand_center, rand_shape_size),
            RegularTriangle.create(rand_center, rand_shape_size),
            RegularSquare.create(rand_center, rand_shape_size),
            RegularPentagon.create(rand_center, rand_shape_size),
            RegularHexagon.create(rand_center, rand_shape_size),
            RegularHeptagon.create(rand_center, rand_shape_size),
            RegularHexagon.create(rand_center, rand_shape_size),
            RegularOctagon.create(rand_center, rand_shape_size),
        ]
        rand_shapes.append(rd.choice(shapes))

    for shape in rand_shapes:
        if not isinstance(shape, Circle):
            shape.rotate(rd.randint(0, 360))

    return rand_shapes
