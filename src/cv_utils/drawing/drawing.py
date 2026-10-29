import cv2
import numpy as np

from ..image_utils import ImageGeometry
from ..shapes import Circle, Polygon, RegularShape


def draw_img_center_point(img: np.ndarray):
    """Draw a green circle at the center of the image.

    Args:
        img: Input image

    Returns:
        Copy of image with center point drawn
    """
    center = ImageGeometry.get_center(img)
    new_img = img.copy()
    cv2.circle(new_img, center, radius=5, color=(0, 255, 0), thickness=-1)
    return new_img


def draw_labeled_box(
    img: np.ndarray,
    tl: tuple[int, int],
    br: tuple[int, int],
    text: str,
    fontscale: tuple = 1.0,
    color: tuple = (250, 0, 0),
    thickness: int = 2,
):
    """convinience function to draw box with text under box."""
    cv2.rectangle(img, tl, br, color, thickness)
    # calculate bottom left and add 50 pix below
    bottom_left = (tl[0], br[1] + 15)
    cv2.putText(
        img,
        text,
        bottom_left,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontscale,
        color,
        thickness,
    )


def draw_filled_shapes(
    img: np.ndarray,
    shapes: list[Polygon,],
    colors: list[tuple,],
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
) -> np.ndarray:
    i = 0
    for shape in shapes:
        img_copy = img.copy()
        if isinstance(shape, (RegularShape, Polygon)):
            cv2.fillPoly(
                img=img_copy,
                pts=[shape.to_array()],
                color=colors[i],
                lineType=1,
            )
        elif isinstance(shape, Circle):
            cv2.circle(
                img=img_copy,
                center=shape.center,
                radius=shape.radius,
                color=colors[i],
                thickness=-1,
            )
        else:
            raise TypeError(
                f"shape must be RegularShape, Polygon or Circle: {type(shape)}"
            )
        # add transparency
        img = cv2.addWeighted(
            src1=img, alpha=alpha, src2=img_copy, beta=beta, gamma=gamma
        )
        i += 1
    return img


def draw_wireframe_shapes(
    img: np.ndarray,
    shapes: list[Polygon,],
    colors: list[tuple,],
    thickness: int,
) -> np.ndarray:
    i = 0
    for shape in shapes:
        img_copy = img.copy()
        if isinstance(shape, (RegularShape, Polygon)):
            cv2.polylines(
                img=img_copy,
                pts=[shape.to_array()],
                isClosed=True,
                color=colors[i],
                thickness=thickness,
                lineType=1,
            )
        elif isinstance(shape, Circle):
            cv2.circle(
                img=img_copy,
                center=shape.center,
                radius=shape.radius,
                color=colors[i],
                thickness=thickness,
            )
        else:
            raise TypeError(
                f"shape must be RegularShape, Polygon or Circle: {type(shape)}"
            )
        i += 1
    return img_copy
