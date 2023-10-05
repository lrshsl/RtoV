from utils.vec import Vec2
from utils.shapes import Shapes

import numpy as np
import cv2
from cv2.typing import Scalar

class DefaultPoints: # {{{
    @staticmethod
    def circle(dim: Vec2, random: bool = True, pad: int = 3) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)
        if random:
            rng = np.random.default_rng()
            pts[0] = rng.integers(
                low = dim.x // 4, high = int(dim.x * 3 // 4))    # center x
            pts[1] = rng.integers(
                low = dim.y // 4, high = int(dim.y * 3 // 4))    # center y
            pts[2] = rng.integers(
                low = 0, high = int(min(
                    pts[0], dim.x - pts[0],
                    pts[1], dim.y - pts[1]
                    )))    # radius
            return pts

        pts[0] = dim.x // 2
        pts[1] = dim.y // 2
        pts[2] = max(dim) // 4
        return pts

    @staticmethod
    def line(dim: Vec2, random: bool = True, pad: int = 4) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)
        if random:
            rng = np.random.default_rng()
            pts[0] = rng.integers(
                low = 0, high = int(dim.x), size = 2)    # x values
            pts[1] = rng.integers(
                low = 0, high = int(dim.x), size = 2)    # x values
            pts[2] = rng.integers(
                low = 0, high = int(dim.y), size = 2)    # y values
            pts[3] = rng.integers(
                low = 0, high = int(dim.y), size = 2)    # y values
            return pts

        # Diagonal line through the image from (0, 0)
        pts[0] = dim.x
        pts[1] = dim.y
        return pts

    @staticmethod
    def triangle(dim: Vec2, random: bool = True, pad: int = 6) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)

        # Random three points
        if random:
            rng = np.random.default_rng()
            pts[:, 0] = rng.integers(
                low = 0, high = int(dim.x), size = 3)    # x values
            pts[:, 1] = rng.integers(
                low = 0, high = int(dim.y), size = 3)    # y values
            return pts

        # (More or less) regular triangle
        pts[0, 0] = dim.x // 4      # Bottom left
        pts[0, 1] = dim.y // 4
        pts[1, 0] = dim.x * 3 // 4  # Bottom right
        pts[1, 1] = dim.y // 4
        pts[2, 0] = dim.x // 2      # Top mid
        pts[2, 1] = dim.y * 3 // 4
        return pts

    @staticmethod
    def rectangle(dim: Vec2, random: bool = True, pad: int = 5) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)

        if random:
            rng = np.random.default_rng()
            # Top left corner
            pts[0] = rng.integers(0, int(dim.x * 3 // 4))  # Leave enough space free
            pts[1] = rng.integers(0, int(dim.y * 3 // 4))
            # Width
            pts[2] = rng.integers(pts[0], int(dim.x))
            # Height
            pts[3] = rng.integers(pts[1], int(dim.y))
            return pts

        # Regular rectangle
        pts[[0, 1], 0] = dim.x // 4     # Left edge
        pts[[0, 3], 1] = dim.y // 4     # Top edge
        pts[[2, 3], 0] = dim.x * 3 // 4 # Right edge
        pts[[1, 2], 1] = dim.y * 3 // 4 # Bottom edge
        return pts
# }}}

# fn draw_on_image {{{
def draw_on_image(img: np.ndarray,
                  shape: Shapes,
                  color: Scalar = (0, 0, 0),
                  random: bool = True,
                  pad: int = 6) -> np.ndarray:
    """Mutates `img` in-place"""

    pts = np.zeros(pad, dtype=np.float32)
    match shape:
        case Shapes.Circle:
            pts = DefaultPoints.circle(
                Vec2(img.shape[0], img.shape[1]), random=random, pad=pad)
            cv2.circle(img, (pts[0], pts[1]), pts[2], color, -1)
            return pts

        case Shapes.Line:
            pts = DefaultPoints.line(
                Vec2(img.shape[0], img.shape[1]), random=random, pad=pad)
            cv2.line(img, pts[0], pts[1], color)
            return pts

        case Shapes.Triangle:
            pts = DefaultPoints.triangle(
                Vec2(img.shape[0], img.shape[1]), random=random, pad=pad)

        case Shapes.Rectangle:
            pts = DefaultPoints.rectangle(
                Vec2(img.shape[0], img.shape[1]), random=random, pad=pad)

        case _:
            raise ValueError("Unknown or unimplemented shape: {}".format(shape))

    cv2.fillPoly(img, [pts], color=color)
    return pts
# }}}

