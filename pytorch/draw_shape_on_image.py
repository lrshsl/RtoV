from utils.vec import Vec2
from utils.shapes import Shapes

import numpy as np
import cv2

class DefaultPoints: # {{{
    @staticmethod
    def line(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((2, 2), dtype=np.int32)
        if random:
            rng = np.random.default_rng()
            pts[:, 0] = rng.integers(
                low = 0, high = int(dim.x), size = 2)    # x values
            pts[:, 1] = rng.integers(
                low = 0, high = int(dim.y), size = 2)    # y values
            return pts

        # Diagonal line through the image from (0, 0)
        pts[0, 0] = dim.x
        pts[0, 1] = dim.y
        return pts

    @staticmethod
    def triangle(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((3, 2), dtype=np.int32)

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
    def rectangle(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((4, 2), dtype=np.int32)

        if random:
            rng = np.random.default_rng()
            # Top left corner
            pts[0, 0] = rng.integers(0, int(dim.x * 3 // 4))  # Leave enough free space
            pts[0, 1] = rng.integers(0, int(dim.y * 3 // 4))
            # Bottom left corner, spans a rect with random height
            pts[1, 0] = pts[0, 0]
            pts[1, 1] = rng.integers(pts[0, 1], int(dim.y))
            # Top right corner -> random width
            pts[3, 0] = rng.integers(pts[0, 0], int(dim.x))
            pts[3, 1] = pts[0, 1]
            # Bottom right corner
            pts[2, 0] = pts[3, 0]
            pts[2, 1] = pts[1, 1]
            return pts

        # Regular rectangle
        pts[[0, 1], 0] = dim.x // 4     # Left edge
        pts[[0, 3], 1] = dim.y // 4     # Top edge
        pts[[2, 3], 0] = dim.x * 3 // 4 # Right edge
        pts[[1, 2], 1] = dim.y * 3 // 4 # Bottom edge
        return pts
# }}}

# fn draw_on_image {{{
def draw_on_image(img: np.ndarray, shape: Shapes, color: int = 0) -> np.ndarray:
    """Mutates `img` in-place"""
    if shape == 'Line':
        pts = DefaultPoints.line(
            Vec2(img.shape[0], img.shape[1]), random=True)
        cv2.line(img, pts[0], pts[1], color)
        return pts
    elif shape == 'Triangle':
        pts = DefaultPoints.triangle(
            Vec2(img.shape[0], img.shape[1]), random=True)
    elif shape == 'Rect':
        pts = DefaultPoints.rectangle(
            Vec2(img.shape[0], img.shape[1]), random=True)
    else:
        raise ValueError("Unknown or unimplemented shape: {}".format(shape))
    cv2.fillPoly(img, [pts], color=color)
    return pts
# }}}
