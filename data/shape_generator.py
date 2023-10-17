from typing import Optional
import numpy as np

from utils.vecs import Vec2


class ShapeGenerator:

    @staticmethod # circle {{{
    def circle(dim: Vec2, random: bool = True, pad: int = 3, seed: Optional[int] = 0) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)
        if random:
            rng = np.random.default_rng(seed)
            pts[0] = rng.integers(
                low = int(dim.x // 4), high = int(dim.x * 3 // 4))    # center x
            pts[1] = rng.integers(
                low = int(dim.y // 4), high = int(dim.y * 3 // 4))    # center y
            pts[2] = rng.integers(
                low = 1, high = int(min(
                    pts[0], dim.x - pts[0],
                    pts[1], dim.y - pts[1]
                    )))    # radius
            return pts

        pts[0] = dim.x // 2
        pts[1] = dim.y // 2
        pts[2] = max(iter(dim)) // 4
        return pts # }}}

    @staticmethod # line {{{
    def line(dim: Vec2, random: bool = True, pad: int = 4, seed: Optional[int] = 0) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)
        if random:
            rng = np.random.default_rng(seed)
            pts[[0, 2]] = rng.integers(
                low = 0, high = int(dim.x), size = 2)    # x values
            pts[[1, 3]] = rng.integers(
                low = 0, high = int(dim.y), size = 2)    # y values
            pts[4] = 1 # Width for now always 1 # TODO
            return pts

        # Diagonal line through the image from (0, 0)
        pts[0] = dim.x
        pts[1] = dim.y
        return pts # }}}

    @staticmethod # triangle {{{
    def triangle(dim: Vec2, random: bool = True, pad: int = 6, seed: Optional[int] = 0) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)

        # Random three points
        if random:
            rng = np.random.default_rng(seed)
            pts[[0, 2, 4]] = rng.integers(
                low = 0, high = int(dim.x), size = 3)    # x values
            pts[[1, 3, 5]] = rng.integers(
                low = 0, high = int(dim.y), size = 3)    # y values
            return pts

        # (More or less) regular triangle
        pts[0] = dim.x // 4      # Bottom left
        pts[1] = dim.y // 4
        pts[2] = dim.x * 3 // 4  # Bottom right
        pts[3] = dim.y // 4
        pts[4] = dim.x // 2      # Top mid
        pts[5] = dim.y * 3 // 4
        return pts # }}}

    @staticmethod # rectangle {{{
    def rectangle(dim: Vec2, random: bool = True, pad: int = 5, seed: Optional[int] = 0) -> np.ndarray:
        pts = np.zeros(pad, dtype=np.int32)

        if random:
            rng = np.random.default_rng(seed)
            # Top left corner
            pts[0] = rng.integers(0, int(dim.x // 2))  # Leave enough space free
            pts[1] = rng.integers(0, int(dim.y // 2))
            # Width
            pts[2] = rng.integers(1, max((int(dim.x) - pts[0]) * 9 // 10, 1))   # The upper limit is at least 1, else 9 10th of the remaining space
            # Height
            pts[3] = rng.integers(1, max((int(dim.y) - pts[1]) * 9 // 10, 1))   # Same with width
            return pts

        # Regular rectangle
        pts[0] = dim.x // 10     # x coordinate of top left corner
        pts[1] = dim.y // 10     # y coordinate
        pts[2] = dim.x * 8 // 10 # Width
        pts[3] = dim.y * 8 // 10 # Height
        return pts # }}}

