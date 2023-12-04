
import numpy as np
import cv2
from cv2.typing import Scalar
from typing import Optional, Callable

from utils.shapes import Shapes
from utils.vecs import Vec2
from data.shape_generator import ShapeGenerator


def draw_on_image(img: np.ndarray,
                  shape: Shapes,
                  color: Scalar = (0, 0, 0),
                  random: bool = True,
                  pad: int = 6,
                  seed: Optional[Callable[[], int]] = None) -> np.ndarray:
    """Mutates `img` in-place"""

    # Prepare array, to fill the points in
    pts = np.zeros(pad, dtype=np.float32)

    seed_int = None if seed is None else seed()

    # Generate the required information and draw the resulting shape
    match shape:

        case Shapes.Circle: # {{{

            # Get points
            pts = ShapeGenerator.circle(
                Vec2(img.shape[0], img.shape[1]),
                random=random, pad=pad, seed=seed_int)

            # Draw
            cv2.circle(img, (pts[0], pts[1]), pts[2], color, -1)

            return pts # }}}

        case Shapes.Line: # {{{

            # Get points
            pts = ShapeGenerator.line(
                Vec2(img.shape[0], img.shape[1]),
                random=random, pad=pad, seed=seed_int)

            # Draw
            cv2.line(img, pts[[0, 2]].tolist(), pts[[1, 3]].tolist(), color)

            return pts # }}}

        case Shapes.Triangle: # {{{

            # Get points
            pts = ShapeGenerator.triangle(
                Vec2(img.shape[0], img.shape[1]),
                random=random, pad=pad, seed=seed_int)

            # Draw
            cv2.fillPoly(img, [pts.reshape(-1, 2)], color=color)

            return pts # }}}

        case Shapes.Rectangle: # {{{

            # Get points
            pts = ShapeGenerator.rectangle(
                Vec2(img.shape[0], img.shape[1]),
                random=random, pad=pad, seed=seed_int)

            # Draw
            cv2.rectangle(img, (pts[0], pts[1]),
                          (pts[2], pts[3]), color, -1)

            return pts # }}}

        case _: # {{{
            raise ValueError("Unknown or unimplemented shape: {}".format(shape))
        # }}}

