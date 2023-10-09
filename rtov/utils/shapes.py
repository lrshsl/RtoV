
import numpy as np
import cv2
import enum


SHAPE_NAMES: tuple[str, ...] = 'Circle', 'Line', 'Triangle', 'Rectangle'

class Shapes(enum.IntEnum):
    Circle = 0
    Line = 1
    Triangle = 2
    Rectangle = 3


def draw_shape(image: np.ndarray,
               shape: Shapes,
               data: np.ndarray,
               color: tuple[int, int, int] = (200, 200, 200)):
    """Draw a shape with given data on an image. Is used to show the predictions of the model."""

    match shape:

        case Shapes.Circle:
            assert len(data) >= 3
            center = data[:2]
            r = data[2]
            if r < 0:
                print('<Error> Could not draw circle: Radius is negative')
                return
            cv2.circle(image, center, r, color, -1)

        case Shapes.Line:
            assert len(data) >= 4
            p1 = data[:2]
            p2 = data[2:4]
            _width = data[4]
            cv2.line(image, p1, p2, color, 1)

        case Shapes.Triangle:
            assert len(data) >= 6
            if not any(data):
                print('<Error> Could not draw triangle: Data is empty (all zeros)')
                return

            p1 = data[:2].tolist()
            p2 = data[2:4].tolist()
            p3 = data[4:6].tolist()
            pts = np.array([p1, p2, p3], dtype=np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(image, [pts], color)

        case Shapes.Rectangle:
            assert len(data) >= 5
            pt = data[:2]
            w, h = data[2:4]
            _orientation = data[4]
            if w == 0 and h == 0:
                print('<Error> Could not draw rectangle: Width and height are 0')
                return
            p1, p2, p3, p4 = pt.tolist(), pt.tolist(), pt.tolist(), pt.tolist()
            # p1        # top-left
            p2[0] += w  # top-right
            p3[0] += w  # bottom-right
            p3[1] += h
            p4[1] += h  # bottom-left
            pts = np.array([p1, p2, p3, p4], dtype=np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(image, [pts], color)







