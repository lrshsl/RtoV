
from torch import nn

from utils.vecs import Vec3

VERSION: str = '1.0.0'


DEFAULT_MODEL: str = 'default_model'
SAVED_MODELS_DIR: str = 'saved_models'

IMG_DIM: Vec3 = Vec3(32, 32, 3)
SHAPE_SPEC_MAX_SIZE: int = 6        # Triangle: 3 corners, x and y each
                                    # Needs to be defined, since all shapes are padded to the same size


SHAPE_LOSS_FN: nn.Module = nn.CrossEntropyLoss()
POINT_LOSS_FN: nn.Module = nn.MSELoss()


ColorTuple = tuple[int, int, int]

class Color:
    WHITE: ColorTuple = (255, 255, 255)
    BLACK: ColorTuple = (0, 0, 0)

    RED: ColorTuple = (0, 0, 255)
    GREEN: ColorTuple = (0, 255, 0)
    BLUE: ColorTuple = (255, 0, 0)
