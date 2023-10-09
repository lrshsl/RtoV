
from rtov.utils.vec import Vec3

DEFAULT_MODEL: str = 'default_model'
SAVED_MODELS_DIR: str = 'saved_models'

IMG_DIM: Vec3 = Vec3(32, 32, 3)
SHAPE_SPEC_MAX_SIZE: int = 6        # Triangle: 3 corners, x and y each
                                    # Needs to be defined, since all shapes are padded to the same size

