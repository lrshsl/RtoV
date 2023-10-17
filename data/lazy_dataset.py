
from data.draw_shape_on_image import draw_on_image
from utils.shapes import Shapes
from utils.vecs import Vec3
import constants

from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Callable


class LazyDataset(Dataset):
    features: np.ndarray
    tensor: np.ndarray

    # __init__ {{{
    def __init__(self, img_dim: Vec3,
                 num_samples, transform=None,
                 seed: Optional[Callable[[], int]] = None,
                 pad: int = constants.SHAPE_SPEC_MAX_SIZE) -> None:
        self.num_samples = num_samples
        self.transform = transform
        self.features = np.zeros(img_dim.as_int_tuple(), dtype=np.uint8)
        self.tensor = np.zeros(img_dim.as_int_tuple(), dtype=np.uint8)
        self.seed = seed
        self.pad = pad # }}}

    # __len__ {{{
    def __len__(self) -> int:
        return self.num_samples # }}}

    # __getitem__ {{{
    def __getitem__(self, _) -> tuple[np.ndarray, Shapes, np.ndarray]:
        self.features[:] = np.full(self.features.shape, 255, dtype=np.uint8)    # Overwrite the memory directly to (hopefully) reduce memory => fewer GC cycles
        # shape = np.random.choice(shape_names)
        shape = np.random.choice(list(Shapes))
        pts = np.zeros(constants.SHAPE_SPEC_MAX_SIZE, dtype=np.float32)
        pts[:] = draw_on_image(self.features,
                               shape,
                               color = (0, 0, 0),
                               pad = self.pad,
                               seed = self.seed)
        if self.transform:
            self.tensor = self.transform(self.features)
            # Don't overwrite with [:]:
            #   DataLoader uses pointers, which are referenced per-batch, not directly after the call
            #   If overwritten, the pointers all point to the last element by the time they are accessed
        # TODO: Calc size + first point
        return self.tensor, shape, pts # }}}

