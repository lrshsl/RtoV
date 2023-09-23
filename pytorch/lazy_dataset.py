from draw_shape_on_image import draw_on_image
from utils.shapes import shape_names

from torch.utils.data import Dataset

import numpy as np

from utils.vec import Vec3

class LazyDataset(Dataset):
    features: np.ndarray
    tensor: np.ndarray

    def __init__(self, img_dim: Vec3, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.features = np.zeros(img_dim.as_int_tuple(), dtype=np.uint8)
        self.tensor = np.zeros(img_dim.as_int_tuple(), dtype=np.uint8)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _) -> tuple[np.ndarray, int]:
        self.features[:] = np.full(self.features.shape, 255, dtype=np.uint8)    # Overwrite the memory directly to (hopefully) reduce memory => fewer GC cycles
        # shape = np.random.default_rng().integers(len(shapes))
        shape = np.random.choice(shape_names)
        pts = draw_on_image(self.features, shape, color=0)
        if self.transform:
            self.tensor = self.transform(self.features)
            # Don't overwrite with [:]:
            #   DataLoader uses pointers, which are referenced per-batch, not per-item.
            #   If overwritten, the pointers all point to the last element by the time they are accessed
        # TODO: Calc size + first point
        return self.tensor, shape_names.index(shape)
