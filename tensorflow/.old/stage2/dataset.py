import tensorflow as tf
import numpy as np

from stage2.image_generator import ImageGenerator

from typing import Sequence


class Dataset:
    def __init__(self, dataset: tf.data.Dataset):
        self.data = dataset;

    def from_arrays(self, data: np.ndarray, labels: np.ndarray):
        tf_dataset = tf.data.Dataset.from_tensor_slices((data, labels));
        return Dataset(tf_dataset);

    def from_image_generator(generator: ImageGenerator):
        return Dataset(
            tf.data.Dataset.from_tensor_slices(
                generator.images(),
                generator.labels(),
            ));

    # def labels(self) -> Sequence[int]:
        # return self.tf_dataset.map(lambda x: x[0]).unbatch()
