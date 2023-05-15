import tensorflow as tf
import numpy as np

from stage2.dataset import Dataset

from typing import Sequence, Self


class Model:
    # Prepare the model
    def prepare_default(self) -> Self:
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=((100, 100, 3))),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(255)
        ]);
        return self;

    # Compile the model
    def compile(self):
        self.model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return self;

    # Fit the model
    def train(self, dataset: Dataset, epochs: int = 5):
        self.model.fit(dataset.data, epochs=epochs);

    # Evaluate the model
    def evaluate(self, dataset: Dataset):
        return self.model.evaluate(dataset);
