import tensorflow as tf

# Utils
import numpy as np
import matplotlib.pyplot as plt
from std.io import cout, endl

from typing import List

CLASS_NAMES: List[str] = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class Dataset:
    def __init__(self, data: [np.ndarray, np.ndarray]):
        self.images: np.ndarray = data[0] / 255.0
        self.labels: np.ndarray = data[1]

def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    data = fashion_mnist.load_data()
    train_data, test_data = Dataset(data[0]), Dataset(data[1])
    # Prepare the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # Train the model
    model.fit(train_data.images, train_data.labels, epochs=7)
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data.images, test_data.labels)
    cout << 'Test loss: ' << test_loss << endl;
    cout << 'Test accuracy: ' << test_acc << endl;

if __name__ == '__main__':
    main()

