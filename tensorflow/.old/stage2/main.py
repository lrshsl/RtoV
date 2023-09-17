import tensorflow as tf
import numpy as np

from .image_generator import ImageGenerator
from stage2.dataset import Dataset
from stage2.model import Model

# Utils
import matplotlib.pyplot as plt
from typing import List, Sequence

from std.io import cout, endl


def main():
    # Load the data
    img_gen = ImageGenerator.default(redundancy=4);
    train_data = Dataset.from_image_generator(img_gen);
    # Create the model
    model: Model = Model().prepare_default().compile();
    cout << train_data.data << endl;
    model.train(train_data, epochs=7);
    # Evaluate the model
    test_loss, test_acc = model.evaluate(train_data);
    cout << 'Test loss: ' << test_loss << endl;
    cout << 'Test accuracy: ' << test_acc << endl;


if __name__ == '__main__':
    main();

