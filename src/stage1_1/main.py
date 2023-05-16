import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from std.io import cout, endl

def main():
    # Generate random images
    images = []
    labels = []
    for i in range(300):
        r = np.random.randint(0, 4)
        r = [0, 100, 200, 254][r]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image += np.array((r, 0, 0), dtype=np.uint8)
        images.append(image)
        labels.append(r)

    # Convert the data to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    # for img, lab in zip(images, labels):
    #     cout << "Image[0]: " << img[0];
    #     cout << "Label" << lab << endl;

    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(255, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(images, labels, epochs=12)

    # Plot the training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model performance')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    plt.savefig('model_performance.png')


if __name__ == '__main__':
    main()
