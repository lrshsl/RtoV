
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    # Generate random images
    images = []
    labels = []
    for i in range(300):
        bg_c = np.random.default_rng().integers(0, 256, dtype=int)
        fill_c = np.random.default_rng().integers(0, 256, dtype=int)
        assert 0 <= fill_c <= 255
        assert isinstance(fill_c, int)   # Gives trouble when np dtype
        assert 0 <= bg_c <= 255
        assert isinstance(bg_c, int)
        bg_c = bg_c, bg_c, bg_c
        fill_c = fill_c, fill_c, fill_c

        image = np.full((100, 100, 3), bg_c, dtype=np.uint8)
        shape = np.random.choice(['circle', 'square', 'triangle'])
        if shape == 'circle':
            x = np.random.randint(20, 80)
            y = np.random.randint(20, 80)
            r = np.random.randint(10, 30)
            image = cv2.circle(image, (x, y), r, fill_c, -1)
            label = 0
        elif shape == 'square':
            x1 = np.random.randint(20, 60)
            y1 = np.random.randint(20, 60)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(20, 40)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), fill_c, -1)
            label = 1
        elif shape == 'triangle':
            x1 = np.random.randint(20, 60)
            y1 = np.random.randint(20, 60)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(20, 40)
            x3 = x1 + np.random.randint(20, 40)
            y3 = y1 + np.random.randint(20, 40)
            cv2.fillPoly(image, [np.array([[x1, y1], [x2, y2], [x3, y3]])], fill_c)
            label = 2
        images.append(image)
        labels.append(label)

    # Convert the data to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(images, labels, epochs=10)

    # Plot the training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model performance')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    # plt.savefig('./' + __name__.split('.')[0] + '/model_performance.png')
    plt.show()

    val_images = []
    val_labels = []
    for i in range(20):
        image = np.full((100, 100, 3), 0, dtype=np.uint8)
        color = 200, 200, 200
        shape = np.random.choice(['circle', 'square', 'triangle'])
        if shape == 'circle':
            x = np.random.randint(20, 80)
            y = np.random.randint(20, 80)
            r = np.random.randint(10, 30)
            image = cv2.circle(image, (x, y), r, color, -1)
            label = 0
        elif shape == 'square':
            x1 = np.random.randint(20, 60)
            y1 = np.random.randint(20, 60)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(20, 40)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            label = 1
        elif shape == 'triangle':
            x1 = np.random.randint(20, 60)
            y1 = np.random.randint(20, 60)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(20, 40)
            x3 = x1 + np.random.randint(20, 40)
            y3 = y1 + np.random.randint(20, 40)
            cv2.fillPoly(image, [np.array([[x1, y1], [x2, y2], [x3, y3]])], fill_c)
            label = 2
        val_images.append(image)
        val_labels.append(label)

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    pred = model.predict(val_images)
    rows = 5
    cols = len(val_labels) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.imshow(val_images[i])
        ax.set_title(f"{val_labels[i]}: {pred[i].argmax()}")
        ax.axis('off')
    plt.tight_layout()
    plt.title(f"Accuracy: {(val_labels == pred.argmax(axis=1)).sum() / len(val_labels)}")
    plt.show()
    print(val_labels)
    print(pred.argmax(axis=1))


if __name__ == '__main__':
    main()
