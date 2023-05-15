import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate random images
images = []
labels = []
for i in range(100):
    image = np.zeros((100, 100, 3))
    shape = np.random.choice(['circle', 'square', 'triangle'])
    if shape == 'circle':
        x = np.random.randint(20, 80)
        y = np.random.randint(20, 80)
        r = np.random.randint(10, 30)
        image = cv2.circle(image, (x, y), r, (255, 255, 255), -1)
        label = 0
    elif shape == 'square':
        x1 = np.random.randint(20, 60)
        y1 = np.random.randint(20, 60)
        x2 = x1 + np.random.randint(20, 40)
        y2 = y1 + np.random.randint(20, 40)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        label = 1
    elif shape == 'triangle':
        x1 = np.random.randint(20, 80)
        y1 = np.random.randint(20, 80)
        x2 = x1 + np.random.randint(10, 30)
        y2 = y1 + np.random.randint(10, 30)
        x3 = x1 + np.random.randint(10, 30)
        y3 = y1 - np.random.randint(10, 30)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
        pts = pts.reshape((-1,1,2))
        image = cv2.polylines(image, [pts], True, (255, 255, 255), -1)
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
plt.show()

