from rtov.utils.vec import Vec3
from rtov.lazy_dataset import LazyDataset
from rtov.utils.shapes import shape_names, Shapes

import torch
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


# TODO: make Recurrent Neural Network instead


# Parameters {{{
IMG_DIM = Vec3(32, 32, 3)

LEARNING_RATE = 0.00005
LEARNING_MOMENTUM = 0.9

BATCH_SIZE = 4
N_EPOCHS = 20
# }}}

# Data Preparation {{{
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = LazyDataset(img_dim=IMG_DIM, num_samples=2000, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = LazyDataset(img_dim=IMG_DIM, num_samples=100, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2) # 16: For 4x4 inspection
# }}}

# Model {{{
import torch.nn as nn
from rtov.lazy_dataset import MAX_DP_PER_SHAPE
# import torch.nn.functional as F


class Net(nn.Module):
    # Layers {{{
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.shape_fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, len(Shapes)),
        )
        self.circle_out = 3
        self.line_out = 5
        self.triangle_out = 6
        self.rectangle_out = 5
        self.fc_circle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.circle_out),        # Center coordinates + radius -> 3 numbers
        )
        self.fc_line = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.line_out),          # Two endpoints for line + width -> 5 coordinates
        )
        self.fc_triangle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.triangle_out),      # Three points for triangle -> 6 coordinates
        )
        self.fc_rectangle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.rectangle_out),     # One point + width + height + orientation -> 5 numbers
        )
    # }}}

    # Forward {{{
    def forward(self, x):

        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = torch.flatten(x, 1)     # Flatten all dimensions except batch

        # Predict shape
        shapes = self.shape_fc(x)

        # Predict color
        # TODO

        # Predict points
        pts = torch.Tensor()
        out = torch.zeros((x.shape[0], MAX_DP_PER_SHAPE), dtype=torch.float32)

        for shape_pred in shapes:
            # Output size depends on shape
            match shape_pred.argmax():
                case Shapes.Circle:
                    out[:, :self.circle_out] = self.fc_circle(x)
                case Shapes.Line:
                    out[:, :self.line_out] = self.fc_line(x)
                case Shapes.Triangle:
                    out[:, :self.triangle_out] = self.fc_triangle(x)
                case Shapes.Rectangle:
                    out[:, :self.rectangle_out] = self.fc_rectangle(x)
                case _:
                    raise ValueError("Unknown or unimplemented shape: {}".format(shape_pred))

        return shapes, out
    # }}}

# }}}

# Loss & optimizer {{{
import torch.optim as optim
import os

# Model
net = Net()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=LEARNING_MOMENTUM)

# Load checkpoint {{{
checkpoint = {
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 0,
}
use_pretrained = input("Use pre-trained model? (y/N) ") == "y"
while use_pretrained:
    load_path = 'saved_models/' + input('Enter name of model: ') + '.pth'

    if os.path.isfile(load_path):
        # Load checkpoint
        checkpoint = torch.load(load_path)
        break

    # Ask again if file doesn't exist
    print(f"File not found: {load_path}")
    use_pretrained = input("Use pre-trained model? (y/N) ") == "y"


# Load values
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# }}}

# Shape loss
shape_loss_fn = nn.CrossEntropyLoss()

# Points loss
points_loss_fn = nn.MSELoss()

# }}}

# Training {{{
shape_losses = []
points_losses = []
# color_losses = []

# Loop over the dataset multiple times (data is regenerated differenty each epoch)
start_epoch = checkpoint['epoch']
for epoch in range(start_epoch, start_epoch + N_EPOCHS):

    # Zero the running losses
    shape_running_loss = 0.0
    points_running_loss = 0.0
    # color_running_loss = 0.0

    i = 0       # If dataset was be empty
    for i, (inputs,
            shape_labels,
            point_labels,
            # color_labels
            ) in enumerate(trainloader):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward call
        shape_out, points_out = net(inputs)
        shape_loss = shape_loss_fn(shape_out, shape_labels)
        points_loss = points_loss_fn(points_out, point_labels)

        # Backpropagation
        shape_loss.backward(retain_graph=True)
        # if shape_loss.item() < 1:   # First teach to recognise the shape
        points_loss.backward(retain_graph=True)

        # Optimize
        optimizer.step()

        # Statistics
        shape_running_loss += shape_loss.item()
        points_running_loss += points_loss.item()

    # Epoch statistics
    n_images = i
    epoch_shape_loss = shape_running_loss / n_images
    epoch_points_loss = points_running_loss / n_images # Here None's are included as 0
    print(f'[{epoch + 1:3}] shape loss: {epoch_shape_loss:.3f}, points loss: {epoch_points_loss:.3f}')
    shape_losses.append(epoch_shape_loss)
    points_losses.append(epoch_points_loss)
    running_loss = 0.0


plt.plot(shape_losses)
plt.plot(points_losses)
plt.legend(['Shape loss', 'Points loss'])
plt.show()
print('Finished Training')
# }}}

# Testing {{{
import cv2

# Preparation {{{
# Get a random batch
images, shape_labels, point_labels = next(iter(testloader))

# Copy the data
images_list = list(images)              # Image: Ground truth
shape_labels_list = list(shape_labels)  # Labels (shapes)
point_labels_list = list(point_labels)  # Labels (points)

# Let the model predict
shape_preds, points_preds = net(images) # Predictions

# Prepare the plotting window
cols = 4
rows = len(shape_labels) // cols
fig, axis = plt.subplots(rows, cols, figsize=(10, 10))
# }}}

# Mark Shape {{{
def mark_shape(image, shape, data, color=(200, 200, 200)):
    match shape:
        case 'Circle':
            center = data[:2]
            r = data[2]
            cv2.circle(image, center, r, color, 1)
        case 'Line':
            p1 = data[:2]
            p2 = data[2:4]
            _width = data[4]
            cv2.line(image, p1, p2, color, 1)
        case 'Triangle':
            p1 = data[:2]
            p2 = data[2:4]
            p3 = data[4:6]
            cv2.line(image, p1, p2, color, 1)
            cv2.line(image, p2, p3, color, 1)
            cv2.line(image, p3, p1, color, 1)
        case 'Rectangle':
            pt = data[:2]
            w, h = data[2:4]
            _orientation = data[4]
            p1 = pt
            p2 = pt + np.array((w, 0))
            p3 = pt + np.array((w, h))
            p4 = pt + np.array((0, h))
            cv2.line(image, p1, p2, color, 1)
            cv2.line(image, p2, p3, color, 1)
            cv2.line(image, p3, p4, color, 1)
            cv2.line(image, p4, p1, color, 1)
# }}}

# Display results {{{
# Show each image on the window
for i in range(len(shape_labels_list)):
    # Where to put the image
    ax = axis[i // cols, i % cols]

    # Truth values
    shape_label = shape_names[shape_labels_list[i].argmax()]
    point_label = [int(l) for l in point_labels_list[i]]

    # Predicted
    shape_pred = shape_names[shape_preds[i].argmax()]
    points_pred = [int(p) for p in points_preds[i]]

    # Prepare the image
    image = images_list[i].numpy() * 255
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image = image.copy() # Other is not writable

    # Mark the predicted values
    # mark_shape(image, shape_label, data=point_label, color=(200, 200, 200))
    mark_shape(image, shape_pred, data=points_pred, color=(0, 0, 255))
    ax.imshow(image)

    # Write the comparison to the image
    ax.set_title(f"{shape_label}: {point_label}\n{shape_pred}: {points_pred}")
    ax.axis('off')

# Show everything
plt.tight_layout()
plt.show()
# }}}

print('Finished testing')
# }}}

# Save model {{{
if input("Save the model? [y/N]") == 'y':
    # Save the model
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': start_epoch + N_EPOCHS,
    }

    # The file path where the model will be saved to
    save_path = 'saved_models/' + input("Enter name of model: ") + '.pth'

    # Save the model
    torch.save(checkpoint, save_path)

    print(f"Model saved to '{save_path}'")
# }}}

