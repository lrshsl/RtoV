from rtov.lazy_dataset import LazyDataset
from rtov.utils.shapes import SHAPE_NAMES
from rtov.utils.to_model_path import to_model_path
from model_analytics import ModelAnalytics
import constants

IMG_DIM = constants.IMG_DIM

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


# Preparation {{{
transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create test/eval data
eval_set = LazyDataset(img_dim = IMG_DIM, num_samples = 100,
                       transform = transform,
                       seed = lambda: int(time.time() * 1000),
                       pad = constants.SHAPE_SPEC_MAX_SIZE)   # Seed makes sure the images are different
eval_loader = DataLoader(eval_set, batch_size = 16, shuffle = True, num_workers = 4)


testset1 = LazyDataset(img_dim = IMG_DIM, num_samples = 100,
                       transform = transform,
                       seed = lambda: int(time.time() * 1000),
                       pad = constants.SHAPE_SPEC_MAX_SIZE)   # Seed makes sure the images are different
testloader1 = DataLoader(testset1, batch_size = 16, shuffle = True, num_workers = 2) # 16: For 4x4 inspection

testset2 = LazyDataset(img_dim = IMG_DIM, num_samples = 100,
                       transform = transform,
                       seed = lambda: int(time.time() * 1000),
                       pad = constants.SHAPE_SPEC_MAX_SIZE)   # Seed makes sure the images are different
testloader2 = DataLoader(testset2, batch_size = 16, shuffle = True, num_workers = 2) # 16: For 4x4 inspection
# }}}

# Model {{{
from rtov.model import RtoVMainModel
net = RtoVMainModel()
net.load_state_dict(torch.load(to_model_path('default_model'))['model_state_dict'])

# }}}

# mark_shape {{{
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

# Evaluate {{{

analysis = ModelAnalytics(
    net,
    testloader1,
    batch_size = 16,
)
# analysis.visualize_samples()
analysis.show_examples(None, hide=False)
analysis.print_model_performance(num_samples = 100)

print('\n\n----- Analytics done -----\n')

correct_shapes: int = 0
points_error: int = 0
nbatches = len(eval_set) // 16

for i, batch_data in enumerate(testloader2):

    # Unpack data
    imgs, shapes, pts = batch_data

    # Predictions
    shape_preds, points_preds = net(imgs)

    # Batch statistics
    batch_correct_shapes = (shape_preds.argmax(dim=1) == torch.Tensor(shapes)).float().sum().item()
    batch_points_error = (points_preds - torch.Tensor(pts)).abs().sum().item()
    print(f'[Batch {i}] Correct shapes: {batch_correct_shapes}, Points error: {batch_points_error}')

    # Update statistics
    correct_shapes += batch_correct_shapes
    points_error += batch_points_error

print('\n\n----- Statistics -----\n')
print(f'Correct shapes: {correct_shapes} / {len(eval_set)} --> {correct_shapes / len(eval_set) * 100}%')
print(f'Points error: {points_error} / {len(eval_set)} --> {points_error / len(eval_set)} per shape')

# }}}

# Display results {{{

# Get a random batch
images, shape_labels, point_labels = next(iter(testloader))

# Let the model predict
shape_preds, points_preds = net(images) # Predictions

# Prepare the plotting window
cols = 4
rows = len(shape_labels) // cols
fig, axis = plt.subplots(rows, cols, figsize=(10, 10))

# Show each image on the window
print(shape_labels_list)
for i in range(len(shape_labels_list)):
    # Where to put the image
    ax = axis[i // cols, i % cols]

    # True values
    shape_label = SHAPE_NAMES[shape_labels_list[i].argmax()]
    point_label = [int(l) for l in point_labels_list[i]]

    # Predicted
    shape_pred = SHAPE_NAMES[shape_preds[i].argmax()]
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

print('Testing finished')
