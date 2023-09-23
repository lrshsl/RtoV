import torch
import torchvision
import torchvision.transforms as transforms

# Utils {{{
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from enum import IntEnum

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# class Shapes(IntEnum):
#     Line = 0
#     Triangle = 1
#     Rect = 2
Shapes = tuple
shapes = 'Line', 'Triangle', 'Rect'

Number = Union[int, float]

class Vec2:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Vec3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def as_tuple(self) -> tuple:
        return self.x, self.y, self.z

# }}}

# Data Generation {{{
import cv2

class DefaultPoints: # {{{
    @staticmethod
    def line(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((2, 2), dtype=np.int32)
        if random:
            rng = np.random.default_rng()
            pts[:, 0] = rng.integers(
                low = 0, high = dim.x, size = 2)    # x values
            pts[:, 1] = rng.integers(
                low = 0, high = dim.y, size = 2)    # y values
            return pts

        # Diagonal line through the image from (0, 0)
        pts[0, 0] = dim.x
        pts[0, 1] = dim.y
        return pts

    @staticmethod
    def triangle(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((3, 2), dtype=np.int32)

        # Random three points
        if random:
            rng = np.random.default_rng()
            pts[:, 0] = rng.integers(
                low = 0, high = dim.x, size = 3)    # x values
            pts[:, 1] = rng.integers(
                low = 0, high = dim.y, size = 3)    # y values
            return pts

        # (More or less) regular triangle
        pts[0, 0] = dim.x // 4      # Bottom left
        pts[0, 1] = dim.y // 4
        pts[1, 0] = dim.x * 3 // 4  # Bottom right
        pts[1, 1] = dim.y // 4
        pts[2, 0] = dim.x // 2      # Top mid
        pts[2, 1] = dim.y * 3 // 4
        return pts

    @staticmethod
    def rectangle(dim: Vec2, random: bool = True) -> np.ndarray:
        pts = np.zeros((4, 2), dtype=np.int32)

        if random:
            rng = np.random.default_rng()
            # Top left corner
            pts[0, 0] = rng.integers(0, dim.x * 3 // 4)  # Leave enough free space
            pts[0, 1] = rng.integers(0, dim.y * 3 // 4)
            # Bottom left corner, spans a rect with random height
            pts[1, 0] = pts[0, 0]
            pts[1, 1] = rng.integers(pts[0, 1], dim.y)
            # Top right corner -> random width
            pts[3, 0] = rng.integers(pts[0, 0], dim.x)
            pts[3, 1] = pts[0, 1]
            # Bottom right corner
            pts[2, 0] = pts[3, 0]
            pts[2, 1] = pts[1, 1]
            return pts

        # Regular rectangle
        pts[[0, 1], 0] = dim.x // 4     # Left edge
        pts[[0, 3], 1] = dim.y // 4     # Top edge
        pts[[2, 3], 0] = dim.x * 3 // 4 # Right edge
        pts[[1, 2], 1] = dim.y * 3 // 4 # Bottom edge
        return pts
# }}}

# fn draw_on_image {{{
def draw_on_image(img: np.ndarray, shape: Shapes, color: int = 0) -> np.ndarray:
    """Mutates `img` in-place"""
    if shape == 'Line':
        pts = DefaultPoints.line(
            Vec2(img.shape[0], img.shape[1]), random=True)
        cv2.line(img, pts[0], pts[1], color)
        return pts
    elif shape == 'Triangle':
        pts = DefaultPoints.triangle(
            Vec2(img.shape[0], img.shape[1]), random=True)
    elif shape == 'Rect':
        pts = DefaultPoints.rectangle(
            Vec2(img.shape[0], img.shape[1]), random=True)
    else:
        raise ValueError("Unknown or unimplemented shape: {}".format(shape))
    cv2.fillPoly(img, [pts], color=color)
    return pts
# }}}

# DataSet {{{
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader

class LazyDataset(Dataset):
    def __init__(self, img_dim: Vec3, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.features = np.zeros(img_dim.as_tuple(), dtype=np.float32)
        self.tensor = np.zeros((img_dim.z, img_dim.x, img_dim.y), dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        self.features[:] = np.full(self.features.shape, 255)    # Overwrite the memory directly to (hopefully) reduce memory => fewer GC cycles
        # shape = np.random.default_rng().integers(len(shapes))
        shape = np.random.choice(shapes)
        pts = draw_on_image(self.features, shape, color=0)
        if self.transform:
            self.tensor = self.transform(self.features)
            # Don't overwrite with [:]:
            #   DataLoader uses pointers, which are referenced per-batch, not per-item.
            #   If overwritten, the pointers all point to the last element by the time they are accessed
        # TODO: Calc size + first point
        return self.tensor, shapes.index(shape)
# }}}

# }}}

# Data Preparation {{{
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

trainset = LazyDataset(img_dim=Vec3(32, 32, 3), num_samples=1000, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = LazyDataset(img_dim=Vec3(32, 32, 3), num_samples=100, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# }}}

# Model {{{
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# }}}

# Loss & optimizer {{{
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# }}}

# Training {{{
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
# }}}

# Testing {{{
images, labels = next(iter(testloader))

# print images (modify this part as needed)
print('GroundTruth: ', ' '.join(f'{shapes[labels[j]]:5s}' for j in range(batch_size)))
imshow(torchvision.utils.make_grid(images))


# dataiter = iter(testloader)
# images, labels = next(dataiter)

# # print images
# print('GroundTruth: ', ' '.join(f'{shapes[labels[j]]:5s}' for j in range(4)))
# imshow(torchvision.utils.make_grid(images))
# }}}

