from utils.vec import Vec3
from lazy_dataset import LazyDataset
from utils.shapes import shape_names, Shapes

import torch
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


# Data Preparation {{{
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4
epochs = 10

trainset = LazyDataset(img_dim=Vec3(32, 32, 3), num_samples=2000, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = LazyDataset(img_dim=Vec3(32, 32, 3), num_samples=100, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
# }}}

# Model {{{
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
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
        )
        self.p1_fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)     # Flatten all dimensions except batch
        # shape = self.shape_fc(x)
        p1 = self.p1_fc(x)
        return p1

net = Net()
# }}}

# Loss & optimizer {{{
import torch.optim as optim
import torchmetrics

def p_loss_fn(outputs, labels):
    loss = nn.MSELoss()(outputs, labels[:, :2])
    return loss

criterion = nn.MSELoss()
shape_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)
# }}}

# Training {{{
losses = []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    images_per_epoch = 0
    for i, (inputs, shape_labels,
            point_labels) in enumerate(trainloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # shape_loss = criterion(outputs, labels)
        loss = p_loss_fn(outputs, point_labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        images_per_epoch += 1

    print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / 2000:.3f}')
    losses.append(running_loss / images_per_epoch)
    running_loss = 0.0


plt.plot(losses)
plt.show()
print('Finished Training')
# }}}

# Testing {{{
import cv2

images, shape_labels, point_labels = next(iter(testloader))
images_list = list(images)
shape_labels_list = list(shape_labels)
point_labels_list = list(point_labels)
preds = net(images)

cols = 4
rows = len(shape_labels) // cols

fig, axis = plt.subplots(rows, cols, figsize=(10, 10))

for i in range(len(shape_labels_list)):
    ax = axis[i // cols, i % cols]
    pred = [int(p) for p in preds[i]]
    image = images_list[i].numpy() * 255
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image = image.copy() # Other is not writable

    cv2.circle(image, pred, 1, (0, 0, 255), -1)
    ax.imshow(image)
    label = [int(l) for l in point_labels_list[i]]
    ax.set_title(f"{label}\n{pred}")
    ax.axis('off')

plt.tight_layout()
plt.show()
print('Finished testing')

# test_accuracy = (preds.argmax(dim=1) == labels).sum().item()

# if input(f'Performance: {test_accuracy}\nSave model? (y/N)') == 'y':
#     name = input('Name: ') + '.pt'
#     torch.save(net.state_dict(), 'model.pt')

# if input('Continue training? (y/N)') == 'y':
    # epochs = input('How many epochs?')
    # train(net, criterion, optimizer, epochs)

# net.load_state_dict(torch.load('model.pt'))
# }}}







