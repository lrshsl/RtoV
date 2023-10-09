from rtov.lazy_dataset import LazyDataset
from rtov.model import RtoVMainModel
from rtov.utils.to_model_path import to_model_path
from rtov.utils.shapes import draw_shape, SHAPE_NAMES
import constants

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Optional, assert_type

# Which device to use
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


_transform = transforms.Compose([
    # torchvision.transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test Parameters {{{
class TestParameters:
    batch_size: int
    num_workers: int
    shape_loss: nn.Module
    points_loss: nn.Module
    # color_loss: nn.Module

    def __init__(self,
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 shape_loss: Optional[nn.Module] = None,
                 points_loss: Optional[nn.Module] = None,
                 # color_loss: Optional[nn.Module],
                 ) -> None:
        self.batch_size = batch_size if batch_size is not None else 4       # Number of images that are processed at once
        self.num_workers = num_workers if num_workers is not None else 8    # ~= Number of cores
        self.shape_loss = shape_loss if shape_loss is not None else nn.CrossEntropyLoss()
        self.points_loss = points_loss if points_loss is not None else nn.MSELoss()
        # self.color_loss = color_loss if color_loss is not None else nn.MSELoss()
# }}}

def test_model(_load_model: Optional[str] = None,
               num_images: int = 2000,
               test_parameters: TestParameters = TestParameters(),
               show_examples: bool = True,
               result_save_path: Optional[str] = None,
               dataset: Optional[LazyDataset] = None,
               ) -> None:
    """Evaluate the model and show random examples."""

    # Model
    load_model: str = _load_model if _load_model is not None else constants.DEFAULT_MODEL

    # Dataset
    if dataset is None:
        dataset = LazyDataset(
                img_dim = constants.IMG_DIM,
                num_samples = num_images,
                transform = _transform)
    assert_type(dataset, LazyDataset) # Tell the type checker that dataset isn't None

    # Dataloader
    dataloader: DataLoader = DataLoader(
            dataset = dataset,
            batch_size = test_parameters.batch_size,
            num_workers = test_parameters.num_workers,
            shuffle = False)

    # Model
    nnmodel: nn.Module = RtoVMainModel()
    load_model_path = to_model_path(load_model)
    if os.path.exists(load_model_path):
        print('Could not import model from {}. Falling back to default model <{}>'.format(load_model_path, constants.DEFAULT_MODEL))
        load_model_path = to_model_path(constants.DEFAULT_MODEL)
        assert os.path.exists(load_model_path), 'Cannot import default model from {}'.format(load_model_path)
        nnmodel.load_state_dict(torch.load(load_model_path)['model_state_dict'])

    # Loss Functions
    shape_loss_fn: nn.Module = nn.CrossEntropyLoss()   # Cross entropy loss
    points_loss_fn: nn.Module = nn.MSELoss()           # Mean Quadratic Error Loss
    # color_loss_fn: nn.Module = nn.MSELoss()           # Mean Squared Error Loss


    # -- Evaluate -- #

    # Change to evaluation mode
    nnmodel.eval()

    # Determine performance
    images, shape_labels, point_labels = next(iter(dataloader))
    shape_preds, points_preds = nnmodel(images)

    shape_accuracy = (shape_preds.argmax(dim=1) == shape_labels).float().mean()
    points_accuracy = 1 / sum(points_preds - point_labels)
    points_loss = points_loss_fn(points_preds, point_labels)

    print(f'Shape Accuracy: {shape_accuracy} (Rate of correct predictions)')
    print(f'Points Loss: {points_loss}, Points Accuracy: {points_accuracy}')

    # Show examples
    if show_examples:
        import matplotlib.pyplot as plt

        # Prepare the plotting window
        cols: int = 5                       # Middle column empty
        rows: int = 4
        num_image_pairs: int = 2 * 4        # 8 label-prediction pairs
        fig, axis = plt.subplots(rows, cols, figsize=(10, 15))

        # Get data from batches, untill there is enough
        images: np.ndarray = np.zeros((num_image_pairs,
                                       int(constants.IMG_DIM[2]),
                                       int(constants.IMG_DIM[0]),
                                       int(constants.IMG_DIM[1])))
        shape_labels: np.ndarray = np.zeros(num_image_pairs)
        point_labels: np.ndarray = np.zeros((num_image_pairs, constants.SHAPE_SPEC_MAX_SIZE))

        # Fetch the data
        for i in range(0, num_image_pairs, test_parameters.batch_size):
            tmp_images, tmp_shape_labels, tmp_point_labels = next(iter(dataloader))
            images[i:i + test_parameters.batch_size] = np.array(list(tmp_images))
            shape_labels[i:i + test_parameters.batch_size] = np.array(list(tmp_shape_labels))
            point_labels[i:i + test_parameters.batch_size] = np.array(list(tmp_point_labels))

        # Let the model predict
        shape_preds, points_preds = nnmodel(torch.Tensor(images))


        # Don't show the axes
        for ax in axis:
            for spot in ax:
                spot.axis('off')


        # Display results
        for i in range(num_image_pairs):

            # Where to put the image
            if i < 4:
                org_spot = axis[i % 4, 0]
                pred_spot = axis[i % 4, 1]
            else:
                org_spot = axis[i % 4, 3]
                pred_spot = axis[i % 4, 4]

            # Actual values
            shape_label = SHAPE_NAMES[shape_labels[i].argmax()]
            point_label = [int(l) for l in point_labels[i]]

            # Predicted values
            shape_pred = shape_preds[i].argmax()
            points_pred = [int(p) for p in points_preds[i]]

            # Prepare the original image (Ground truth)
            org_image = images[i] * 255                     # From 0 to 1 -> 0 to 255
            org_image = org_image.astype(np.uint8)          # Now we can use unsigned 8-bit intergers
            org_image = np.transpose(org_image, (1, 2, 0))  # From CHW to HWC   ( Height-Width-Channel )
            org_image = org_image.copy()                    # Otherwise it isn't writable

            # Create the predicted image
            pred_image: np.ndarray = np.full(org_image.shape, 255, dtype=np.uint8)                  # White canvas
            draw_shape(pred_image, shape_pred, np.array(points_pred, np.int32), color=(0, 0, 0))    # Draw the predicted shape

            # Show the original image
            org_spot.imshow(org_image)
            org_spot.set_title(f'{shape_label}\n{point_label}')

            # Show the predicted image
            pred_spot.imshow(pred_image)
            pred_spot.set_title(f'.\n{SHAPE_NAMES[shape_pred]}\n{points_pred}')

        # Show everything
        plt.tight_layout()
        plt.show()

        if result_save_path is not None:
            plt.savefig(result_save_path)


