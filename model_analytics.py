from icecream import ic


import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt

from typing import Optional

from rtov.utils.shapes import draw_shape, Shapes, SHAPE_NAMES
import constants


class ModelAnalytics:
    model: nn.Module
    dataloader: DataLoader
    batch_size: int

    # __init__ {{{
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 batch_size: int) -> None:
        self.model = model
        self.dataloader = dataloader
        self.batch_size = batch_size
    # }}}

    # _get_samples {{{
    def _get_samples(self, num_samples: int) -> tuple[np.ndarray, ...]:
        """Requests `num_samples` samples from the dataloader and collects them into numpy arrays."""

        # Transpose the image shape (WHC -> CWH, where C = channels = 3 and W, H = width, height of images)
        img_shape: tuple[int, ...] = constants.IMG_DIM.as_int_tuple()
        img_shape = (
            img_shape[2],
            img_shape[0],
            img_shape[1])

        # Number of batches needed to achieve `num_samples` samples
        nbatches: int = num_samples // self.batch_size + 1

        # Empty arrays to be filled
        images: np.ndarray = np.zeros(
                (nbatches, self.batch_size, *img_shape),
                dtype=np.uint8)
        shape_labels: np.ndarray = np.zeros(
                (nbatches, self.batch_size),
                dtype=np.int32)
        point_labels: np.ndarray = np.zeros(
                (nbatches, self.batch_size, constants.SHAPE_SPEC_MAX_SIZE),
                dtype=np.float32)

        # Fill the arrays
        for i in range(nbatches):
            for data in self.dataloader:

                # Unpack data
                imgs, shapes, pts = data

                # In case it doesn't add up
                if imgs.shape[0] != self.batch_size:
                    images[i, :imgs.shape[0]] = imgs.clone()
                    break

                # Write to arrays
                images[i] = imgs.clone()
                shape_labels[i] = shapes.clone()
                point_labels[i] = pts.clone()

        return images, shape_labels, point_labels
    # }}}

    # print_model_performance {{{
    def print_model_performance(self, num_samples: int) -> None:

        # Prepare data
        images, shape_labels, point_labels = self._get_samples(num_samples)

        # Running statistics
        correct_shapes: int = 0
        points_error: int = 0

        nbatches = num_samples // self.batch_size

        # Run the model per batch
        for n in range(0, nbatches + 1, len(self.dataloader)):
            for i, batch_data in enumerate(self.dataloader):

                # Unpack data
                imgs, shapes, pts = batch_data

                # Let the model predict
                shape_preds, points_preds = self.model(torch.Tensor(imgs))

                # Batch statistics
                batch_correct_shapes = (shape_preds.argmax(dim=1) == torch.Tensor(shapes)).float().sum().item()
                batch_points_error = (points_preds - torch.Tensor(pts)).abs().sum().item()
                print(f'[Batch {i} / {nbatches}] Correct shapes: {batch_correct_shapes} / {self.batch_size}, Points error: {batch_points_error}')

                # Update statistics
                correct_shapes += batch_correct_shapes
                points_error += batch_points_error

        print('\n\n----- Statistics -----\n')
        print(f'Correct shapes: {correct_shapes} / {num_samples} --> {correct_shapes / num_samples * 100}%')
        print(f'Points error: {points_error} / {num_samples} --> {points_error / num_samples} per shape')
    # }}}

    # show_examples {{{
    def show_examples(self, result_save_path: Optional[str]) -> None:
        """Plot the labels and predictions pairwise next to each other."""

        # Prepare the plotting window
        cols: int = 5                       # Middle column empty
        rows: int = 4
        num_image_pairs: int = 2 * 4        # 8 label-prediction pairs
        fig, axis = plt.subplots(rows, cols, figsize=(10, 15))

        # Hide the axes
        for ax in axis:
            for spot in ax:
                spot.axis('off')

        # Create an iterator
        labels_and_predictions = self._unpack_and_predict(self.dataloader, self.model)

        # Go through each image spot
        for i in range(num_image_pairs):

            # Unpack data from the iterator
            org_images, shape_labels, point_labels, shape_preds, points_preds = next(labels_and_predictions)

            # Where to put the image
            if i < 4:
                org_spot = axis[i % 4, 0]   # Column 1
                pred_spot = axis[i % 4, 1]  # Column 2
            else:
                org_spot = axis[i % 4, 3]   # Column 4
                pred_spot = axis[i % 4, 4]  # Column 5

            # Actual values
            shape_label: str = SHAPE_NAMES[shape_labels.item()]
            point_label: list[int] = [int(l) for l in point_labels]

            # Predicted values
            shape_pred: Shapes = Shapes.from_int(shape_preds.argmax().item())
            points_pred = [int(p) for p in points_preds]

            # Prepare the original image (Ground truth)
            org_image = org_images.numpy().copy()           # Otherwise it isn't writable
            org_image = org_image * 255                     # From 0 to 1 -> 0 to 255
            org_image = org_image.astype(np.uint8)          # Now we can use unsigned 8-bit intergers
            org_image = np.transpose(org_image, (1, 2, 0))  # From CHW to HWC   ( Height-Width-Channel )

            # Create the predicted image from the given specification
            pred_image: np.ndarray = np.full(org_image.shape, 255, dtype=np.uint8)                      # White canvas
            ic(points_pred)
            draw_shape(pred_image[:], shape_pred, np.array(points_pred, np.int32), color=(0, 0, 0))     # Draw the predicted shape

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
    # }}}

    # unpack_and_predict {{{
    @staticmethod
    def _unpack_and_predict(dataloader, model):

         # Go through the batches
         for data in itt.chain.from_iterable(itt.repeat(dataloader)):

             # Predict per-batch
             shape_preds, point_preds = model(data[0])

             # Yield one element at a time
             for img, shape, pt, shape_pred, pt_pred in zip(
                     data[0], data[1], data[2], shape_preds, point_preds):

                 # Output the results
                 yield img, shape, pt, shape_pred, pt_pred
    # }}}

    # visualize_samples {{{
    def visualize_samples(self):
        images, *_ = self._get_samples(16)
        ic(images.shape)
        images = images.reshape(-1, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        print(images.shape)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Directly from the dataset
        images = next(iter(self.dataloader))[0]
        images = images.reshape(-1, 32, 32, 3)
        print(images.shape)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Even more directly
        for imgs, *_ in self.dataloader:
            imgs = imgs.reshape(-1, 32, 32, 3)
    # }}}



