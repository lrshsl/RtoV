import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt

from typing import Optional

from rtov.utils.shapes import draw_shape, Shapes, SHAPE_NAMES
import rtov.utils.utils as utils

class ModelAnalytics:
    model: nn.Module
    dataloader: DataLoader
    batch_size: int

    # [public] __init__ {{{
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 batch_size: int) -> None:
        self.model = model
        self.dataloader = dataloader
        self.batch_size = batch_size
    # }}}

    # _unpack_and_predict {{{
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

    # Plotting {{{
    def _prepare_plot(self) -> tuple:
        """Prepare the plotting window."""
        cols, rows = 5, 4
        fig, axis = plt.subplots(rows, cols, figsize=(15, 10))
        
        # Hide axes
        for col in range(cols):
            for row in range(rows):
                ax = axis[row, col]
                ax.set_axis_off()

        return fig, axis

    def _make_plot(self, result_save_path: Optional[str], hide: bool) -> None:

        # Prepare the plotting window
        plt.tight_layout()

        # Show the plot
        if not hide:
            plt.show()

        # Save the plot as a image
        if result_save_path is not None:
            plt.savefig(result_save_path)
    # }}}

    # [public] print_model_performance {{{
    def print_model_performance(self, num_samples: int) -> None:

        # Running statistics
        self.correct_shapes: int = 0
        self.points_error: int = 0
        self.running_sample_count: int = 0

        nbatches: int = num_samples // self.batch_size

        # Reset the dataloader as many times as necessary
        for _ in range(0, num_samples, self.batch_size * len(self.dataloader)):

            # Retrieve and process the batches from the dataloader
            if not self._print_performance_once(nbatches, num_samples):
                break

        print('\n\n----- Statistics -----\n')
        print(f'Correct shapes: {self.correct_shapes} / {num_samples} --> {self.correct_shapes / num_samples * 100}%')
        print(f'Points error: {self.points_error} / {num_samples} --> {self.points_error / num_samples} px per shape')
        if self.running_sample_count != num_samples:
            print(f"[Error] Expected {num_samples} samples to be processed, got {self.running_sample_count}")


    def _print_performance_once(self, nbatches: int, num_samples: int) -> bool:

        # Go through the batches
        for i, batch_data in enumerate(self.dataloader):

            # Unpack data
            imgs, shapes, pts = batch_data

            # Let the model predict
            shape_preds, points_preds = self.model(torch.Tensor(imgs))

            # Batch statistics
            batch_correct_shapes = (shape_preds.argmax(dim=1) == torch.Tensor(shapes)).float().sum().item()
            batch_points_error = (points_preds - torch.Tensor(pts)).abs().sum().item()
            print(f'[Batch {i + 1} / {nbatches}] Correct shapes: {batch_correct_shapes} / {self.batch_size}, Points error: {batch_points_error}')

            # Update statistics
            self.correct_shapes += batch_correct_shapes
            self.points_error += batch_points_error
            self.running_sample_count += self.batch_size

            if self.running_sample_count >= num_samples:
                print('---<< [Log] "Print model performance" done >>--')
                return False

        # Not necessarily done
        return True
    # }}}

    # [public] show_examples {{{
    def show_examples(self, result_save_path: str | None, hide: bool) -> None:
        """Plot the labels and predictions pairwise next to each other."""

        # Prepare the plotting window
        _, axis = self._prepare_plot()

        num_image_pairs: int = 2 * 4        # 8 label-prediction pairs

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
            org_image: np.ndarray = utils.tensor2image(org_images)

            # Create the predicted image from the given specification
            pred_image: np.ndarray = np.full(org_image.shape, 255, dtype=np.uint8)                      # White canvas
            draw_shape(pred_image[:], shape_pred, np.array(points_pred, np.int32), color=(0, 0, 0))     # Draw the predicted shape

            # Show the original image
            org_spot.imshow(org_image)
            org_spot.set_title(f'{shape_label}\n{point_label}')

            # Show the predicted image
            pred_spot.imshow(pred_image)
            pred_spot.set_title(f'.\n{SHAPE_NAMES[shape_pred]}\n{points_pred}')

        # Show everything
        self._make_plot(result_save_path, hide)
    # }}}


