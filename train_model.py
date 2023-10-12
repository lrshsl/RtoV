from typing import Optional
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from rtov.lazy_dataset import LazyDataset
from model_analytics import ModelAnalytics
from rtov.model import RtoVMainModel
from model_utils import ModelParameters, get_dataloader, get_dataset
from model_utils import get_model, load_checkpoint, save_checkpoint
import rtov.utils.utils as utils
import constants


# TrainParameters {{{
class TrainParameters(ModelParameters):
    epochs: int
    learning_rate: float
    learning_momentum: float
    # weight_decay: float

    def __init__(self,
                 num_workers: int = 8,
                 batch_size: int = 4,
                 samples_per_epoch: int = 2000,
                 epochs: int = 10,
                 learning_rate: float = 0.00005,
                 learning_momentum: float = 0.9,
                 # weight_decay: float = 0.0
                 ) -> None:
        super().__init__(num_workers, batch_size, samples_per_epoch)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        # self.weight_decay = weight_decay
# }}}


# {{{
def train_model(base_model: Optional[str],
                train_parameters: TrainParameters = TrainParameters(),
                hide_plot: bool = True,
                model_save_name: Optional[str] = None,
                ) -> None:
    """Train a model and show some statistics."""

    # -- Prepare -- #

    # Dataset
    dataset: LazyDataset = get_dataset(train_parameters)

    # Dataloader
    dataloader: DataLoader = get_dataloader(dataset, train_parameters)

    # Model
    nnmodel: nn.Module = get_model(base_model)

    # Optimizer
    optimizer: optim.SGD = optim.SGD(nnmodel.parameters(),
                                     lr = train_parameters.learning_rate,
                                     momentum = train_parameters.learning_momentum)


    # -- Train -- #

    # Change to training mode
    nnmodel.train()

    # Load checkpoint
    checkpoint = load_checkpoint(model_save_name)

    # Train once
    train(nnmodel, optimizer, dataloader, train_parameters, checkpoint)

    # Show examples
    if not hide_plot:
        show_examples(nnmodel, dataloader, train_parameters)

    # Continue training as long as the user wants
    while input("Continue training? (y/N): ") == "y":

        # Ask for the number of epochs
        train_parameters.epochs = int(input("How many epochs: "))
         
        # Train
        train(nnmodel, optimizer, dataloader, train_parameters, checkpoint)

        # Show examples
        if not hide_plot:
            show_examples(nnmodel, dataloader, train_parameters)

    # Save
    if model_save_name is None and input("Save model? (y/N): ") == "y":
        model_save_name = input("Enter name of model: ")
        save_checkpoint(nnmodel, optimizer,
                        checkpoint,
                        additional_epochs = train_parameters.epochs,
                        save_name = model_save_name)
    elif model_save_name is not None:
        save_checkpoint(nnmodel, optimizer,
                        checkpoint,
                        additional_epochs = train_parameters.epochs,
                        save_name = model_save_name)


def train(model: nn.Module,
          optimizer: optim.SGD,
          dataloader: DataLoader,
          parameters: TrainParameters,
          checkpoint: dict) -> tuple[list[float], list[float]]:
    """Train the model."""

    # Collections for the loss data
    shape_losses = []
    points_losses = []
    # color_losses = []

    start_epoch = checkpoint['epoch']
    for epoch in range(start_epoch, start_epoch + parameters.epochs):

        # Zero the running losses
        shape_running_loss: float = 0.0
        points_running_loss: float = 0.0
        # color_running_loss: float = 0.0

        i: int = 0       # If dataset was be empty

        for i, batch in enumerate(dataloader):

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Unpack the batch
            images, shape_labels, points_labels = batch

            # Forward pass
            shape_preds, point_preds = model(images)

            # Compute the loss
            shape_loss = constants.SHAPE_LOSS_FN(shape_preds, shape_labels)
            point_loss = constants.POINT_LOSS_FN(point_preds, points_labels)

            # Backpropagation
            shape_loss.backward(retain_graph=True)
            point_loss.backward(retain_graph=True)

            # Optimize
            optimizer.step()

            # Statistics
            shape_running_loss += shape_loss.item()
            points_running_loss += point_loss.item()

        # Epoch statistics
        n_images = i
        epoch_shape_loss = shape_running_loss / n_images
        epoch_points_loss = points_running_loss / n_images # Here None's are included as 0
        print(f'[{epoch + 1:3}] shape loss: {epoch_shape_loss:.3f}, points loss: {epoch_points_loss:.3f}')
        shape_losses.append(epoch_shape_loss)
        points_losses.append(epoch_points_loss)

    return shape_losses, points_losses



def show_examples(nnmodel: nn.Module,
                 dataloader: DataLoader,
                 train_parameters: TrainParameters) -> None:
    """Show some examples in a plot."""

    # Create an ModelAnalytics instance
    analytics: ModelAnalytics = ModelAnalytics(
        model = nnmodel,
        dataloader = dataloader,
        batch_size = train_parameters.batch_size)
    analytics.show_examples(result_save_path=None, hide=False)



