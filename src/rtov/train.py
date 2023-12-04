from typing import Optional
import os

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from data.lazy_dataset import LazyDataset
from model.model_analytics import ModelAnalytics
from model.model_utils import ModelParameters, get_dataloader, get_dataset
from model.model_utils import load_model, load_checkpoint, save_checkpoint, str2model_type
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
                 num_samples: int = 2000,
                 epochs: int = 10,
                 learning_rate: float = 0.00005,
                 learning_momentum: float = 0.9,
                 # weight_decay: float = 0.0
                 ) -> None:
        super().__init__(num_workers, batch_size, num_samples)
        self.epochs = epochs
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        # self.weight_decay = weight_decay
# }}}


# [public] train_model {{{
def train_model(base_model: Optional[str],
                train_parameters: TrainParameters = TrainParameters(),
                hide_plot: bool = True,
                model_save_name: Optional[str] = None,
                model_type_str: Optional[str] = None
                ) -> None:
    """Train a model and show some statistics."""

    # -- Prepare -- #

    # Dataset
    dataset: LazyDataset = get_dataset(train_parameters)

    # Dataloader
    dataloader: DataLoader = get_dataloader(dataset, train_parameters)

    # Model
    nnmodel: nn.Module = load_model(base_model, str2model_type(model_type_str))

    # Optimizer
    optimizer: optim.SGD = optim.SGD(nnmodel.parameters(),
                                     lr = train_parameters.learning_rate,
                                     momentum = train_parameters.learning_momentum)


    # -- Train -- #

    # Change to training mode
    nnmodel.train()

    # Load checkpoint
    if model_save_name is not None and os.path.exists(model_save_name):
        checkpoint: dict = load_checkpoint(model_save_name)
    elif base_model is not None and os.path.exists(base_model):
        checkpoint: dict = load_checkpoint(base_model)
    else:
        checkpoint: dict = {
                "model": nnmodel.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 0 }

    # Train once
    _train(nnmodel, optimizer, dataloader, train_parameters, checkpoint, epochs_done = 0)

    # Show examples
    if not hide_plot:
        _show_examples(nnmodel, dataloader, train_parameters)

    # Continue training as long as the user wants
    while input("Continue training? (y/N): ") == "y":

        # Adapt parameters
        epochs_done: int = train_parameters.epochs
        epochs = input("How many epochs: ")
        lr = input("Learning rate (default: 0.00005): ")
        train_parameters.epochs = int(epochs) if epochs else 10
        train_parameters.learning_rate = float(lr) if lr else 0.00005

        # Train
        _train(nnmodel, optimizer, dataloader, train_parameters, checkpoint, epochs_done)

        # Show examples
        if not hide_plot:
            _show_examples(nnmodel, dataloader, train_parameters)

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
# }}}

# _train {{{
def _train(model: nn.Module,
           optimizer: optim.SGD,
           dataloader: DataLoader,
           parameters: TrainParameters,
           checkpoint: dict,
           epochs_done: int,
           ) -> tuple[list[float], list[float]]:
    """Train the model."""

    # Collections for the loss data
    shape_losses = []
    points_losses = []
    # color_losses = []

    start_epoch = checkpoint['epoch'] + epochs_done
    for epoch in range(start_epoch, start_epoch + parameters.epochs):

        # Zero the running losses
        shape_running_loss: float = 0.0
        points_running_loss: float = 0.0
        # color_running_loss: float = 0.0

        i: int = 0       # If dataset was be empty

        for i, batch in enumerate(dataloader):  # Generate images and labels

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

            if verbose > 1:
                print("batch {}/{}, epochs: {}/{}".format(i, len(dataloader), epoch + 1, parameters.epochs))

        # Epoch statistics
        n_images = i
        epoch_shape_loss = shape_running_loss / n_images
        epoch_points_loss = points_running_loss / n_images # Here None's are included as 0
        if verbose > 0:
            print(f'[{epoch + 1:3}] shape loss: {epoch_shape_loss:.3f}, points loss: {epoch_points_loss:.3f}')
        shape_losses.append(epoch_shape_loss)
        points_losses.append(epoch_points_loss)

    return shape_losses, points_losses
# }}}

# _show_examples {{{
def _show_examples(nnmodel: nn.Module,
                 dataloader: DataLoader,
                 train_parameters: TrainParameters) -> None:
    """Show some examples in a plot."""

    # Create an ModelAnalytics instance
    analytics: ModelAnalytics = ModelAnalytics(
        model = nnmodel,
        dataloader = dataloader,
        batch_size = train_parameters.batch_size)
    analytics.show_examples(result_save_path=None, hide=False)
# }}}



