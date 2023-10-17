import os
from typing import Optional, Type
from abc import ABC

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import constants
from model.model import RtoVMainModel
from data.lazy_dataset import LazyDataset
import utils.utils as utils

# ModelParameters {{{
class ModelParameters(ABC):
    """Abstract base class for the model parameters."""

    # Number of workers, that serve the dataset to the gpu / cpu
    num_workers: int

    # Number of images that are processed at once
    batch_size: int

    # Total number of images
    num_samples: int

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 num_samples: int) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_samples = num_samples
# }}}

# Helper functions {{{
def get_dataloader(dataset: Dataset, parameters: ModelParameters) -> DataLoader:
    """Create a dataloader for testing for the given dataset."""
    return DataLoader(
            dataset = dataset,
            batch_size = parameters.batch_size,
            num_workers = parameters.num_workers,
            shuffle = True)         # Shuffling the dataset shouldn't matter in this case, but might when changing the implementation of the dataset


def get_dataset(parameters: ModelParameters) -> LazyDataset:
    """Create a new dataset with the given number of images."""
    return LazyDataset(
            img_dim = constants.IMG_DIM,
            num_samples = parameters.num_samples,
            transform = utils.transform_fn)


def get_model(base_model: Optional[str] = None,
              model_type: Type[nn.Module] = RtoVMainModel,
              ) -> nn.Module:
    """Load a model. If no path is given, use the default model."""

    # Create a model instance
    nnmodel: nn.Module = model_type()

    # If no path is given, return the new model
    if base_model is None:
        print(f'---<< [Log] Using new model >>--')
        return nnmodel

    # Make the name to a path ('m1' -> 'saved_models/m1.pth')
    load_model_path = utils.to_model_path(base_model)

    # If it doesn't exist, throw an error
    if not os.path.exists(load_model_path):
        raise FileNotFoundError(f'No model found at {load_model_path}')

    # Load the model
    nnmodel.load_state_dict(torch.load(load_model_path)['model_state_dict'])
    print(f'---<< [Log] Using model from {load_model_path} >>--')

    # Return it
    return nnmodel


def load_checkpoint(model_name: Optional[str] = None) -> dict:
    """Load a checkpoint."""

    # If no path is given, use the default
    if model_name is None:
        model_name = constants.DEFAULT_MODEL

    # Create the path from the name
    path: str = utils.to_model_path(model_name)

    # If the path doesn't exist, throw an error
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}')

    # Load the checkpoint
    return torch.load(path)


def save_checkpoint(model: nn.Module,
                    optimizer: optim.SGD,
                    previous_checkpoint: dict,
                    additional_epochs: int,
                    save_name: str) -> None:

    # Create the checkpoint
    checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': previous_checkpoint['epoch'] + additional_epochs}

    # Create the path from the name
    path: str = utils.to_model_path(save_name)

    # Save the checkpoint
    torch.save(checkpoint, path)

# }}}
