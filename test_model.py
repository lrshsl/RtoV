import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from typing import Optional

from rtov.lazy_dataset import LazyDataset
from rtov.model import RtoVMainModel
import rtov.utils.utils as utils
from model_analytics import ModelAnalytics
import constants


# Test Parameters {{{
class TestParameters:

    # Number of images that are processed at once
    batch_size: int

    # Total number of images
    total_num_samples: int

    # Number of workers, that serve the dataset to the gpu / cpu
    num_workers: int

    # Shape loss function (default is Cross Entropy Loss)
    shape_loss: nn.Module

    # Point loss function (default is Mean Squared Error)
    points_loss: nn.Module
    # color_loss: nn.Module

    def __init__(self,
                 batch_size: Optional[int] = None,
                 total_num_samples: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 shape_loss: Optional[nn.Module] = None,
                 points_loss: Optional[nn.Module] = None,
                 # color_loss: Optional[nn.Module],
                 ) -> None:
        # Initialize the attributes, from parameter or default
        self.batch_size = batch_size if batch_size is not None else 4
        self.total_num_samples = total_num_samples if total_num_samples is not None else 2000
        self.num_workers = num_workers if num_workers is not None else 8                        # Should be ~Number of cores
        self.shape_loss = shape_loss if shape_loss is not None else nn.CrossEntropyLoss()
        self.points_loss = points_loss if points_loss is not None else nn.MSELoss()
        # self.color_loss = color_loss if color_loss is not None else nn.MSELoss()
# }}}

# Helper Functions {{{
def get_dataloader(dataset: Dataset, test_parameters: TestParameters) -> DataLoader:
    """Create a dataloader for testing for the given dataset."""
    return DataLoader(
            dataset = dataset,
            batch_size = test_parameters.batch_size,
            num_workers = test_parameters.num_workers,
            shuffle = True)         # Shuffling the dataset shouldn't matter in this case, but might when changing the implementation of the dataset

def get_dataset(test_parameters: TestParameters = TestParameters()) -> LazyDataset:
    """Create a new dataset with the given number of images."""
    return LazyDataset(
            img_dim = constants.IMG_DIM,
            num_samples = test_parameters.total_num_samples,
            transform = utils.transform_fn)


def get_model(load_model: Optional[str] = None) -> nn.Module:
    nnmodel: nn.Module = RtoVMainModel()

    if load_model is None:
        load_model = constants.DEFAULT_MODEL

    # Make the name to a path ('m1' -> 'saved_models/m1.pth')
    load_model_path = utils.to_model_path(load_model)

    # If doesn't exist, print error
    if not os.path.exists(load_model_path):
        print('Could not import model from {}. Falling back to default model <{}>'.format(load_model_path, constants.DEFAULT_MODEL))

        # Try the default model instead
        load_model_path = utils.to_model_path(constants.DEFAULT_MODEL)

        # Fail is the default doesn't exist
        assert os.path.exists(load_model_path), 'Cannot import default model from {}'.format(load_model_path)
    
    # Load the model
    nnmodel.load_state_dict(torch.load(load_model_path)['model_state_dict'])
    print(f'---<< [Log] Loaded model from {load_model_path} >>--')

    # Return it
    return nnmodel
# }}}

# test_model {{{
def test_model(load_model: Optional[str] = None,
               test_parameters: TestParameters = TestParameters(),
               show_examples: bool = True,
               hide_plot: bool = False,
               result_save_path: Optional[str] = None,
               ) -> None:
    """Evaluate the model and show random examples."""

    # -- Prepare -- #

    # Dataset
    dataset: LazyDataset = get_dataset(test_parameters)

    # Dataloader
    dataloader: DataLoader = get_dataloader(dataset, test_parameters)

    # Model
    nnmodel: nn.Module = get_model(load_model)


    # -- Evaluate -- #

    # Change to evaluation mode
    nnmodel.eval()

    # Create an ModelAnalytics instance
    analytics: ModelAnalytics = ModelAnalytics(
        model = nnmodel,
        dataloader = dataloader,
        batch_size = test_parameters.batch_size
    )

    # Determine performance
    analytics.print_model_performance(test_parameters.total_num_samples)

    # Show examples
    if show_examples:
        analytics.show_examples(result_save_path, hide=hide_plot)
# }}}


