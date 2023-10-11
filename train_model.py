
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, assert_type

from rtov.lazy_dataset import LazyDataset
from rtov.model import RtoVMainModel
from model_analytics import ModelAnalytics
import constants


_transform = transforms.Compose([
    # torchvision.transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class TrainParameters:
    batch_size: int
    learning_rate: float
    learning_momentum: float
    # weight_decay: float
    num_workers: int

    def __init__(self,
                 batch_size: Optional[int],
                 learning_rate: Optional[float],
                 learning_momentum: Optional[float],
                 weight_decay: Optional[float],
                 num_workers: Optional[int]) -> None:
        self.batch_size = batch_size if batch_size is not None else 4       # Number of images that are processed at once
        self.learning_rate = learning_rate if learning_rate is not None else 0.00005
        self.learning_momentum = learning_momentum if learning_momentum is not None else 0.9
        # self.weight_decay = weight_decay if weight_decay is not None else 0.00005
        self.num_workers = num_workers if num_workers is not None else 8    # ~= Number of cores


def train_model(load_model: Optional[str],
                train_parameters: TrainParameters,
                show_examples: bool = True,
                save_path: Optional[str] = None,
                dataset: Optional[LazyDataset] = None,
                model_type: type[torch.nn.Module] = RtoVMainModel,
                ) -> None:
    """Evaluate the model and show random examples."""
    # Either load a model or create one. Can't have both arguments
    assert load_model is None or model_type == RtoVMainModel

    # Dataset
    if dataset is None:
        dataset = LazyDataset(
                img_dim = constants.IMG_DIM,
                num_samples = 2000,
                transform = _transform)
    assert_type(dataset, LazyDataset) # Tell the type checker that dataset isn't None

    # Dataloader
    dataloader: DataLoader = DataLoader(
            dataset = dataset,
            batch_size = train_parameters.batch_size,
            num_workers = train_parameters.num_workers,
            shuffle = True)

    # Model
    nnmodel = model_type()


