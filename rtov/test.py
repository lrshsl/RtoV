from typing import Optional
from torch import nn

from torch.utils.data import DataLoader

from data.lazy_dataset import LazyDataset
from model.model_analytics import ModelAnalytics
from model.model_utils import ModelParameters, get_dataloader, get_dataset, get_model
import constants


# TestParameters {{{
class TestParameters(ModelParameters):

    # Number of images that are processed at once
    batch_size: int

    # Total number of images
    total_num_samples: int

    # Number of workers, that serve the dataset to the gpu / cpu
    num_workers: int

    def __init__(self,
                 num_workers: int = 8,
                 batch_size: int = 4,
                 total_num_samples: int = 2000,
                 ) -> None:
        super().__init__(num_workers, batch_size, total_num_samples)
# }}}

# test_model {{{
def test_model(base_model: Optional[str] = None,
               test_parameters: TestParameters = TestParameters(),
               hide_plot: bool = False,
               demonstration_save_path: Optional[str] = None,
               ) -> None:
    """Evaluate the model and show random examples."""

    # -- Prepare -- #

    # Dataset
    dataset: LazyDataset = get_dataset(test_parameters)

    # Dataloader
    dataloader: DataLoader = get_dataloader(dataset, test_parameters)

    # Model
    if base_model is None:
        base_model = constants.DEFAULT_MODEL
    nnmodel: nn.Module = get_model(base_model)


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
    analytics.print_model_performance(test_parameters.num_samples)

    # Show examples
    if demonstration_save_path is not None or not hide_plot:
        analytics.show_examples(demonstration_save_path, hide=hide_plot)
# }}}


