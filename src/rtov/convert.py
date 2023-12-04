from typing import Optional
from torch import nn
import torch
import numpy as np
from PIL import Image

from model.model_utils import load_model, str2model_type
import utils.utils as utils
import utils.shapes as shapes
import constants


# Helper Functions {{{
def load_image(image_path: str):
    """Load a single image"""
    image = Image.open(image_path)
    np_image = np.array(image)
    return np_image

def write_image(image: np.ndarray, image_path: str):
    """Write a single image to a given path"""
    pil_image = Image.fromarray(image)
    pil_image.save(image_path)

def display_image(image: np.ndarray):
    """Display a single image"""
    pil_image = Image.fromarray(image)
    pil_image.show()
# }}}

def convert_image(input_image_path: str,
                  output_path: str,
                  output_format: str = 'svg',
                  model: Optional[str] = None,
                  model_type_str: Optional[str] = None,
                  ) -> None:
    # -- Prepare -- #

    # Load the image
    np_image = load_image(input_image_path)

    # Preprocess the image
    np_image = np_image.astype('uint8')                         # Convert into a supported data type
    image: torch.Tensor = utils.transform_fn_resize(np_image)   # Resize and normalize the image
    image = image.unsqueeze(0)                                  # Add a batch dimension

    # Load the model
    if model is None:
        model = constants.DEFAULT_MODEL
    nnmodel: nn.Module = load_model(model, str2model_type(model_type_str))


    # -- Convert -- #

    # Change to evaluation mode
    nnmodel.eval()

    # Predict the shapes
    with torch.no_grad():
        shape_pred, shape_data_pred = nnmodel(image)
    shape_pred = shapes.Shapes.from_int(shape_pred.argmax(dim=1))

    # Convert the prediction into an image
    match output_format:
        case 'svg':
            output_image = utils.specification2svg(
                    input_image_path, shape_pred, shape_data_pred.numpy()[0])
        case _:
            raise NotImplementedError


    # -- Save the Results -- #

    # Save the output
    output_image.saveas(output_path, pretty=True)



