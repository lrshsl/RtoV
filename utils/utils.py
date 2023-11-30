import numpy as np
import torch
from typing import Callable

# Function 'tensor2image' {{{
def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.numpy().copy()           # Otherwise it isn't writable
    image = image * 255                     # From 0 to 1 -> 0 to 255
    image = image.astype(np.uint8)          # Now we can use unsigned 8-bit intergers
    image = np.transpose(image, (1, 2, 0))  # From CHW to HWC   ( Height-Width-Channel )
    return image
# }}}

# Function 'specification2svg' {{{
import svgwrite
from utils.shapes import Shapes

def specification2svg(name: str, shape: Shapes, specification: np.ndarray) -> svgwrite.Drawing:
    spec: list = list(map(int, specification))
    img = svgwrite.Drawing(name, profile='tiny')
    
    # Set the Background Color
    background = img.rect(insert=(0, 0),
                          size=('100%', '100%'),
                          fill='white')
    img.add(background)

    match shape:
        case Shapes.Line:
            img.add(img.line((spec[0], spec[1]),
                             (spec[2], spec[3])))
        case Shapes.Rectangle:
            img.add(img.rect((spec[0], spec[1]),
                             (spec[2], spec[3])))
        case Shapes.Circle:
            img.add(img.circle((spec[0], spec[1]),
                               spec[2]))
        case Shapes.Triangle:
            img.add(img.polygon([(spec[0], spec[1]),
                                 (spec[2], spec[3]),
                                 (spec[4], spec[5])]))
    return img
# }}}

# Function 'to_model_path' {{{
import os
import constants

def to_model_path(model_name: str) -> str:
    return os.path.join(constants.SAVED_MODELS_DIR, model_name + '.pth')
# }}}

# Image Transforms {{{
from torchvision import transforms

transform_fn: Callable[[np.ndarray], torch.Tensor] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_fn_resize: Callable[[np.ndarray], torch.Tensor] = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# }}}

# Enum PlotType {{{
import enum

class PlotType(enum.IntEnum):
    Pairs = 0
    Overlay = 1
# }}}
