import numpy as np
import torch

def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.numpy().copy()           # Otherwise it isn't writable
    image = image * 255                     # From 0 to 1 -> 0 to 255
    image = image.astype(np.uint8)          # Now we can use unsigned 8-bit intergers
    image = np.transpose(image, (1, 2, 0))  # From CHW to HWC   ( Height-Width-Channel )
    return image



import os
import constants

def to_model_path(model_name: str) -> str:
    return os.path.join(constants.SAVED_MODELS_DIR, model_name + '.pth')



from torchvision import transforms

transform_fn = transforms.Compose([
    # torchvision.transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

