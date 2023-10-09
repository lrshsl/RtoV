import os
import constants

def to_model_path(model_name: str) -> str:
    return os.path.join(constants.SAVED_MODELS_DIR, model_name + '.pth')
