import os
import sys
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)
sys.path.insert(0, os.path.join(package_dir, 'birdlib'))
import yaml

import torchvision.transforms.functional as F

import folder_paths

from .birdlib.utils import dict2namespace, load_pretrained_diffusion_model

class BirdLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        model_name = next((model for model in checkpoints if model == 'celeba_hq.ckpt'), None)
        return {
            "required": {
                "model_name": ("STRING", {"default": model_name}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ()

    FUNCTION = "load_model"

    CATEGORY = "NuA/BIRD"

    def load_model(self, model_name):
        model_dir = os.path.join(folder_paths.models_dir, "checkpoints", model_name)

        with open(os.path.join(package_dir, "birdlib/data/celeba_hq.yml"), "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)
        model, device = load_pretrained_diffusion_model(config, model_dir)

        return (model, )