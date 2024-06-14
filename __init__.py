from .bird_loader import *
from .bird_deblurring import *
from .bird_non_uniform_deblurring import *
from .bird_denoising import *
from .bird_inpainting import *
from .bird_super_resolution import *

NODE_CLASS_MAPPINGS = {
    "Bird_Loader_NuA": BirdLoader,
    "Bird_Deblurring_NuA": BirdDeblurring,
    "Bird_Non_Uniform_Deblurring_NuA": BirdNonUniformDeblurring,
    "Bird_Denoising_NuA": BirdDenoising,
    "Bird_Inpainting_NuA": BirdInpainting,
    "Bird_Super_Resolution_NuA": BirdSuperResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Bird_Loader_NuA": "Bird Loader",
    "Bird_Deblurring_NuA": "Bird Deblurring",
    "Bird_Non_Uniform_Deblurring_NuA": "Bird Non Uniform Deblurring",
    "Bird_Denoising_NuA": "Bird Denoising",
    "Bird_Inpainting_NuA": "Bird Inpainting",
    "Bird_Super_Resolution_NuA": "Bird Super Resolution",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
