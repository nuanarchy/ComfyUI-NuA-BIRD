import torchvision.transforms.functional as F
from comfy.utils import ProgressBar
from .birdlib.api import BIRD

class BirdDenoising:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "image": ("IMAGE", {"forceInput": True}),

                "denoising_steps": ("INT", {"default": 10}),
                "optimization_steps": ("INT", {"default": 200}),
                "lr": ("FLOAT", {"default": 0.01, "step": 0.01}),
                "delta_t": ("INT", {"default": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()

    FUNCTION = "execute"

    CATEGORY = "NuA/BIRD"

    def execute(self, model, image, denoising_steps, optimization_steps, lr, delta_t, seed):
        bird = BIRD(model)

        task = "denoising"
        task_config = {
            "Denoising_steps": denoising_steps,
            "Optimization_steps": optimization_steps,
            "lr": lr,
            "delta_t": delta_t,
            "seed": seed,
        }

        img = image.squeeze(0)
        img = img.permute(2, 0, 1)
        img = F.to_pil_image(img)

        img = bird.process(
            task,
            task_config,
            img,
        )

        img = F.to_tensor(img)
        img = img.permute(1, 2, 0).unsqueeze(0)

        return (img)

