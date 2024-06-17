import torch
import torchvision.transforms.functional as F
from comfy.utils import ProgressBar
from .birdlib.api import BIRD

class BirdNonUniformDeblurring:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "image": ("IMAGE", {"forceInput": True}),

                "optimization_steps": ("INT", {"default": 25}),
                "lr": ("FLOAT", {"default": 0.002, "step": 0.001}),
                "delta_t": ("INT", {"default": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()

    FUNCTION = "execute"

    CATEGORY = "NuA/BIRD"

    @torch.inference_mode(False)
    def execute(self, model, image, optimization_steps, lr, delta_t, seed):
        pbar = ProgressBar(int(optimization_steps))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        bird = BIRD(model, on_progress=prog)

        task = "non_uniform_deblurring"
        task_config = {
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
        img = img.permute(1, 2, 0)
        img = img[None].unsqueeze(0)

        return (img)

