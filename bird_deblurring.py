import torch
import torchvision.transforms.functional as F
from comfy.utils import ProgressBar
from .birdlib.api import BIRD

class BirdDeblurring:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "image": ("IMAGE", {"forceInput": True}),

                "denoising_steps": ("INT", {"default": 10}),
                "optimization_steps": ("INT", {"default": 100}),
                "lr_blur": ("FLOAT", {"default": 0.0002, "step": 0.0001}),
                "lr_img": ("FLOAT", {"default": 0.003, "step": 0.001}),
                "delta_t": ("INT", {"default": 100}),
                "kernel_size": ("INT", {"default": 41}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()

    FUNCTION = "execute"

    CATEGORY = "NuA/BIRD"

    @torch.inference_mode(False)
    def execute(self, model, image, denoising_steps, optimization_steps, lr_blur, lr_img, delta_t, kernel_size, seed):
        pbar = ProgressBar(int(optimization_steps))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        bird = BIRD(model, on_progress=prog)
        
        task = "deblurring"
        task_config = {
            "Denoising_steps": denoising_steps,
            "Optimization_steps": optimization_steps,
            "lr_blur": lr_blur,
            "lr_img": lr_img,
            "delta_t": delta_t,
            "kernel_size": kernel_size,
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

