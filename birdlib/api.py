import os
import sys
#package_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, package_dir)
import yaml
import numpy as np
import tqdm
import torch
from torch import nn
from PIL import Image, ImageOps
from guided_diffusion.models import Model
import random
from ddim_inversion_utils import *
from utils import *

class BIRD():
    def __init__(self, model, on_progress=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model

        package_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(package_dir, "data/celeba_hq.yml"), "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)

        ### Define the DDIM scheduler
        self.ddim_scheduler = DDIMScheduler(beta_start=self.config.diffusion.beta_start, beta_end=self.config.diffusion.beta_end,
                                       beta_schedule=self.config.diffusion.beta_schedule)

        self.progress_hook = on_progress if on_progress else None

    def process(self, task, task_config, img, mask=None):
        ### Reproducibility
        torch.set_printoptions(sci_mode=False)
        ensure_reproducibility(task_config['seed'])

        self.ddim_scheduler.set_timesteps(
            self.config.diffusion.num_diffusion_timesteps // task_config['delta_t'])  # task_config['Denoising_steps']

        if (task == 'deblurring'):
            # scale=41
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            net_kernel = fcn(200, task_config['kernel_size'] * task_config['kernel_size']).cuda()
            net_input_kernel = get_noise(200, 'noise', (1, 1)).cuda()
            net_input_kernel.squeeze_()

            img = img.resize((self.config.data.image_size, self.config.data.image_size))
            img = np.array(img).astype(np.float32) / 255 * 2 - 1
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

            radii = torch.ones([1, 1, 1]).cuda() * (np.sqrt(self.config.data.image_size * self.config.data.image_size * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(
                torch.randn(1, self.config.model.in_channels, self.config.data.image_size, self.config.data.image_size).to(self.device))
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr_img']},
                                          {'params': net_kernel.parameters(), 'lr': task_config['lr_blur']}])

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                out_k = net_kernel(net_input_kernel)

                out_k_m = out_k.view(-1, 1, task_config['kernel_size'], task_config['kernel_size'])

                blurred_xt = nn.functional.conv2d(x_0_hat.view(-1, 1, self.config.data.image_size, self.config.data.image_size),
                                                  out_k_m, padding="same", bias=None).view(1, 3, self.config.data.image_size,
                                                                                           self.config.data.image_size)
                loss = l2_loss(blurred_xt, img)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                #if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                if self.progress_hook: self.progress_hook(iteration)
            img = Image.fromarray(process(x_0_hat, 0))
        elif (task == 'non_uniform_deblurring'):
            img = img.resize((self.config.data.image_size, self.config.data.image_size))
            img_np = (np.array(img) / 255.) * 2. - 1.
            img_torch = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
            radii = torch.ones([1, 1, 1]).cuda() * (
                np.sqrt(self.config.data.image_size * self.config.data.image_size * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(
                torch.randn(1, self.config.model.in_channels, self.config.data.image_size, self.config.data.image_size).to(self.device))
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr']}])  #

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                loss = l2_loss(x_0_hat, img_torch.cuda())
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                #if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                if self.progress_hook: self.progress_hook(iteration)
            img = Image.fromarray(process(x_0_hat, 0))
        elif (task == 'denoising'):
            img = img.resize((self.config.data.image_size, self.config.data.image_size))
            img = np.array(img).astype(np.float32) / 255 * 2 - 1
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()
            
            radii = torch.ones([1, 1, 1]).cuda() * (
                np.sqrt(self.config.data.image_size * self.config.data.image_size * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(
                torch.randn(1, self.config.model.in_channels, self.config.data.image_size, self.config.data.image_size).to(self.device))
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr']}])  #

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                loss = l2_loss(x_0_hat, img)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                #if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                if self.progress_hook: self.progress_hook(iteration)
            img = Image.fromarray(process(x_0_hat, 0))
        elif (task == 'inpainting'):
            img = img.resize((self.config.data.image_size, self.config.data.image_size))
            img = np.array(img).astype(np.float32) / 255 * 2 - 1
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

            mask = mask.resize((self.config.data.image_size, self.config.data.image_size))
            mask = ImageOps.invert(mask)
            mask = np.array(mask).astype(np.float32) / 255

            t_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
            radii = torch.ones([1, 1, 1]).cuda() * (
                np.sqrt(self.config.data.image_size * self.config.data.image_size * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(
                torch.randn(1, self.config.model.in_channels, self.config.data.image_size, self.config.data.image_size).to(self.device))
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr']}])  #

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                loss = l2_loss(x_0_hat * t_mask, img * t_mask)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                #if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                if self.progress_hook: self.progress_hook(iteration)
            img = Image.fromarray(process(x_0_hat, 0))
        elif (task == 'super_resolution'):
            image_size_x = img.size[0]
            image_size_y = img.size[1]
            image_size = min(image_size_x, image_size_y)
            upsampling_ratio = self.config.data.image_size / image_size

            img = img.resize((image_size, image_size))
            img = np.array(img).astype(np.float32) / 255 * 2 - 1
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

            downsampling_op = torch.nn.AdaptiveAvgPool2d((image_size, image_size)).cuda()
            for param in downsampling_op.parameters():
                param.requires_grad = False

            radii = torch.ones([1, 1, 1]).cuda() * (np.sqrt(self.config.data.image_size
                                                            * self.config.data.image_size
                                                            * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(torch.randn(1,
                                                              self.config.model.in_channels,
                                                              self.config.data.image_size,
                                                              self.config.data.image_size).to(self.device))
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr']}])

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                loss = l2_loss(downsampling_op(x_0_hat), img)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius 1
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                #if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                if self.progress_hook: self.progress_hook(iteration)
            img = Image.fromarray(process(x_0_hat, 0))
        else:
            pass

        return img