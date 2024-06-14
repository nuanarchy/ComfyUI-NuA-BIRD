import os
import sys
#package_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, package_dir)
import yaml
import numpy as np
import tqdm
import torch
from torch import nn
import sys
from guided_diffusion.models import Model
import random
from ddim_inversion_utils import *
from utils import *

class BIRD():
    def __init__(self, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model

        package_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(package_dir, "data/celeba_hq.yml"), "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)

        ### Define the DDIM scheduler
        self.ddim_scheduler = DDIMScheduler(beta_start=self.config.diffusion.beta_start, beta_end=self.config.diffusion.beta_end,
                                       beta_schedule=self.config.diffusion.beta_schedule)


    def process(self, task, task_config, img):
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

            img_pil, downsampled_torch = generate_blurry_image(img)
            radii = torch.ones([1, 1, 1]).cuda() * (np.sqrt(256 * 256 * 3))

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
                loss = l2_loss(blurred_xt, downsampled_torch)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                    img = Image.fromarray(np.concatenate(
                        [process(downsampled_torch, 0), process(x_0_hat, 0), np.array(img_pil).astype(np.uint8)],
                        1))
        elif (task == 'non_uniform_deblurring'):
            img = img.resize((self.config.data.image_size, self.config.data.image_size))
            img_np = (np.array(img) / 255.) * 2. - 1.
            img_pil = img
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

                if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                    img = Image.fromarray(np.concatenate(
                        [process(img_torch.cuda(), 0), process(x_0_hat, 0), np.array(img_pil).astype(np.uint8)],
                        1))
        elif (task == 'denoising'):
            img_pil, img_np = generate_noisy_image(img)
            img_torch = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
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

                if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                    img = Image.fromarray(np.concatenate(
                        [process(img_torch.cuda(), 0), process(x_0_hat, 0), np.array(img_pil).astype(np.uint8)],
                        1))
        elif (task == 'inpainting'):
            img_pil, img_np, mask = generate_noisy_image_and_mask(img)
            img_torch = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
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
                loss = l2_loss(x_0_hat * t_mask, img_torch.cuda() * t_mask)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius sqrt(D)
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                    img = Image.fromarray(np.concatenate([process(img_torch.cuda() * t_mask, 0), process(x_0_hat, 0),
                                                    np.array(img_pil).astype(np.uint8)], 1))
        elif (task == 'super_resolution'):
            img_pil, downsampled_torch, downsampling_op = generate_lr_image(img, task_config['downsampling_ratio'])
            radii = torch.ones([1, 1, 1]).cuda() * (
                np.sqrt(self.config.data.image_size * self.config.data.image_size * self.config.model.in_channels))

            latent = torch.nn.parameter.Parameter(
                torch.randn(1, self.config.model.in_channels, self.config.data.image_size, self.config.data.image_size).to(self.device))
            l2_loss = nn.MSELoss()  # nn.L1Loss()
            optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr']}])  #

            for iteration in range(task_config['Optimization_steps']):
                optimizer.zero_grad()
                x_0_hat = DDIM_efficient_feed_forward(latent, self.model, self.ddim_scheduler)
                loss = l2_loss(downsampling_op(x_0_hat), downsampled_torch)
                loss.backward()
                optimizer.step()

                # Project to the Sphere of radius 1
                for param in latent:
                    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                    param.data.mul_(radii)

                if iteration % 10 == 0:
                    # psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
                    # print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
                    img = Image.fromarray(np.concatenate(
                        [process(MeanUpsample(downsampled_torch, task_config['downsampling_ratio']), 0),
                         process(x_0_hat, 0), np.array(img_pil).astype(np.uint8)], 1))
        else:
            pass

        return img