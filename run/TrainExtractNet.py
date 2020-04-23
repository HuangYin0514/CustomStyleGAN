
import json
import multiprocessing
from math import floor, log2
from pathlib import Path
from random import random
from shutil import rmtree

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils import data

from datasets.Datasets import Dataset
from net import StyleGAN2, ExtractNet
from utils import *

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# speed up
cudnn.benchmark = True
num_cores = multiprocessing.cpu_count()


class Trainer():

    def __init__(self,
                 name,
                 results_dir,
                 models_dir,
                 log_dir,
                 batch_size=4,
                 lr=2e-4,
                 save_every=1000,
                 mixed_prob=0.9,
                 epoch_number=1,
                 *args,
                 **kwargs):
        self.Net_params = [args, kwargs]
        self.StyleGAN = None
        self.ExtractNet = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.log_dir = Path(log_dir)
        self.tb_writer = SummaryWriter(self.log_dir / name)

        self.lr = lr
        self.batch_size = batch_size
        self.mixed_prob = mixed_prob
        self.epoch_number = epoch_number

        self.save_every = save_every
        self.steps = 0
        self.E_loss = 0
        self.BER_1, self.BER_2, self.BER_3 = 0., 0., 0.
        self.MSELoss = nn.MSELoss()
        self.init_folders()
        self.fix = torch.rand(self.batch_size,100)

    def init_StyleGAN(self, num):
        self.StyleGAN = StyleGAN2(lr=self.lr, image_size=64, )
        self.StyleGAN.to(device)

        name = num
        load_model_name = f'model_{name}.pt'
        load_temp_GAN = torch.load(
            load_model_name, map_location=torch.device(device))
        for state_name in load_temp_GAN:
            self.StyleGAN.state_dict(
            )[state_name][:] = load_temp_GAN[state_name]
        print(f'load stylegan from {load_model_name}')

    def init_ExtractNet(self):
        self.ExtractNet = ExtractNet(lr=self.lr)
        self.ExtractNet.to(device)

    def sample_StyleGAN_input_data(self):
        batch_size = self.batch_size
        latent_dim = self.StyleGAN.G.latent_dim
        num_layers = self.StyleGAN.G.num_layers
        # w
        get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        style = get_latents_fn(batch_size, num_layers, latent_dim)
        w_space = latent_to_w(self.StyleGAN.S, style)
        w_styles = styles_def_to_tensor(w_space)
        # noise
        noise = custom_image_nosie(batch_size, 100)
        noise_styles = latent_to_nosie(self.StyleGAN.N, noise)
        secret = noise
        return w_styles, noise_styles, secret

    def train(self):

        if self.ExtractNet is None:
            self.init_ExtractNet()
        assert self.StyleGAN is not None, 'You must first initialize the Style GAN'

        # train
        self.ExtractNet.E.zero_grad()
        w_styles, noise_styles, secret = self.sample_StyleGAN_input_data()
        generated_images = self.StyleGAN.G(w_styles, noise_styles)
        decode = self.ExtractNet.E(generated_images.clone().detach())
        divergence = self.batch_size *self.MSELoss(decode,secret)
        E_loss = divergence
        E_loss.register_hook(raise_if_nan)
        E_loss.backward()
        # compute BER
        self.BER_1 = compute_BER(decode.detach(), secret, sigma=1)
        self.BER_2 = compute_BER(decode.detach(), secret, sigma=2)
        self.BER_3 = compute_BER(decode.detach(), secret, sigma=3)
        # print(decode)
        # print('**'*100)
        # print(secret)
        # record total loss
        self.E_loss = float(divergence.detach().item())
        self.ExtractNet.E_opt.step()

        self.tb_writer.add_scalar('Train/loss', self.E_loss, self.steps)
        self.tb_writer.add_scalars('Train/BERs',  {'BER1': self.BER_1,
                                                   'BER2': self.BER_2,
                                                   'BER3': self.BER_3
                                                   }, self.steps)
        self.tb_writer.flush()
        # save from NaN errors
        checkpoint_num = floor(self.steps / self.save_every)
        # periodically save results
        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        self.steps += 1

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_E{num}.pt')

    def save(self, num):
        torch.save(self.ExtractNet.state_dict(), self.model_name(num))

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
        rmtree(f'./logs/{self.name}', True)
        (self.log_dir / self.name).mkdir(parents=True, exist_ok=True)

    def print_log(self):
        print(
            f'E: {self.E_loss:.2f} | BER_1: {self.BER_1:.4f} | BER_2: {self.BER_2:.4f} | BER_3: {self.BER_3:.4f}'
        )

    def load_part_state_dict(self, num=-1):

        if self.ExtractNet is None:
            self.init_ExtractNet()

        name = num
        if num == -1:
            file_paths = [
                p for p in Path(self.models_dir / self.name).glob('modelE_*.pt')
            ]
            saved_nums = sorted(
                map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        load_model_name = f'modelE_{name}.pt'
        ExtractNet = torch.load(
            load_model_name, map_location=torch.device(device))

        for state_name in ExtractNet:
            self.ExtractNet.state_dict(
            )[state_name][:] = ExtractNet[state_name]

        print(f'load stylegan from {load_model_name}')
