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

from net import BackOptNet
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
                 epoch_number=100,
                 *args,
                 **kwargs):
        self.Net_params = [args, kwargs]
        self.NET = None

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
        self.loss = 0
        self.init_folders()

        self.MSELoss = nn.MSELoss()

        if self.NET is None:
            self.init_NET()

        batch_size = self.batch_size
        latent_dim = self.NET.G.latent_dim
        num_layers = self.NET.G.num_layers

        get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        self.style = get_latents_fn(batch_size, num_layers, latent_dim)
        self.noise = custom_image_nosie(batch_size, 100)

    def init_NET(self):
        self.NET = BackOptNet(self.lr)
        self.NET.to(device)

    def train(self):
        assert self.init_NET is not None, 'You must first initialize the NET'

        # w
        w_space = latent_to_w(self.NET.S, self.style)
        w_styles = styles_def_to_tensor(w_space)
        # noise
        noise_styles = latent_to_nosie(self.NET.N, self.noise)
        secret = self.noise

        # train
        self.NET.N.zero_grad()
        generated_images = self.NET.G(w_styles, noise_styles)
        decode = self.NET.E(generated_images)
        divergence = self.batch_size * self.MSELoss(decode, secret)
        E_loss = divergence
        E_loss.register_hook(raise_if_nan)
        E_loss.backward()
        # compute BER
        self.BER_1 = compute_BER(decode.detach(), secret, sigma=1)
        self.BER_2 = compute_BER(decode.detach(), secret, sigma=2)
        self.BER_3 = compute_BER(decode.detach(), secret, sigma=3)
        # record total loss
        self.E_loss = float(divergence.detach().item())
        self.NET.N_opt.step()

        self.tb_writer.add_scalar('Train/loss', self.E_loss, self.steps)
        self.tb_writer.add_scalars('Train/BERs',  {'BER1': self.BER_1,
                                                   'BER2': self.BER_2,
                                                   'BER3': self.BER_3
                                                   }, self.steps)
        self.tb_writer.flush()

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_E{num}.pt')

    def save(self, num):
        torch.save(self.NET.state_dict(), self.model_name(num))

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
        rmtree(f'./logs/{self.name}', True)
        (self.log_dir / self.name).mkdir(parents=True, exist_ok=True)

    def print_log(self):
        print(
            f'E: {self.E_loss:.2f} | BER_1: {self.BER_1:.4f} | BER_2: {self.BER_2:.4f} | BER_3: {self.BER_3:.4f}'
        )

    def load_part_state_dict(self, style_num=-1, extract_num=-1):
        if self.NET is None:
            self.init_NET()

        name = style_num
        load_model_name = f'model_{name}.pt'
        load_model = torch.load(
            load_model_name, map_location=torch.device(device))
        # load style gan
        dont_load_list = []
        for state_name in load_model:
            try:
                self.NET.state_dict(
                )[state_name][:] = load_model[state_name]
            except KeyError as identifier:
                dont_load_list.append(identifier)
        print(f'load from {load_model_name}')
        print(f'dont load {dont_load_list[0:2]} ...')

        # load extract
        load_model_name = f'modelE_{extract_num}.pt'
        load_model = torch.load(
            load_model_name, map_location=torch.device(device))
        for state_name in load_model:
            if load_model[state_name].shape == torch.Size([]):
                self.NET.state_dict(
                )[state_name] = load_model[state_name]
                continue
            self.NET.state_dict(
            )[state_name][:] = load_model[state_name]
        print(f'load from {load_model_name}')
