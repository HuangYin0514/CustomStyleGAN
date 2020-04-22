
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
from net import StyleGAN2
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
                 *args,
                 **kwargs):
        self.ExtractNet_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.log_dir = Path(log_dir)
        self.tb_writer = SummaryWriter(self.log_dir / name)

        self.lr = lr
        self.batch_size = batch_size

        self.save_every = save_every
        self.steps = 0

        self.loss = 0

        self.init_folders()

    # TODO
    def train(self):
        pass

    # TODO
    @torch.no_grad()
    def evaluate(self):
        pass

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
        rmtree(f'./logs/{self.name}', True)
        (self.log_dir / self.name).mkdir(parents=True, exist_ok=True)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(f'./logs/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    # TODO save model
    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))
        self.write_config()

    # TODO laod model
    def load_part_state_dict(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [
                p for p in Path(self.models_dir / self.name).glob('model_*.pt')
            ]
            saved_nums = sorted(
                map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        load_model_name = f'model_{name}.pt'

        load_temp_GAN = torch.load(
            load_model_name, map_location=torch.device(device))

        for state_name in load_temp_GAN:
            self.GAN.state_dict()[state_name][:] = load_temp_GAN[state_name]

        print(load_model_name)
