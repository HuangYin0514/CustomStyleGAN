
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
        self.loss = 0
        self.init_folders()

    def train(self):
        pass

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_E{num}.pt')

    def save(self, num):
        torch.save(self.ExtractNet.state_dict(), self.model_name(num))

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
        rmtree(f'./logs/{self.name}', True)
        (self.log_dir / self.name).mkdir(parents=True, exist_ok=True)

    def load_part_state_dict(self, style_num=-1, extract_num=-1):
        print('t')
