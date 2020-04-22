import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *

if __name__ == '__main__':
    inp = torch.rand(64, 3, 64, 64)
    print(inp.shape)
    netE = ExtractNet()
    out = netE(inp)
    print(out.shape)
