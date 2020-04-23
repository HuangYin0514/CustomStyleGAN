import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *

if __name__ == '__main__':
    # list1 = [12, 3, 4, 5]
    # list2 = [12, 3, 4, 5]
    # list3 = [12, 3, 4, 5]
    # list_t = [list1, list2, list3]
    # res = list(list(map(lambda x: x/10, i)) for i in list_t)
    # print(res)
    # a = torch.Tensor([[0.998, ]])
    # b = torch.Tensor([[-0.998, ]])
    # print(compute_BER(a, b, sigma=1))
    # print(compute_BER(a, b, sigma=2))
    # print(compute_BER(a, b, sigma=3))
    # print(a.shape)
    s = torch.FloatTensor(4, 100).uniform_(-1., 1.).to(device)
    extractNet = ExtractNet()
    input = torch.randn(4, 3, 64, 64)
    ouput = extractNet.E(input)
    print(ouput)
    print(compute_BER(ouput, s, sigma=1))
    print(compute_BER(ouput, s, sigma=2))
    print(compute_BER(ouput, s, sigma=3))
