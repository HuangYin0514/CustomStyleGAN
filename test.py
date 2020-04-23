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
    a = torch.Tensor([[0.998, ]])
    b = torch.Tensor([[-0.998, ]])
    print(compute_BER(a, b, sigma=1))
    print(compute_BER(a, b, sigma=2))
    print(compute_BER(a, b, sigma=3))
    print(a.shape)
