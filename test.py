import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *

if __name__ == '__main__':
    res = 0
    res1 = 0
    for i in range(10):
        res +=  (i/10)
        res1 +=i
    print(res)
    print(res1)
