import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *

if __name__ == '__main__':
    bs = 12
    a = custom_image_nosie(bs, 100)
    res = latent_to_nosie(NoiseVectorizer, a)
    print(res)
