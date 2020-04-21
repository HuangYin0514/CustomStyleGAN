# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *

# %%
bs = 12
a = custom_image_nosie(bs, 100)
noiseVectorizer = NoiseVectorizer(100)
res = latent_to_nosie(noiseVectorizer, a)
print(res)


# %%
custom_GAN = StyleGAN2(64)
load_temp = torch.load(
    'model_12.pt', map_location=torch.device(device))

# %%
for i in load_temp:
    print(i)


# %%
for state_name in load_temp:
    custom_GAN.state_dict()[state_name][:] = load_temp[state_name]
    print(f'laod {state_name}')

# %%
load_temp['S.net.0.weight'][0:3],custom_GAN.state_dict()['S.net.0.weight'][0:3]

# %%
