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
