import torch
import numpy as np
import time
from retry.api import retry_call
from utils import *
import fire
from net import *
from shutil import rmtree
from random import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from net import *
from utils import *
from run.TrainStylegan2WithNoise import Trainer


data = '../../gan/custom_dataset'
results_dir = './GoodResult/results'
models_dir = './GoodResult/models'
name = 'mytest'
new = False
load_from = -1
image_size = 64
network_capacity = 16
transparent = False
batch_size = 3
gradient_accumulate_every = 5
num_train_steps = 100000
learning_rate = 2e-4
num_workers = None
save_every = 10000
generate = False
num_image_tiles = 8
trunc_psi = 0.6
transparent = False
num_image_tiles = 8
batch_size = 64
log_dir = './GoodResult/logs'
ext = 'jpg' if not transparent else 'png'
num_rows = num_image_tiles


GAN = None
load_model_name = 'model_10.pt'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
GAN = StyleGAN2(lr=2e-4,
                image_size=64,
                network_capacity=16,
                transparent=False,)
GAN.to(device)
load_temp_GAN = torch.load(
    load_model_name, map_location=torch.device(device))
for state_name in load_temp_GAN:
    GAN.state_dict()[state_name][:] = load_temp_GAN[state_name]

latent_dim = GAN.G.latent_dim
image_size = GAN.G.image_size
num_layers = GAN.G.num_layers


def init_train():
    trainer = Trainer(name,
                      results_dir,
                      models_dir,
                      log_dir,
                      batch_size=batch_size,
                      gradient_accumulate_every=gradient_accumulate_every,
                      image_size=image_size,
                      network_capacity=network_capacity,
                      transparent=transparent,
                      lr=learning_rate,
                      num_workers=num_workers,
                      save_every=save_every,
                      trunc_psi=trunc_psi)
    return trainer


def generate_images(stylizer, generator, latents, noise):
    w = latent_to_w(stylizer, latents)
    w_styles = styles_def_to_tensor(w)
    generated_images = evaluate_in_chunks(batch_size, generator,
                                          w_styles, noise)
    generated_images.clamp_(0., 1.)
    return generated_images


if __name__ == '__main__':
    trainer = init_train()
    num_rows = 4
    # w
    latents = noise_list(num_rows**2, num_layers, latent_dim)
    style_n = noise(num_rows, latent_dim)  # mix image
    # noise
    # noise_ = custom_image_nosie(num_rows**2, 100)
    # n = latent_to_nosie(GAN.N, noise_)
    # n = nn.Sigmoid()(n)
    # n = torch.zeros(num_rows**2, 64, 64, 1)
    # n = nn.Sigmoid()(n)
    n = torch.randn(num_rows**2, 64, 64, 1)*10000
    # n= n.clamp_(0.9, 1.)
    # n_std = torch.std(n)
    # n_mean = torch.mean(n)
    # n = (n-n_mean)/(n_std+1e-8)
    # moving averages
    n = nn.Sigmoid()(n)*0.5
    generated_images = trainer.generate_truncated(GAN.SE,
                                                  GAN.GE,
                                                  latents,
                                                  n,
                                                  trunc_psi=0.6)
    plt.figure()
    plt.axis("off")
    plt.title("moving averages Images")
    plt.show()
    plt.imshow(np.transpose(vutils.make_grid(generated_images,
                                             padding=2,
                                             normalize=True).detach().numpy(),
                            (1, 2, 0)))
    plt.savefig('./image2/2_value_image_generator.png')

    plt.figure(figsize=(10, 10))
    plt.hist(n.reshape(-1))
    plt.show()
    plt.savefig('./image2/n_distrbution.png')

