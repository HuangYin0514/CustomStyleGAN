# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import torch
from model import StyleGAN2
from utils import *
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# %%
GAN = None
load_model_name = 'model_10.pt'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
GAN = StyleGAN2(lr=2e-4,
                image_size=64,
                network_capacity=16,
                transparent=False,)
GAN.to(device)
GAN.load_state_dict(torch.load(load_model_name,
                               map_location=torch.device(device)))

# %%
@torch.no_grad()
def generate_truncated(S, G, style, noi, trunc_psi=0.6, num_image_tiles=8):
    latent_dim = G.latent_dim
    av = None
    if av is None:
        z = noise(2000, latent_dim)
        samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
        av = np.mean(samples, axis=0)
        av = np.expand_dims(av, axis=0)

    w_space = []
    for tensor, num_layers in style:
        tmp = S(tensor)
        av_torch = torch.from_numpy(av).to(device)
        tmp = trunc_psi * (tmp - av_torch) + av_torch
        w_space.append((tmp, num_layers))

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks(
        batch_size, G, w_styles, noi)
    return generated_images.clamp_(0., 1.)


# %%
GAN.eval()
transparent = False
num_image_tiles = 8
batch_size = 64
ext = 'jpg' if not transparent else 'png'
num_rows = num_image_tiles


def generate_images(stylizer, generator, latents, noise):
    w = latent_to_w(stylizer, latents)
    w_styles = styles_def_to_tensor(w)
    generated_images = evaluate_in_chunks(batch_size, generator,
                                          w_styles, noise)
    generated_images.clamp_(0., 1.)
    return generated_images


# %%

latent_dim = GAN.G.latent_dim
image_size = GAN.G.image_size
num_layers = GAN.G.num_layers
# latents and noise
latents = noise_list(num_rows**2, num_layers, latent_dim)
n = image_noise(num_rows**2, image_size)

# %%

# regular
generated_images = generate_images(GAN.S, GAN.G, latents, n)

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Ori Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images,
                                         padding=2,
                                         normalize=True).detach().numpy(),
                        (1, 2, 0)))


# %%
# moving averages
generated_images = generate_truncated(GAN.SE,
                                      GAN.GE,
                                      latents,
                                      n,
                                      trunc_psi=0.6)

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("moving averages Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images,
                                         padding=2,
                                         normalize=True).detach().numpy(),
                        (1, 2, 0)))

# %%
nn = noise(num_rows, latent_dim)
# %%
# mixing regularities


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([
            init_dim * np.arange(n_tile) + i for i in range(init_dim)
        ])).to(device)
    return torch.index_select(a, dim, order_index)


tmp1 = tile(nn, 0, num_rows)
tmp2 = nn.repeat(num_rows, 1)

tt = int(num_layers / 2)
mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

n = torch.FloatTensor(num_rows**2, image_size, image_size, 1).uniform_(0., 1.).to(device)
import torch.nn.functional as F
n = F.sigmoid(n)
generated_images = generate_truncated(GAN.SE,
                                      GAN.GE,
                                      mixed_latents,
                                      n,
                                      trunc_psi=0.6)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("mixing regularities Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images,
                                         padding=2,
                                         normalize=True).detach().numpy(),
                        (1, 2, 0)))


# %%
