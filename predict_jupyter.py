# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from IPython import get_ipython

from net import *
from utils import *
from run import Trainer

# %%
GAN = None
load_model_name = 'model_12.pt'
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
ext = 'jpg' if not transparent else 'png'
num_rows = num_image_tiles
latent_dim = GAN.G.latent_dim
image_size = GAN.G.image_size
num_layers = GAN.G.num_layers

trainer = Trainer(name,
                  results_dir,
                  models_dir,
                  batch_size=batch_size,
                  gradient_accumulate_every=gradient_accumulate_every,
                  image_size=image_size,
                  network_capacity=network_capacity,
                  transparent=transparent,
                  lr=learning_rate,
                  num_workers=num_workers,
                  save_every=save_every,
                  trunc_psi=trunc_psi)


def generate_images(stylizer, generator, latents, noise):
    w = latent_to_w(stylizer, latents)
    w_styles = styles_def_to_tensor(w)
    generated_images = evaluate_in_chunks(batch_size, generator,
                                          w_styles, noise)
    generated_images.clamp_(0., 1.)
    return generated_images


# %%
GAN.eval()
# latents and noise
#####################
# latents = noise_list(num_rows**2, num_layers, latent_dim)
# n = image_noise(num_rows**2, image_size)
# style_n = noise(num_rows, latent_dim)
#####################

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
generated_images = trainer.generate_truncated(GAN.SE,
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


tmp1 = tile(style_n, 0, num_rows)
tmp2 = style_n.repeat(num_rows, 1)
tt = int(num_layers / 2)
mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]
generated_images = trainer.generate_truncated(GAN.SE,
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
input_ = torch.randn(10, 100)
sigmoid_output = GAN.N(input_)
sigmoid_output_np = sigmoid_output.detach().numpy()
sigmoid_output_shape = sigmoid_output_np.reshape(-1, 64*64)

# %%
hist, bin_edges = np.histogram(sigmoid_output_shape[0])

# %%
n1, bins, patches = plt.hist(x=sigmoid_output_shape[0], bins='auto', color='#0504aa',
                             alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# %%
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


ema = EMA(0.995)
list1 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
list1

# %%
res = []
for _ in list1:
    res.append(ema.update_average(2, _))
res

# %%
load_model_name1 = 'model_12.pt'
GAN1 = StyleGAN2(lr=2e-4,
                 image_size=64,
                 network_capacity=16,
                 transparent=False,)

load_temp_GAN1 = torch.load(
    load_model_name1, map_location=torch.device(device))

for state_name in load_temp_GAN1:
    GAN1.state_dict()[state_name][:] = load_temp_GAN1[state_name]

load_model_name2 = 'model_10.pt'
GAN2 = StyleGAN2(lr=2e-4,
                 image_size=64,
                 network_capacity=16,
                 transparent=False,)

load_temp_GAN2 = torch.load(
    load_model_name2, map_location=torch.device(device))

for state_name in load_temp_GAN2:
    GAN2.state_dict()[state_name][:] = load_temp_GAN2[state_name]


# %%
mean1 = torch.mean(GAN1.state_dict()[
                   'S.net.2.weight']-GAN2.state_dict()['S.net.2.weight'])
mean2 = torch.mean(GAN1.state_dict()[
                   'G.blocks.0.to_style1.weight']-GAN2.state_dict()['G.blocks.0.to_style1.weight'])
mean3 = torch.mean(GAN1.state_dict()[
                   'D.blocks.0.net.0.weight']-GAN2.state_dict()['D.blocks.0.net.0.weight'])
mean4 = torch.mean(GAN1.state_dict()[
                   'SE.net.0.weight']-GAN2.state_dict()['SE.net.0.weight'])
mean5 = torch.mean(GAN1.state_dict()[
                   'GE.blocks.0.to_style2.weight']-GAN2.state_dict()['GE.blocks.0.to_style2.weight'])
print([mean1, mean2, mean3, mean4, mean5])

# %%
