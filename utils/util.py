# helper classes

import torch
from torch import nn
from torch.autograd import grad as torch_grad


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# helpers


def default(value, d):
    return d if value is None else value


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size()).to(device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).to(device)


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)


def custom_image_nosie(n, latent_dim):
    return torch.FloatTensor(n, latent_dim).uniform_(-1., 1.).to(device)


def latent_to_nosie(noise_vectorizer, latent_descr):
    return noise_vectorizer(latent_descr)


def leaky_relu(p):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(
        zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
