# helper classes
import torch
from torch import nn
from torch.autograd import grad as torch_grad
import numpy as np


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


def compare_num_tenToTwo(a, b, num_len):
    numberof2_list_a = str(bin(a)).split('0b')
    numberof2_list_b = str(bin(b)).split('0b')
    numberof2_list_a = list(numberof2_list_a[1])
    numberof2_list_b = list(numberof2_list_b[1])
    while len(numberof2_list_a) < num_len:
        numberof2_list_a = ['0'] + numberof2_list_a
    while len(numberof2_list_b) < num_len:
        numberof2_list_b = ['0'] + numberof2_list_b
    result = [
        int(numberof2_list_a[i]) - int(numberof2_list_b[i])
        for i in range(len(numberof2_list_a))
    ]
    return np.sum(np.abs(result))


def compute_BER(input_noise, output_noise, sigma):
    number_row = input_noise.size(0)
    number_line = input_noise.size(1)
    input_noise_to_msg = torch.floor((input_noise + 1) * 2**(sigma - 1))
    output_noise_to_msg = torch.floor((output_noise + 1) * 2**(sigma - 1))
    if len(input_noise_to_msg.shape) > 2:
        input_noise_to_msg = input_noise_to_msg.squeeze(3).squeeze(2)
    if len(output_noise_to_msg.shape) > 2:
        output_noise_to_msg = output_noise_to_msg.squeeze(3).squeeze(2)
    input_noise_to_msg_numpy = np.array(input_noise_to_msg.detach().cpu(),
                                        dtype=np.integer)
    output_noise_to_msg_numpy = np.array(output_noise_to_msg.detach().cpu(),
                                         dtype=np.integer)
    error_counter = 0
    for row in range(number_row):
        result = [
            compare_num_tenToTwo(input_noise_to_msg_numpy[row][i],
                                 output_noise_to_msg_numpy[row][i], sigma)
            for i in range(number_line)
        ]
        error_counter += np.sum(result)
    ber = error_counter / (number_row * number_line * sigma)
    return ber
