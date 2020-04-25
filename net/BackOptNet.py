import torch

from utils import *
from net import *


class BackOptNet(nn.Module):

    def __init__(self,
                 lr=2e-4,
                 steps=1,
                 beta1=0.5,
                 beta2=0.999,
                 latent_dim=512,
                 noise_dim=100,
                 image_size=64,
                 network_capacity=16,
                 transparent=False,
                 style_depth=8):
        super().__init__()
        self.lr = lr
        self.steps = steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.NE = NoiseVectorizer(noise_dim)
        self.GE = Generator(image_size, latent_dim,
                            network_capacity, transparent=transparent)
        self.D = Discriminator(image_size, network_capacity,
                               transparent=transparent)
        self.E = ExtractModule()

        generator_params = list(self.NE.parameters())
        print(list(self.NE.state_dict())[-2::2])
        self.opt = torch.optim.Adam(generator_params,
                                    lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                    'min',
                                                                    factor=0.99,
                                                                    patience=10,
                                                                    verbose=True,
                                                                    threshold=0.0001)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return x
