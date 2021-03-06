import torch

from utils import *
from net import *


class ExtractNet(nn.Module):

    def __init__(self, lr=2e-4, steps=1, beta1=0.5, beta2=0.999):
        super().__init__()
        self.lr = lr
        self.steps = steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.E = ExtractModule()

        self.E_opt = torch.optim.Adam(self.E.parameters(),
                                      lr=self.lr,
                                      betas=(self.beta1, self.beta2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return x
