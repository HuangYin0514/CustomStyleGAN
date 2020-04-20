from torch import nn
from utils import leaky_relu


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu(0.2)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
