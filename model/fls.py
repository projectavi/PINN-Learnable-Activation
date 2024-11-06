# baseline implementation of First Layer Sine
# paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2109.09338

import torch
import torch.nn as nn
from model.act import *

        

class FLS(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation):
        super(FLS, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(eval(f"{activation}()"))

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        return self.linear(src)