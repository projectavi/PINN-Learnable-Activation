import torch
import torch.nn as nn

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)

class SinAct(nn.Module):
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class PeriodicWaveAct(nn.Module):
    def __init__(self):
        super(PeriodicWaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(self.b1 * x)+ self.w2 * torch.cos(self.b2 * x)

class LearnableLeakyRelu(nn.Module):
    def __init__(self):
        super(LearnableLeakyRelu, self).__init__()
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        if x < 0:
            return self.a * x
        else:
            return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return x * nn.Sigmoid(self.b * x)

class LAU(nn.Module):
    def __init__(self):
        super(LAU, self).__init__()
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return x * torch.log(1 + self.a * nn.Sigmoid(self.b * x))