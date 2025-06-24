import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_in = x
        x = x + 0.044715 * torch.pow(x, 3)
        x = torch.sqrt(torch.tensor(2.0/torch.pi)) * x
        x = 1 + torch.tanh(x)
        x = 0.5 * x_in * x
        return x