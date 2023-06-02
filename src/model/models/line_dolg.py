import math

import torch
import torch.nn as nn
from tqdm import tqdm

from .dolg import DolgNet


class LineDolgNet(nn.Module):
    def __init__(self, model_path, win_size, stride, imb_dim, device):
        super().__init__()
        self.dolg = torch.load(model_path, map_location=device)
        self.WW = win_size
        self.WS = stride
        self.imb_dim = imb_dim

    def forward(self, x):
        B, _C, H, W = x.shape
        S = math.floor((W - self.WW) / self.WS + 1)

        # NOTE: type_as properly sets device
        activations = torch.zeros((B, 512, S)).type_as(x)
        for s in tqdm(range(S)):
            start_w = self.WS * s
            end_w = start_w + self.WW
            window = x[:, :, :, start_w:end_w]  # -> (B, C, H, self.WW)
            activations[:, :, s] = self.dolg(window)
        return activations
