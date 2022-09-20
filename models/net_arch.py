import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Bicubic(nn.Module):
    def __init__(self, scale_factor=1/2):
        super(Bicubic, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
