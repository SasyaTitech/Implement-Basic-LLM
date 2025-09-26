import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np


class SiLU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def extra_repr(self) -> str:
        return "SiLU Activation Function"
