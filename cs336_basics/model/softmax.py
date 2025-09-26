import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np
from cs336_basics.model.silu import SiLU
from cs336_basics.model.linear import Linear



class Sofmtax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_x: torch.Tensor = torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x - max_x)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    def extra_repr(self) -> str:
        return f"softmax"
