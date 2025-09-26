import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np
from cs336_basics.model.activation import SiLU
from cs336_basics.model.linear import Linear


class SwiGLU(nn.Module):
    dim: int
    dim_ff: int
    silu = SiLU()

    def __init__(self, dim: int, dim_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.dim = dim
        if dim_ff > 0:
            self.dim_ff = dim_ff
        else:
            # make dim_ff 8/3 of dim rounded to a multiple of 64
            self.dim_ff = int(np.ceil(8 * self.dim / 3 / 64)) * 64
        self.fc1 = Linear(self.dim, self.dim_ff, device=device, dtype=dtype)
        self.fc3 = Linear(self.dim, self.dim_ff, device=device, dtype=dtype)
        self.fc2 = Linear(self.dim_ff, self.dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.silu(self.fc1(x))
        b = self.fc3(x)
        c = a * b
        out = self.fc2(c)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, dim_ff={self.dim_ff}"
