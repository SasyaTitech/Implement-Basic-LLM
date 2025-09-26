import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np


class Linear(nn.Module):
    in_features: int
    out_features: int

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        nn.Linear

        # make parameters in the correct device and dtype
        init_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std: float = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(init_tensor, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(init_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... input, output input->... output")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
