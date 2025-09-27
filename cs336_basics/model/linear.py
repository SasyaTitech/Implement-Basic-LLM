import torch
from torch import nn, Tensor
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class Linear(nn.Module):
    in_features: int
    out_features: int

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # make parameters in the correct device and dtype
        init_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std: float = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(init_tensor, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(init_tensor)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        Flops: 2 * in_features * out_features * ... (batch size and other dimensions)
        Output: (..., out_features)
        """
        return einsum(x, self.weight, "... input, output input->... output")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
