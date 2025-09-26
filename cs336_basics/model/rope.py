import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker


class RoPE(nn.Module):
    theta: float
    d_k: int
    max_seq_len: int

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        d_k_2 = d_k // 2
        assert d_k % 2 == 0, f"d_k {d_k} must be even"

        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        seq_pos = torch.arange(0, max_seq_len, device=device)
        assert seq_pos.shape == (
            max_seq_len,
        ), f"pos shape {seq_pos.shape} does not match expected shape {(max_seq_len,)}"

        inv_freq = einsum(seq_pos, inv_freq, "pos , d_k_2 -> pos d_k_2")

        cos = torch.cos(inv_freq)
        assert cos.shape == (
            max_seq_len,
            d_k_2,
        ), f"cos shape {cos.shape} does not match expected shape {(max_seq_len, d_k_2)}"

        sin = torch.sin(inv_freq)
        assert sin.shape == (
            max_seq_len,
            d_k_2,
        ), f"sin shape {sin.shape} does not match expected shape {(max_seq_len, d_k_2)}"

        self.register_buffer("sin", sin, persistent=False)
        self.sin = sin
        
        self.register_buffer("cos", cos, persistent=False)
        self.cos = cos

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_k"], 
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        if token_positions is None:
            batch_dims = x.shape[:-2]
            seq_len = x.shape[-2]
            token_positions = torch.arange(0, seq_len, device=x.device)
            token_positions = token_positions.expand(*batch_dims, seq_len)
        assert x.shape[-2] == token_positions.shape[-1], f"x shape {x.shape} and token_positions shape {token_positions.shape} are not compatible"
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        assert cos.shape[-1] == self.d_k // 2, f"cos last dimension {cos.shape[-1]} does not match d_k {self.d_k // 2}"
        assert sin.shape[-1] == self.d_k // 2, f"sin last dimension {sin.shape[-1]} does not match d_k {self.d_k // 2}"
        assert x.shape[-1] == self.d_k, f"x last dimension {x.shape[-1]} does not match d_k {self.d_k}"
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_sin = x_even * sin + x_odd * cos
        x_out = torch.zeros_like(x)
        x_out[..., ::2] = x_even_rot
        x_out[..., 1::2] = x_odd_sin
        assert x_out.shape == x.shape, f"x_out shape {x_out.shape} does not match x shape {x.shape}"
        return x_out
    
    def extra_repr(self) -> str:
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"


if __name__ == "__main__":
    rope = RoPE(theta=10000, d_k=64, max_seq_len=2048)
    x = torch.randn(2, 128, 64)
    token_positions = torch.arange(0, 128).unsqueeze(0).repeat(2, 1)
    print(x.shape, token_positions.shape)
    rope(x, token_positions)
