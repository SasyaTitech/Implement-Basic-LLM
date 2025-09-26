import torch
from torch import nn, Tensor
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from cs336_basics.model.attention import MultiHeadAttention
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.rope import RoPE
from cs336_basics.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    multi_head_attention: MultiHeadAttention
    rms_norm1: RMSNorm
    rms_norm2: RMSNorm
    ffn: SwiGLU

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_k: int = d_model // num_heads
        if rope is not None:
            assert rope.d_k == d_k, f"RoPE d_k {rope.d_k} must match model d_model/num_heads {d_k}"
        else:
            rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)

        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "batch_size ... seq_len d_model"],
        token_positions: Int[Tensor, "batch_size ... seq_len"] | None = None,
    ) -> Float[Tensor, " batch_size ... seq_len d_model"]:
        norm_x = self.rms_norm1(x)
        atten_x = self.multi_head_attention(norm_x, token_positions=token_positions)
        y0 = x + atten_x
        norm_y0 = self.rms_norm2(y0)
        ffn_y = self.ffn(norm_y0)
        y = y0 + ffn_y
        return y

    def extra_repr(self) -> str:
        return f"rms_norm1={self.rms_norm1}, multi_head_attention={self.multi_head_attention}, rms_norm2={self.rms_norm2}, ffn={self.ffn}"
