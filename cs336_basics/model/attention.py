from typing import Optional
import torch
from torch import nn, Tensor
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        query: Float[Tensor, "batch ... n d_k"],
        key: Float[Tensor, "batch ... m d_k"],
        value: Float[Tensor, "batch ... m d_v"],
        mask: Bool[Tensor, "batch ... n m"] | None = None,
    ) -> Float[Tensor, "batch ... n d_v"]:
        d_k = query.shape[-1]
        assert key.shape[-1] == query.shape[-1], f"Key shape {key.shape} must match query shape {query.shape}"
        assert value.shape[-2] == key.shape[-2], f"Value shape {value.shape} must match key shape {key.shape}"

        attention_scores = einsum(query, key, "batch ... n d_k, batch ... m d_k -> batch ... n m")
        attention_scores = attention_scores / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = self.softmax(attention_scores)
        value = einsum(attention_weights, value, "batch ... n m, batch ... m d_v -> batch ... n d_v")
        return value
