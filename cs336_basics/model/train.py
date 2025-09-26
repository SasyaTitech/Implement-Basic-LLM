from types import NoneType
from typing import Callable, Iterable, Optional, overload
import torch
from torch import nn
import numpy as np
import random
from einops import rearrange, einsum
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
from cs336_basics.model.linear import Linear

is_main_file = __name__ == "__main__"

# Torch
seed = 0
torch.manual_seed(seed)
# NumPy
np.random.seed(seed)
# Python
random.seed(seed)


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


class LinearModel(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Linear(dim, dim))
        self.final_layer = Linear(dim, 1)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[torch.Tensor, "batch dim"]) -> Float[torch.Tensor, "batch"]:
        B, D = x.shape
        assert (
            D == self.layers[0].in_features
        ), f"Input dimension {D} does not match model dimension {self.layers[0].in_features}"
        for layer in self.layers:
            x = layer(x)

        x = self.final_layer(x)
        assert x.shape == (B, 1), f"Output shape {x.shape} does not match expected shape {(B, 1)}"

        x = x.squeeze(-1)
        assert x.shape == (B,), f"Squeezed output shape {x.shape} does not match expected shape {(B,)}"
        return x


class AdaGrab(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self) -> None: # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                g2 = state.get("g2", torch.zeros_like(param))
                g2 += torch.square(param.grad)
                state["g2"] = g2

                grad = param.grad.data
                param.data -= lr * grad / (torch.sqrt(g2) + 1e-8)
        return None


if is_main_file:
    dim = 4
    batch_size = 2
    model = LinearModel(dim, 2)
    model = model.to(get_device())
    for param in model.parameters():
        print(param.shape)
        print(param.device)
    x = torch.randn(batch_size, dim).to(get_device())
    pred_y = model(x)
    label = torch.randn(batch_size).to(get_device())
    loss: torch.Tensor = ((pred_y - label) ** 2).mean()
    loss.backward()
    print(pred_y)

    optimizer = AdaGrab(model.parameters(), lr=1e-2)
    optimizer.step()
    print(optimizer.state)

    optimizer.zero_grad(set_to_none=True)
