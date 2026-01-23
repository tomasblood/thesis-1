# Code Patterns

Common implementation patterns for this project.

## RMSNorm

Use RMSNorm instead of LayerNorm:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

## Einops Patterns

Always use einops for tensor operations:

```python
from einops import rearrange, reduce, repeat

# Instead of: x.view(batch, -1)
x = rearrange(x, 'b n k -> b (n k)')

# Instead of: x.transpose(-2, -1)
x = rearrange(x, 'b n k -> b k n')

# Instead of: x.unsqueeze(1)
x = rearrange(x, 'b d -> b 1 d')

# Batch matrix multiply
result = einsum(A, B, 'b n k, b k m -> b n m')
```

## Type Annotations with beartype + jaxtyping

```python
from beartype import beartype
from jaxtyping import Float, Int
import torch

@beartype
def process_embedding(
    Phi: Float[torch.Tensor, "batch n k"],
    t: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "batch n k"]:
    ...
```

## Stiefel Manifold Operations

```python
# Tangent space projection
# V_tan = V - Phi @ sym(Phi^T @ V)
PhiTV = einsum(Phi, V, 'b n k, b n k -> b k k')
sym_PhiTV = (PhiTV + rearrange(PhiTV, 'b i j -> b j i')) / 2
V_tangent = V - einsum(Phi, sym_PhiTV, 'b n k, b k l -> b n l')

# QR retraction
Q, R = torch.linalg.qr(Phi + dt * V)
```
