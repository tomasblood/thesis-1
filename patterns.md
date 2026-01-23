# Code Patterns

Reference patterns for research-grade PyTorch. **Pointers, not copies** — implementations live in your codebase.

---

## Standard Utilities

Location: `src/utils.py`

Implement these once per project:
- `exists(val)` — `val is not None`
- `default(val, d)` — return `val` if exists else `d`
- `identity(t)` — passthrough
- `cycle(dl)` — infinite dataloader iterator

---

## Tensor Operations

**Library:** `einops`

| Operation | Pattern |
|-----------|---------|
| Reshape | `rearrange(x, 'b c h w -> b (h w) c')` |
| Reduce | `reduce(x, 'b n d -> b d', 'mean')` |
| Broadcast | `repeat(x, 'b d -> b n d', n=seq_len)` |
| Multimodal pack | `pack([text, image], 'b * d')` |

Docs: einops.aimo.ai

---

## Type Hints

**Libraries:** `beartype`, `jaxtyping`

Pattern:
```
@beartype
def forward(self, x: Float[Tensor, 'b n d']) -> Float[Tensor, 'b n d']
```

---

## Normalization

**Use:** `RMSNorm` (pre-norm), never vanilla `LayerNorm`

Location: `src/layers/norm.py` (when implemented)

Reference: lucidrains repos — search "RMSNorm"

For conditioning: add scale/shift projection, zero-init weights.

---

## FeedForward

**Use:** GEGLU or SwiGLU with ~4/3 expansion

Location: `src/layers/feedforward.py` (when implemented)

Reference: lucidrains repos — search "GEGLU"

---

## Attention

**Use:** `torch.nn.functional.scaled_dot_product_attention`

- Automatic Flash Attention backend
- Apply RoPE to Q, K
- QK-Norm for stability at scale

Reference: PyTorch docs, lucidrains `attend.py`

---

## Fourier Features

For continuous coordinate encoding.

Location: `src/layers/encoding.py` (when implemented)

Pattern: `sin/cos` of `x * freqs * 2π`, concatenated

---

## Numerical Stability

| Issue | Solution |
|-------|----------|
| Log-sum-exp | `torch.logsumexp()` |
| Safe log | `torch.log(x.clamp(min=1e-8))` |
| NaN check | `assert not torch.isnan(x).any()` |
| Gradient explosion | `clip_grad_norm_(params, 1.0)` |

---

## Logging

**Library:** `loguru`

Pattern:
```
logger.info(f"step={step} loss={loss:.4f}")
```

Location: configure in `train.py` entry point

---

## Config

**Format:** YAML + dataclasses

Files:
- `config.yaml` — default values
- `src/config.py` — dataclass definition, `load_config()`, `save_config()`

---

## Reproducibility

Location: `src/utils.py`

Implement `set_seed(seed)` covering:
- `random.seed()`
- `np.random.seed()`
- `torch.manual_seed()`
- `torch.cuda.manual_seed_all()`

---

## Checkpoints

Location: `src/training/checkpoint.py` (when implemented)

Save: `step`, `config`, `model`, `ema`, `optimizer`

Naming: `checkpoints/step_{step:06d}.pt`

---

## Trainer

Location: `src/training/trainer.py` (when implemented)

**Libraries:** `accelerate`, `ema-pytorch`

Key patterns:
- `Accelerator()` for device management
- `EMA(model, beta=0.9999)`
- `cycle(dataloader)` for step-based training
- Log every 100 steps, checkpoint every 10k

---

## Testing

Location: `tests/test_model.py`

Pattern: `assert`-based shape and gradient checks

```
assert out.shape == expected
assert x.grad is not None
assert not torch.isnan(x.grad).any()
```

Run: `python tests/test_model.py`

---

## CLI

**Library:** `fire`

Pattern: expose `train()` function, fire auto-maps args

---

## Finding Implementations

| Need | Source |
|------|--------|
| Transformer components | lucidrains GitHub |
| einops patterns | einops.aimo.ai |
| Flash Attention | `torch.nn.functional.scaled_dot_product_attention` |
| Live docs | Context7 MCP (if installed) |

---

*When you implement a pattern, update this file with the actual `file:line` reference.*
