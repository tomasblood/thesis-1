# Mathematical Specifications

Reference equations for flow matching, optimal transport, and spectral methods.

---

## Flow Matching (Curved Trajectories)

**NOT rectified flows.** Preserve trajectory curvature via optimal transport or geodesic paths.

### Interpolation

General form:
$$x_t = \psi_t(x_0, x_1)$$

Linear (baseline):
$$x_t = (1-t) x_0 + t x_1$$

Curved (OT/geodesic):
$$x_t = \exp_{x_0}(t \cdot \log_{x_0}(x_1))$$

### Velocity Field

The velocity is the time derivative of the interpolation:
$$v_t = \frac{\partial x_t}{\partial t}$$

For linear interpolation:
$$v_t = x_1 - x_0$$

### Training Objective

```python
def flow_matching_loss(model, x0, x1, t):
    x_t = interpolate(x0, x1, t)
    v_target = compute_velocity(x0, x1, t)
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v_target)
```

### Sampling (ODE)

Euler method with explicit precision:

```python
@torch.no_grad()
def sample(model, shape, steps=50, dtype=torch.bfloat16):
    z = torch.randn(shape, device=device, dtype=dtype)
    dt = 1.0 / steps
    for t in torch.linspace(0, 1 - dt, steps):
        v = model(z, t)
        z = z + v * dt
    return z
```

---

## Optimal Transport

### Wasserstein Distance

$$W_p(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x - y\|^p d\gamma(x, y) \right)^{1/p}$$

### Monge Problem

Find map $T: X \to Y$ minimizing:
$$\inf_T \int \|x - T(x)\|^2 d\mu(x)$$

### Kantorovich Relaxation

Optimize over couplings $\gamma$ instead of maps.

### Sinkhorn Algorithm

Entropic regularization for differentiable OT:

```python
def sinkhorn(C, reg=0.1, n_iters=100):
    """
    C: (n, m) cost matrix
    Returns: (n, m) transport plan
    """
    K = torch.exp(-C / reg)
    u = torch.ones(C.shape[0], device=C.device)
    for _ in range(n_iters):
        v = 1.0 / (K.T @ u)
        u = 1.0 / (K @ v)
    return torch.diag(u) @ K @ torch.diag(v)
```

---

## Spectral Methods

### Fourier Features

For positional encoding of continuous coordinates:

$$\gamma(x) = [\sin(2\pi \sigma^{j/m} x), \cos(2\pi \sigma^{j/m} x)]_{j=0}^{m-1}$$

### Graph Laplacian

Unnormalized:
$$L = D - A$$

Normalized:
$$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$

### Eigendecomposition

```python
# For small matrices
eigenvalues, eigenvectors = torch.linalg.eigh(L)

# For large sparse matrices
eigenvalues, eigenvectors = torch.lobpcg(L, k=num_eigenvectors)

# Use float64 for numerical precision
L = L.to(torch.float64)
```

---

## Riemannian Geometry

### Exponential Map

$$\exp_p(v) = \gamma(1)$$

where $\gamma$ is the geodesic starting at $p$ with initial velocity $v$.

### Logarithm Map

$$\log_p(q) = v$$

where $v$ is the initial velocity of the geodesic from $p$ to $q$.

### Parallel Transport

Transport vector $v$ along geodesic from $p$ to $q$:
$$\Gamma_{p \to q}(v)$$

---

## Numerical Stability Notes

- Use `torch.logsumexp` for log-domain operations
- Clamp before `log`: `torch.log(x.clamp(min=1e-8))`
- Use `float64` for eigendecomposition
- Check for NaNs after each operation during debugging
- Gradient clipping: `clip_grad_norm_(params, max_norm=1.0)`
