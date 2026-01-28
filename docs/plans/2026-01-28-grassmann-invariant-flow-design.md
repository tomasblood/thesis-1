# Grassmann-Invariant Spectral Flow Matching

**Date:** 2026-01-28
**Status:** Approved
**Goal:** Build a flow model that learns spectral frame dynamics on the Stiefel manifold, with all alignment and loss functions defined through Grassmannian projection

---

## Overview

**Key principle**: Represent on Stiefel (compact), evaluate on Grassmann (invariant).

This eliminates sign flips, permutations, and eigenvalue ordering issues by construction. The projection P = ΦΦᵀ is invariant to Φ → ΦR for any R ∈ O(k).

---

## File Structure

```
src/temporal_spectral_flow/
├── grassmann_ops.py      # Grassmannian operations for invariant computations
├── stiefel_ops.py        # Stiefel manifold operations
├── networks.py           # Neural network components
├── unified_flow.py       # Main flow model with Grassmann-invariant loss
├── training.py           # Training with Grassmann-invariant loss
├── interpolation.py      # Inference using trained model
└── preprocessing.py      # Minimal preprocessing (no canonicalization needed)

tests/
├── test_grassmann_ops.py
├── test_stiefel_ops.py
└── test_training.py

scripts/
└── example_grassmann_flow.py

# Visualization uses existing viz/ machinery (no new files needed):
# - viz/styles.py          → VizStyle, ColorPalette, create_figure()
# - viz/components/        → eigenvalue/eigenvector/embedding panels
# - viz/static.py          → flow trajectories, comparisons
```

---

## Module Specifications

### `grassmann_ops.py`

**`to_projection(Phi)`**
- Input: Φ ∈ R^{n×k} with orthonormal columns
- Output: P = Φ @ Φ.T ∈ R^{n×n}
- P is invariant to Φ → ΦR for any R ∈ O(k)

**`grassmann_distance(Phi_0, Phi_1)`**
- Geodesic distance via principal angles
- M = Phi_0.T @ Phi_1
- SVD: U, s, Vt = svd(M)
- Clip singular values to [-1, 1]
- Principal angles: θ = arccos(s)
- Return: norm(θ)

**`projection_distance(Phi_0, Phi_1)`**
- Frobenius norm: ||P_0 - P_1||_F
- Invariant to all basis choices

**`grassmann_log_approx(Phi_0, Phi_1)`**
- Approximate logarithm map on Grassmannian
- M = Phi_0.T @ Phi_1
- SVD: U, s, Vt = svd(M)
- Optimal rotation: R = U @ Vt
- Aligned: Phi_1_aligned = Phi_1 @ R.T
- Tangent: V = Phi_1_aligned - Phi_0 @ (Phi_0.T @ Phi_1_aligned)
- Project to Stiefel tangent space

**`optimal_rotation(Phi_0, Phi_1)`**
- Find R ∈ O(k) minimizing ||Phi_0 - Phi_1 @ R||_F
- M = Phi_0.T @ Phi_1
- SVD: U, s, Vt = svd(M)
- R = Vt.T @ U.T

---

### `stiefel_ops.py`

**`project_to_tangent(Z, Phi)`**
- Project ambient Z to tangent space at Phi
- A = Phi.T @ Z
- skew_A = (A - A.T) / 2
- V = Phi @ skew_A + (I - Phi @ Phi.T) @ Z
- Satisfies: Phi.T @ V + V.T @ Phi = 0

**`stiefel_retract(Phi, V, dt)`**
- QR retraction along tangent V
- Y = Phi + dt * V
- Q, R = qr(Y)
- Sign correction via diagonal of R

**`random_stiefel(n, k)`**
- Uniform sample from Stiefel manifold
- QR factorization of random Gaussian

---

### `networks.py`

**`SinusoidalTimeEmbedding(embed_dim)`**
- Frequencies: [1, 2, 4, 8, 16, ...]
- Output: [sin(2π·f·t), cos(2π·f·t)]

**`ResidualBlock(dim)`**
- x + Linear(GELU(LayerNorm(Linear(x))))

**`ResidualMLP(input_dim, hidden_dim, output_dim, n_blocks)`**
- Input projection → ResidualBlocks → Output projection

**`EigenvalueVelocityField(k, hidden_dim=256, n_blocks=3, time_embed_dim=64)`**
- Input: lambda_current, lambda_target, t
- Output: v_lambda (k,)

**`EigenvectorVelocityField(n, k, hidden_dim=1024, n_blocks=4, time_embed_dim=64)`**
- Input: Phi_current, Phi_target, lambda_current, t
- Output: project_to_tangent(Z, Phi_current)

---

### `unified_flow.py`

**`class GrassmannInvariantFlow`**

- Contains EigenvalueVelocityField and EigenvectorVelocityField

**`velocity(Phi_current, lambda_current, Phi_target, lambda_target, t)`**
- Returns (v_lambda, v_Phi)

**`integrate(Phi_start, lambda_start, Phi_target, lambda_target, t_end, n_steps)`**
- Euler integration with Stiefel retraction

---

### `training.py`

**`grassmann_invariant_loss(Phi_pred, Phi_target)`**
- loss = ||P_pred - P_target||_F^2
- Invariant to sign, permutation, rotation

**`principal_angle_loss(Phi_pred, Phi_target)`**
- loss = sum(angles^2) via SVD

**`train_step(model, Phi_0, lambda_0, Phi_1, lambda_1, optimizer)`**
1. Sample interpolation time s ~ U(0,1)
2. Interpolate eigenvalues linearly
3. Interpolate eigenvectors via optimal rotation + QR retraction
4. Compute target velocity
5. Predict velocity
6. MSE loss on velocities
7. Backprop

**`train_epoch(model, frames, optimizer, batch_size)`**
- Shuffle consecutive pairs, train on batches

**`train(model, frames, n_epochs, batch_size, lr)`**
- Main training loop with Adam

---

### `interpolation.py`

**`interpolate(model, Phi_0, lambda_0, Phi_1, lambda_1, s_target, n_steps=50)`**
- Integrate from t=0 to t=s_target

**`get_trajectory(model, Phi_0, lambda_0, Phi_1, lambda_1, n_points=10)`**
- Full trajectory from frame 0 to frame 1

**`extrapolate(model, frames, n_steps_forward, dt=0.1)`**
- Predict beyond last observation

---

### Visualization (uses existing `viz/` machinery)

**No new visualization module needed.** Use existing tools from `viz/`:

**From `viz/styles.py`:**
- `VizStyle`, `ColorPalette` - consistent styling
- `create_figure()`, `save_figure()` - figure helpers
- `get_time_colors()` - time-based coloring

**From `viz/components/`:**
- `plot_eigenvalue_trajectories()` - eigenvalue evolution over time
- `plot_eigenvector_heatmap()` - heatmap of Phi matrix
- `plot_spectral_embedding()` - 2D scatter with trails
- `plot_embedding_trajectory()` - node trajectories through spectral space

**From `viz/static.py`:**
- `plot_comparison_panel()` - side-by-side flow comparisons
- `plot_density_evolution()` - multi-panel time evolution

**New helper in `grassmann_ops.py`:**
- `plot_projection_distance_matrix(Phi_sequence)` - pairwise Grassmann distance heatmap
  - Uses `grassmann_distance()` to compute D[i,j]
  - Styled with `VizStyle` and `create_figure()`

---

### `preprocessing.py`

**`compute_spectral_frame(X, k, sigma=None)`**
- Build kNN/kernel graph → Laplacian → k smallest eigenpairs
- Return (Phi, lambda, W)

**`prepare_sequence(X_sequence, k)`**
- Apply to each timestep
- **No alignment needed** - Grassmann loss handles invariance

**`sort_eigenvalues(Phi, lambda_)`**
- Optional convenience sorting

---

## Tests

### `test_grassmann_ops.py`
- Projection invariant to sign, permutation, rotation
- Distance invariant to basis choice
- Distance zero for same subspace
- Optimal rotation minimizes distance

### `test_stiefel_ops.py`
- Tangent projection satisfies skew-symmetry condition
- Retraction stays on manifold
- Random samples are valid

### `test_training.py`
- Grassmann loss invariant
- Training loss decreases
- Integration preserves manifold

---

## Implementation Order

1. `grassmann_ops.py` + tests
2. `stiefel_ops.py` + tests
3. `networks.py`
4. `unified_flow.py`
5. `preprocessing.py`
6. `training.py` + tests
7. `interpolation.py`
8. `scripts/example_grassmann_flow.py` (uses existing `viz/` machinery)
