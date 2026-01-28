# Temporal Spectral Geodesic Flow Matching (TS-GFM)

**Date:** 2026-01-28
**Status:** Approved
**Goal:** Learn continuous-time dynamics on Stiefel manifold for spectral frame evolution

---

## Overview

Learn a conditional continuous-time vector field that captures how spectral geometry (eigenvectors + eigenvalues) evolves over time.

**Core Principle:** Do not design interpolation paths. Let the manifold geometry itself define the reference flow via Log/Exp maps, and train a neural vector field to match that intrinsic flow.

This is **Riemannian Flow Matching**, not hand-crafted interpolation.

**Enables:**
- Learned (non-geodesic) interpolation between time steps
- Extrapolation into the future
- Identification of stable vs transient subspaces (Temporal Intrinsic Dimension)

---

## Data

Temporal sequence of spectral frames (no alignment/canonicalization):

```
(Φ_0, λ_0), (Φ_1, λ_1), ..., (Φ_T, λ_T)
```

- `Φ_t ∈ R^{n×k}` with `Φ_t^T Φ_t = I_k` (Stiefel)
- `λ_t ∈ R^k` (eigenvalues)

Each frame from: graph → Laplacian → top-k eigenpairs.

---

## State Space

- **Eigenvectors:** Stiefel manifold `St(n, k)`
- **Eigenvalues:** `R^k`
- **Intrinsic state:** Subspace `P = Φ Φ^T` (Grassmann-invariant)

---

## Model

Conditional continuous-time vector field:

**Eigenvector dynamics:**
```
Φ̇(t) = Π_{T_Φ(t) St}(g_θ(Φ(t), t | Φ_1, λ_1))
```
- `g_θ` is neural network
- Output projected to Stiefel tangent space
- Integration uses QR retraction

**Eigenvalue dynamics:**
```
λ̇(t) = f_θ(λ(t), t | λ_1)
```

---

## Training: Conditional Riemannian Flow Matching

Training on consecutive frame pairs without manual interpolation.

### Step 1 — Sample target
```
(Φ_1, λ_1) = (Φ_{t+1}, λ_{t+1})
```

### Step 2 — Sample random start (noise)
```
Φ_0 ~ Uniform(St(n, k))
λ_0 ~ N(0, σ²I)
```
This defines a noise → data bridge.

### Step 3 — Sample time
```
s ~ Uniform(0, 1)
```

### Step 4 — Define reference bridge (intrinsic, not manual)

**Eigenvectors (Stiefel geodesic):**
```
Φ_s = Exp_{Φ_0}(s · Log_{Φ_0}(Φ_1))
```
- Uniquely determined by geometry
- Invariant to sign flips/permutations at subspace level
- Not chosen by user

**Eigenvalues (Euclidean bridge):**
```
λ_s = (1 - s) λ_0 + s λ_1
```

### Step 5 — Define target velocities (canonical)

**Eigenvector target velocity:**
```
Φ̇_s* = (1 / (1 - s + ε)) · Log_{Φ_s}(Φ_1)
```
Intrinsic geodesic velocity pointing toward endpoint.

**Eigenvalue target velocity:**
```
λ̇_s* = λ_1 - λ_0
```

### Step 6 — Predict velocities
```
Φ̇_s = Π_{T_Φ_s} g_θ(Φ_s, s | Φ_1, λ_1)
λ̇_s = f_θ(λ_s, s | λ_1)
```

### Step 7 — Losses

**Eigenvector flow-matching loss:**
```
L_Φ = ||Φ̇_s - Φ̇_s*||_F²
```

**Eigenvalue flow-matching loss:**
```
L_λ = ||λ̇_s - λ̇_s*||_2²
```

**Optional: Grassmann endpoint consistency:**
```
L_Grass = ||Φ̂_1 Φ̂_1^T - Φ_1 Φ_1^T||_F²
```
After integrating s=0→1. Ignores sign flips and crossings.

**Total loss:**
```
L = L_Φ + α L_λ + β L_Grass
```

---

## Inference

Given observed frames `(Φ_A, λ_A)` and `(Φ_B, λ_B)`:
1. Integrate learned dynamics from `(Φ_A, λ_A)`
2. Condition on `(Φ_B, λ_B)`
3. Stop at intermediate `s` for interpolation, or continue past 1 for extrapolation

---

## File Structure

```
src/temporal_spectral_flow/
├── stiefel.py            # Stiefel manifold: Exp, Log, tangent projection, retraction
├── grassmann.py          # Grassmann metrics: distance, projection
├── networks.py           # Neural velocity fields g_θ, f_θ
├── flow.py               # GeodesicFlowModel: velocity, integrate
├── training.py           # Riemannian flow matching training loop
├── preprocessing.py      # Graph → spectral frames (no alignment)
└── inference.py          # Interpolation, extrapolation

tests/
├── test_stiefel.py       # Exp/Log inverse, tangent validity, retraction on manifold
├── test_grassmann.py     # Distance invariance, projection properties
└── test_training.py      # Loss decreases, integration stays on manifold
```

---

## Module Specifications

### `stiefel.py`

**`stiefel_exp(Phi, V)`**
- Exponential map: move from Φ along tangent V
- Input: Φ ∈ St(n,k), V ∈ T_Φ St (tangent)
- Output: Φ' ∈ St(n,k)
- Implementation: matrix exponential or closed-form for small k

**`stiefel_log(Phi_0, Phi_1)`**
- Logarithm map: tangent vector at Φ_0 pointing to Φ_1
- Input: Φ_0, Φ_1 ∈ St(n,k)
- Output: V ∈ T_{Φ_0} St
- This is the key operation for geodesic flow matching

**`stiefel_geodesic(Phi_0, Phi_1, s)`**
- Geodesic interpolation: Exp_{Φ_0}(s · Log_{Φ_0}(Φ_1))
- Input: endpoints and s ∈ [0, 1]
- Output: Φ_s ∈ St(n,k)

**`project_to_tangent(Z, Phi)`**
- Project ambient Z to tangent space at Phi
- V = Z - Φ @ sym(Φ^T @ Z)
- Satisfies: Φ^T V + V^T Φ = 0

**`stiefel_retract(Phi, V, dt)`**
- QR retraction for integration
- Y = Φ + dt * V
- Q, R = qr(Y), sign correction

**`uniform_stiefel(n, k)`**
- Sample uniformly from St(n, k)
- QR of Gaussian matrix

### `grassmann.py`

**`to_projection(Phi)`**
- P = Φ @ Φ^T
- Invariant to Φ → Φ R for R ∈ O(k)

**`grassmann_distance(Phi_0, Phi_1)`**
- Geodesic distance via principal angles
- M = Φ_0^T @ Φ_1, SVD, θ = arccos(clamp(s))
- Return ||θ||

**`projection_frobenius(Phi_0, Phi_1)`**
- ||P_0 - P_1||_F
- Alternative invariant metric

### `networks.py`

**`SinusoidalTimeEmbedding(embed_dim)`**
- Fourier features for time

**`StiefelVelocityField(n, k, hidden_dim, n_blocks)`**
- Input: Φ_current (n,k), Φ_target (n,k), λ_target (k,), t
- Process: flatten, concatenate, MLP
- Output: raw Z (n,k), then project to tangent at Φ_current
- Returns V ∈ T_{Φ_current} St

**`EigenvalueVelocityField(k, hidden_dim, n_blocks)`**
- Input: λ_current (k,), λ_target (k,), t
- Output: v_λ (k,)

### `flow.py`

**`class GeodesicFlowModel`**
- Contains StiefelVelocityField, EigenvalueVelocityField

**`velocity(Phi, lambda_, Phi_target, lambda_target, t)`**
- Returns (v_Phi, v_lambda)

**`integrate(Phi_0, lambda_0, Phi_target, lambda_target, t_end, n_steps)`**
- Euler integration with Stiefel retraction
- Returns trajectory

### `training.py`

**`sample_noise_start(n, k, sigma_lambda)`**
- Φ_0 ~ Uniform(St(n,k))
- λ_0 ~ N(0, σ²I)

**`compute_target_velocity(Phi_s, Phi_1, lambda_0, lambda_1, s, eps=1e-4)`**
- Φ̇_s* = Log_{Φ_s}(Φ_1) / (1 - s + eps)
- λ̇_s* = λ_1 - λ_0

**`train_step(model, Phi_1, lambda_1, optimizer, sigma_lambda)`**
```
# Sample noise start
Phi_0 = uniform_stiefel(n, k)
lambda_0 = sigma_lambda * randn(k)

# Sample time
s = uniform(0, 1)

# Geodesic interpolation (reference bridge)
Phi_s = stiefel_geodesic(Phi_0, Phi_1, s)
lambda_s = (1 - s) * lambda_0 + s * lambda_1

# Target velocities
v_Phi_target = stiefel_log(Phi_s, Phi_1) / (1 - s + eps)
v_lambda_target = lambda_1 - lambda_0

# Predicted velocities
v_Phi_pred, v_lambda_pred = model.velocity(
    Phi_s, lambda_s, Phi_1, lambda_1, s
)

# Losses
L_Phi = ||v_Phi_pred - v_Phi_target||_F²
L_lambda = ||v_lambda_pred - v_lambda_target||²

loss = L_Phi + alpha * L_lambda
loss.backward()
optimizer.step()
```

**`train_epoch(model, frames, optimizer, ...)`**
- For each consecutive pair (Φ_t, λ_t), (Φ_{t+1}, λ_{t+1}):
  - Use (Φ_{t+1}, λ_{t+1}) as target
  - Call train_step

### `preprocessing.py`

**`compute_spectral_frame(X, k)`**
- Build graph → Laplacian → k smallest eigenpairs
- Return (Φ, λ, W)
- **No alignment**

**`prepare_sequence(X_sequence, k)`**
- Apply to each timestep
- Return list of (Φ, λ, W)

### `inference.py`

**`interpolate(model, Phi_A, lambda_A, Phi_B, lambda_B, s_target, n_steps)`**
- Integrate from (Φ_A, λ_A) conditioned on (Φ_B, λ_B)
- Stop at s_target

**`extrapolate(model, Phi_start, lambda_start, Phi_cond, lambda_cond, t_forward, n_steps)`**
- Integrate beyond t=1 for extrapolation

---

## Tests

### `test_stiefel.py`

- `test_exp_log_inverse`: Log then Exp returns original
- `test_log_exp_inverse`: Exp then Log returns original tangent
- `test_tangent_is_valid`: Φ^T V + V^T Φ = 0
- `test_retraction_on_manifold`: Q^T Q = I
- `test_geodesic_endpoints`: s=0 → Φ_0, s=1 → Φ_1
- `test_uniform_sample_valid`: Q^T Q = I

### `test_grassmann.py`

- `test_projection_invariant_to_rotation`: P(Φ) = P(Φ R)
- `test_distance_invariant`: d(Φ_0, Φ_1) = d(Φ_0 R, Φ_1 R')
- `test_distance_zero_same_subspace`: d(Φ, Φ R) = 0

### `test_training.py`

- `test_loss_decreases`: Training reduces loss
- `test_integration_on_manifold`: Φ^T Φ = I throughout

---

## Implementation Order

1. `stiefel.py` + tests (critical: Exp/Log must be correct)
2. `grassmann.py` + tests
3. `networks.py`
4. `flow.py`
5. `preprocessing.py`
6. `training.py` + tests
7. `inference.py`
8. Example script using existing `viz/` machinery

---

## Key Differences from Original Plan

| Aspect | Original (Wrong) | Correct (This Plan) |
|--------|------------------|---------------------|
| Interpolation | Hand-crafted linear + QR | Geodesic via Exp/Log |
| Alignment | Procrustes before interpolation | None needed |
| Reference velocity | Manual difference | Intrinsic Log / (1-s) |
| Noise distribution | Previous frame | Uniform on Stiefel |
| Grassmann loss | Primary loss | Optional endpoint check |
| Core principle | Force invariance | Let geometry define flow |
