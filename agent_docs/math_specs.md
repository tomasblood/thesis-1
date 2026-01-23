# Mathematical Specifications

## Stiefel Manifold

The Stiefel manifold St(N, k) consists of orthonormal k-frames in R^N:

```
St(N, k) = { Φ ∈ R^{N×k} : Φ^T Φ = I_k }
```

### Tangent Space

At point Φ, the tangent space is:

```
T_Φ St(N, k) = { V ∈ R^{N×k} : Φ^T V + V^T Φ = 0 }
```

### Projection to Tangent Space

```
proj_Φ(V) = V - Φ sym(Φ^T V)
```

where sym(A) = (A + A^T) / 2.

### Retraction (QR)

```
R_Φ(V) = qr(Φ + V)
```

## Spectral Flow

Given temporal sequence {X_t}, compute spectral embeddings {Φ_t} and learn flow:

```
dΦ/dt = v(Φ, t)
```

where v is a learned velocity field with v(Φ, t) ∈ T_Φ St(N, k).

## Optimal Transport Alignment

For embeddings Φ_s, Φ_t of different sizes, find transport plan π minimizing:

```
min_π ∑_{i,j} π_{ij} ||φ_s^i - φ_t^j||^2
```

subject to marginal constraints (relaxed for unbalanced OT).
