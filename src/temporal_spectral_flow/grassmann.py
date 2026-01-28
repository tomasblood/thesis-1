"""
Grassmannian operations for invariant spectral comparisons.

The Grassmannian Gr(n, k) is the space of k-dimensional subspaces of R^n.
A point on Gr(n, k) can be represented by the projection matrix P = Φ Φ^T
where Φ ∈ St(n, k) is any orthonormal basis for the subspace.

Key property: P is invariant to Φ → Φ R for any R ∈ O(k).
This eliminates sign flips, permutations, and basis rotations by construction.

Used for:
- Grassmann-invariant loss functions
- Evaluating subspace similarity regardless of basis choice
- Endpoint consistency in flow matching
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
import torch


# =============================================================================
# NumPy implementations
# =============================================================================


def to_projection(Phi: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute projection matrix from orthonormal basis.

    Args:
        Phi: Orthonormal matrix of shape (n, k), Φ^T Φ = I_k

    Returns:
        Projection matrix P = Φ Φ^T of shape (n, n)

    Note:
        P is invariant to Φ → Φ R for any R ∈ O(k).
        This is the key property for Grassmann-invariant comparisons.
    """
    return Phi @ Phi.T


def grassmann_distance(
    Phi_0: NDArray[np.floating],
    Phi_1: NDArray[np.floating],
) -> float:
    """
    Compute geodesic distance on the Grassmannian via principal angles.

    The geodesic distance is d(Φ_0, Φ_1) = ||θ||_2 where θ are the
    principal angles between the subspaces.

    Args:
        Phi_0: First orthonormal basis (n, k)
        Phi_1: Second orthonormal basis (n, k)

    Returns:
        Grassmann geodesic distance (invariant to basis choice)

    Note:
        This distance is invariant to:
        - Sign flips of columns
        - Permutations of columns
        - Rotations within eigenspaces (Φ → Φ R)
    """
    # Compute M = Φ_0^T Φ_1
    M = Phi_0.T @ Phi_1

    # SVD gives singular values = cos(principal angles)
    _, s, _ = svd(M, full_matrices=False)

    # Clamp for numerical stability
    s = np.clip(s, -1.0, 1.0)

    # Principal angles
    theta = np.arccos(s)

    # Geodesic distance
    return float(np.linalg.norm(theta))


def projection_frobenius(
    Phi_0: NDArray[np.floating],
    Phi_1: NDArray[np.floating],
) -> float:
    """
    Compute Frobenius distance between projection matrices.

    Alternative to geodesic distance, also Grassmann-invariant.

    Args:
        Phi_0: First orthonormal basis (n, k)
        Phi_1: Second orthonormal basis (n, k)

    Returns:
        ||P_0 - P_1||_F where P = Φ Φ^T
    """
    P_0 = to_projection(Phi_0)
    P_1 = to_projection(Phi_1)
    return float(np.linalg.norm(P_0 - P_1, ord='fro'))


def projection_frobenius_efficient(
    Phi_0: NDArray[np.floating],
    Phi_1: NDArray[np.floating],
) -> float:
    """
    Efficient computation of ||P_0 - P_1||_F without forming full projections.

    Uses the identity:
        ||P_0 - P_1||_F^2 = 2k - 2||Φ_0^T Φ_1||_F^2

    Args:
        Phi_0: First orthonormal basis (n, k)
        Phi_1: Second orthonormal basis (n, k)

    Returns:
        ||P_0 - P_1||_F
    """
    k = Phi_0.shape[1]
    M = Phi_0.T @ Phi_1
    return float(np.sqrt(max(0.0, 2 * k - 2 * np.sum(M ** 2))))


def principal_angles(
    Phi_0: NDArray[np.floating],
    Phi_1: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute principal angles between two subspaces.

    Args:
        Phi_0: First orthonormal basis (n, k)
        Phi_1: Second orthonormal basis (n, k)

    Returns:
        Array of k principal angles in [0, π/2]
    """
    M = Phi_0.T @ Phi_1
    _, s, _ = svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


# =============================================================================
# PyTorch implementations (for training)
# =============================================================================


def to_projection_torch(Phi: torch.Tensor) -> torch.Tensor:
    """
    Compute projection matrix from orthonormal basis (PyTorch).

    Args:
        Phi: Orthonormal tensor of shape (..., n, k)

    Returns:
        Projection matrix P = Φ Φ^T of shape (..., n, n)
    """
    return torch.matmul(Phi, Phi.transpose(-2, -1))


def grassmann_distance_torch(
    Phi_0: torch.Tensor,
    Phi_1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute geodesic distance on Grassmannian (PyTorch, differentiable).

    Args:
        Phi_0: First orthonormal basis (..., n, k)
        Phi_1: Second orthonormal basis (..., n, k)

    Returns:
        Grassmann distance (scalar or batch)
    """
    # M = Φ_0^T Φ_1
    M = torch.matmul(Phi_0.transpose(-2, -1), Phi_1)

    # SVD
    _, s, _ = torch.linalg.svd(M)

    # Clamp for stability
    s = torch.clamp(s, -1.0 + 1e-7, 1.0 - 1e-7)

    # Principal angles
    theta = torch.acos(s)

    # Geodesic distance
    return torch.linalg.norm(theta, dim=-1)


def projection_frobenius_torch(
    Phi_0: torch.Tensor,
    Phi_1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ||P_0 - P_1||_F^2 efficiently (PyTorch, differentiable).

    Uses: ||P_0 - P_1||_F^2 = 2k - 2||Φ_0^T Φ_1||_F^2

    Args:
        Phi_0: First orthonormal basis (..., n, k)
        Phi_1: Second orthonormal basis (..., n, k)

    Returns:
        ||P_0 - P_1||_F^2 (squared for numerical stability in gradients)
    """
    k = Phi_0.shape[-1]
    M = torch.matmul(Phi_0.transpose(-2, -1), Phi_1)
    M_norm_sq = torch.sum(M ** 2, dim=(-2, -1))
    return 2 * k - 2 * M_norm_sq


def grassmann_loss(
    Phi_pred: torch.Tensor,
    Phi_target: torch.Tensor,
) -> torch.Tensor:
    """
    Grassmann-invariant loss for training.

    Computes ||P_pred - P_target||_F^2 efficiently.
    Invariant to sign flips, permutations, rotations.

    Args:
        Phi_pred: Predicted orthonormal basis (..., n, k)
        Phi_target: Target orthonormal basis (..., n, k)

    Returns:
        Loss value (mean over batch if batched)
    """
    loss = projection_frobenius_torch(Phi_pred, Phi_target)

    # Mean over batch dimensions if present
    if loss.dim() > 0:
        return loss.mean()
    return loss


# =============================================================================
# Visualization helper
# =============================================================================


def compute_distance_matrix(
    Phi_sequence: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """
    Compute pairwise Grassmann distance matrix for a sequence.

    Args:
        Phi_sequence: List of orthonormal bases over time

    Returns:
        Distance matrix D where D[i,j] = grassmann_distance(Φ_i, Φ_j)
    """
    n = len(Phi_sequence)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = grassmann_distance(Phi_sequence[i], Phi_sequence[j])
            D[i, j] = d
            D[j, i] = d

    return D
