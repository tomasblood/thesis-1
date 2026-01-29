"""
Grassmann-invariant loss functions for eigenvector supervision.

These losses are invariant to sign flips, permutations, and eigenvalue crossings,
making them suitable for training on raw spectral data without alignment.

Key loss: Principal angle loss based on the angles between subspaces spanned
by predicted and target eigenvectors.
"""

import torch
from einops import einsum
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def principal_angle_loss(
    Phi_pred: Float[torch.Tensor, "batch n k"],
    Phi_true: Float[torch.Tensor, "batch n k"],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Grassmann-invariant loss using principal angles.

    The principal angles between two subspaces measure the geometric distance
    on the Grassmann manifold. This loss is invariant to:
    - Sign flips of individual eigenvectors
    - Permutations of eigenvectors
    - Orthogonal rotations within the span

    Args:
        Phi_pred: Predicted eigenvectors (batch, N, k)
        Phi_true: Target eigenvectors (batch, N, k)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss (or per-batch if reduction='none')
    """
    # Compute M = Phi_pred^T @ Phi_true -> (batch, k, k)
    M = einsum(Phi_pred, Phi_true, 'b n k, b n l -> b k l')

    # Singular values of M give cos(principal angles)
    # Using SVD to get singular values
    s = torch.linalg.svdvals(M)  # (batch, min(k, k)) = (batch, k)

    # Clamp to valid range for arccos (numerical stability)
    s = torch.clamp(s, -1.0 + 1e-7, 1.0 - 1e-7)

    # Principal angles
    angles = torch.acos(s)  # (batch, k)

    # Loss: sum of squared angles
    angle_loss = torch.sum(angles ** 2, dim=-1)  # (batch,)

    if reduction == "mean":
        return torch.mean(angle_loss)
    elif reduction == "sum":
        return torch.sum(angle_loss)
    else:
        return angle_loss


@jaxtyped(typechecker=beartype)
def projection_loss(
    Phi_pred: Float[torch.Tensor, "batch n k"],
    Phi_true: Float[torch.Tensor, "batch n k"],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Grassmann-invariant loss using projection matrices.

    Measures ||P_pred - P_true||_F^2 where P = Phi @ Phi^T is the
    projection onto the subspace. This is equivalent to the geodesic
    distance on Grassmann manifold.

    Args:
        Phi_pred: Predicted eigenvectors (batch, N, k)
        Phi_true: Target eigenvectors (batch, N, k)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    # P_pred = Phi_pred @ Phi_pred^T, P_true = Phi_true @ Phi_true^T
    # ||P_pred - P_true||_F^2 = 2k - 2 * ||Phi_pred^T @ Phi_true||_F^2

    # Compute Phi_pred^T @ Phi_true
    M = einsum(Phi_pred, Phi_true, 'b n k, b n l -> b k l')

    # Frobenius norm squared of M
    frob_sq = torch.sum(M ** 2, dim=(-2, -1))  # (batch,)

    k = Phi_pred.shape[-1]
    loss = 2 * k - 2 * frob_sq

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        return loss


@jaxtyped(typechecker=beartype)
def chordal_grassmann_distance(
    Phi_pred: Float[torch.Tensor, "batch n k"],
    Phi_true: Float[torch.Tensor, "batch n k"],
) -> torch.Tensor:
    """
    Compute chordal distance on Grassmann manifold.

    d_chordal^2 = k - ||Phi_pred^T @ Phi_true||_F^2 / 2

    Equivalent to projection_loss / 2.

    Args:
        Phi_pred: Predicted eigenvectors (batch, N, k)
        Phi_true: Target eigenvectors (batch, N, k)

    Returns:
        Squared chordal distance per batch element
    """
    M = einsum(Phi_pred, Phi_true, 'b n k, b n l -> b k l')
    frob_sq = torch.sum(M ** 2, dim=(-2, -1))
    k = Phi_pred.shape[-1]
    return k - frob_sq


@jaxtyped(typechecker=beartype)
def eigenvalue_mse_loss(
    lambda_pred: Float[torch.Tensor, "batch k"],
    lambda_true: Float[torch.Tensor, "batch k"],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE loss for eigenvalues.

    Simple Euclidean loss in R^k for eigenvalue prediction.

    Args:
        lambda_pred: Predicted eigenvalues (batch, k)
        lambda_true: Target eigenvalues (batch, k)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    mse = torch.sum((lambda_pred - lambda_true) ** 2, dim=-1)  # (batch,)

    if reduction == "mean":
        return torch.mean(mse)
    elif reduction == "sum":
        return torch.sum(mse)
    else:
        return mse


@jaxtyped(typechecker=beartype)
def velocity_energy_regularization(
    v_lambda: Float[torch.Tensor, "batch k"],
    v_Phi: Float[torch.Tensor, "batch n k"],
    gamma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute energy regularization on velocity field.

    Penalizes large velocities for smoothness.

    L_energy = ||v_lambda||^2 + gamma * ||v_Phi||^2

    Args:
        v_lambda: Eigenvalue velocity (batch, k)
        v_Phi: Eigenvector velocity (batch, N, k)
        gamma: Weight for eigenvector velocity term
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Energy regularization loss
    """
    energy_lambda = torch.sum(v_lambda ** 2, dim=-1)  # (batch,)
    energy_Phi = torch.sum(v_Phi ** 2, dim=(-2, -1))  # (batch,)

    energy = energy_lambda + gamma * energy_Phi

    if reduction == "mean":
        return torch.mean(energy)
    elif reduction == "sum":
        return torch.sum(energy)
    else:
        return energy
