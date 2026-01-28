"""
Riemannian Flow Matching on the Stiefel manifold.

Implements Temporal Spectral Geodesic Flow Matching (TS-GFM):
- Conditional velocity fields that take (Φ_current, t | Φ_target, λ_target)
- Geodesic reference bridge via Exp/Log maps
- Manifold-aware integration with QR retraction

Core principle: Let the manifold geometry define the reference flow,
train a neural vector field to match intrinsic geodesic velocities.
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from beartype import beartype
from jaxtyping import Float, jaxtyped

from temporal_spectral_flow.stiefel import StiefelManifold


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (Fourier features).

    Maps scalar time t ∈ [0, 1] to a rich embedding space.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time values.

        Args:
            t: Time values of shape (batch,) or scalar

        Returns:
            Embeddings of shape (batch, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) *
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))

        return embedding


# =============================================================================
# Network Building Blocks
# =============================================================================


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ResidualMLP(nn.Module):
    """MLP with residual blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_blocks: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# =============================================================================
# Conditional Velocity Fields
# =============================================================================


class ConditionalEigenvalueVelocityField(nn.Module):
    """
    Predicts velocity for eigenvalues in R^k.

    Conditional on target eigenvalues λ_target.

    Input: (λ_current, t | λ_target)
    Output: v_λ ∈ R^k
    """

    def __init__(
        self,
        k: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.k = k
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: λ_current (k) + λ_target (k) + time_embed
        input_dim = 2 * k + time_embed_dim

        self.net = ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=k,
            n_blocks=n_blocks,
        )

    def forward(
        self,
        lambda_current: torch.Tensor,
        lambda_target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict eigenvalue velocity.

        Args:
            lambda_current: Current eigenvalues (batch, k) or (k,)
            lambda_target: Target eigenvalues (batch, k) or (k,)
            t: Time (batch,) or scalar

        Returns:
            Velocity v_λ of shape matching lambda_current
        """
        # Handle unbatched input
        squeeze_output = False
        if lambda_current.dim() == 1:
            lambda_current = lambda_current.unsqueeze(0)
            lambda_target = lambda_target.unsqueeze(0)
            squeeze_output = True

        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and lambda_current.shape[0] > 1:
            t = t.expand(lambda_current.shape[0])

        # Time embedding
        t_embed = self.time_embed(t)

        # Concatenate inputs
        x = torch.cat([lambda_current, lambda_target, t_embed], dim=-1)

        # Predict velocity
        v = self.net(x)

        if squeeze_output:
            v = v.squeeze(0)

        return v


class ConditionalStiefelVelocityField(nn.Module):
    """
    Predicts tangent velocity on the Stiefel manifold.

    Conditional on target frame (Φ_target, λ_target).

    Input: (Φ_current, t | Φ_target, λ_target)
    Output: V ∈ T_{Φ_current} St(n, k)

    The output is guaranteed to lie in the tangent space via projection.
    """

    def __init__(
        self,
        n: int,
        k: int,
        hidden_dim: int = 1024,
        n_blocks: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.n = n
        self.k = k
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: Φ_current (n*k) + Φ_target (n*k) + λ_target (k) + time_embed
        input_dim = 2 * n * k + k + time_embed_dim

        self.net = ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n * k,
            n_blocks=n_blocks,
        )

    def forward(
        self,
        Phi_current: torch.Tensor,
        Phi_target: torch.Tensor,
        lambda_target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict tangent velocity at Φ_current toward Φ_target.

        Args:
            Phi_current: Current frame (batch, n, k) or (n, k)
            Phi_target: Target frame (batch, n, k) or (n, k)
            lambda_target: Target eigenvalues (batch, k) or (k,)
            t: Time (batch,) or scalar

        Returns:
            Tangent velocity V ∈ T_{Φ_current} St, same shape as Phi_current
        """
        # Handle unbatched input
        squeeze_output = False
        if Phi_current.dim() == 2:
            Phi_current = Phi_current.unsqueeze(0)
            Phi_target = Phi_target.unsqueeze(0)
            lambda_target = lambda_target.unsqueeze(0)
            squeeze_output = True

        batch_size, n, k = Phi_current.shape

        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)

        # Time embedding
        t_embed = self.time_embed(t)

        # Flatten and concatenate
        Phi_current_flat = rearrange(Phi_current, 'b n k -> b (n k)')
        Phi_target_flat = rearrange(Phi_target, 'b n k -> b (n k)')

        x = torch.cat([
            Phi_current_flat,
            Phi_target_flat,
            lambda_target,
            t_embed
        ], dim=-1)

        # Predict raw output
        Z = self.net(x)
        Z = rearrange(Z, 'b (n k) -> b n k', n=n, k=k)

        # Project to tangent space at Phi_current
        V = self._project_to_tangent(Z, Phi_current)

        if squeeze_output:
            V = V.squeeze(0)

        return V

    def _project_to_tangent(
        self,
        Z: torch.Tensor,
        Phi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project Z to tangent space at Phi.

        T_Φ St = {V : Φ^T V + V^T Φ = 0}
        V = Z - Φ @ sym(Φ^T @ Z)
        """
        # PhiTZ: (batch, k, k)
        PhiTZ = einsum(Phi, Z, 'b n k, b n l -> b k l')

        # Symmetric part
        sym_PhiTZ = (PhiTZ + PhiTZ.transpose(-2, -1)) / 2

        # Project: V = Z - Φ @ sym(Φ^T Z)
        V = Z - einsum(Phi, sym_PhiTZ, 'b n k, b k l -> b n l')

        return V


# =============================================================================
# Combined Flow Model
# =============================================================================


class GeodesicFlowModel(nn.Module):
    """
    Geodesic Flow Model for temporal spectral dynamics.

    Combines conditional velocity fields for eigenvectors and eigenvalues.
    Integrates using manifold-aware retraction.

    Training: Match intrinsic geodesic velocities (Riemannian Flow Matching)
    Inference: Integrate learned dynamics from start to target
    """

    def __init__(
        self,
        n: int,
        k: int,
        hidden_dim_phi: int = 1024,
        hidden_dim_lambda: int = 256,
        n_blocks_phi: int = 4,
        n_blocks_lambda: int = 3,
        time_embed_dim: int = 64,
    ):
        """
        Initialize flow model.

        Args:
            n: Ambient dimension (number of nodes)
            k: Spectral dimension (number of eigenvectors)
            hidden_dim_phi: Hidden dimension for eigenvector network
            hidden_dim_lambda: Hidden dimension for eigenvalue network
            n_blocks_phi: Number of residual blocks for eigenvector network
            n_blocks_lambda: Number of residual blocks for eigenvalue network
            time_embed_dim: Dimension of time embedding
        """
        super().__init__()

        self.n = n
        self.k = k

        self.phi_velocity = ConditionalStiefelVelocityField(
            n=n,
            k=k,
            hidden_dim=hidden_dim_phi,
            n_blocks=n_blocks_phi,
            time_embed_dim=time_embed_dim,
        )

        self.lambda_velocity = ConditionalEigenvalueVelocityField(
            k=k,
            hidden_dim=hidden_dim_lambda,
            n_blocks=n_blocks_lambda,
            time_embed_dim=time_embed_dim,
        )

    def velocity(
        self,
        Phi_current: torch.Tensor,
        lambda_current: torch.Tensor,
        Phi_target: torch.Tensor,
        lambda_target: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict velocity at current state toward target.

        Args:
            Phi_current: Current eigenvectors (batch, n, k) or (n, k)
            lambda_current: Current eigenvalues (batch, k) or (k,)
            Phi_target: Target eigenvectors
            lambda_target: Target eigenvalues
            t: Time

        Returns:
            (v_Phi, v_lambda): Velocities for eigenvectors and eigenvalues
        """
        v_Phi = self.phi_velocity(Phi_current, Phi_target, lambda_target, t)
        v_lambda = self.lambda_velocity(lambda_current, lambda_target, t)

        return v_Phi, v_lambda

    def integrate(
        self,
        Phi_start: torch.Tensor,
        lambda_start: torch.Tensor,
        Phi_target: torch.Tensor,
        lambda_target: torch.Tensor,
        t_end: float = 1.0,
        n_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, list, list]
    ]:
        """
        Integrate flow from start toward target.

        Args:
            Phi_start: Starting eigenvectors
            lambda_start: Starting eigenvalues
            Phi_target: Target eigenvectors (conditioning)
            lambda_target: Target eigenvalues (conditioning)
            t_end: Integration end time
            n_steps: Number of integration steps
            return_trajectory: If True, return full trajectory

        Returns:
            (Phi_end, lambda_end) or (Phi_end, lambda_end, Phi_traj, lambda_traj)
        """
        dt = t_end / n_steps
        Phi = Phi_start.clone()
        lambda_ = lambda_start.clone()

        if return_trajectory:
            Phi_traj = [Phi.clone()]
            lambda_traj = [lambda_.clone()]

        for step in range(n_steps):
            t = torch.tensor(step * dt / t_end, device=Phi.device, dtype=Phi.dtype)

            # Predict velocity
            v_Phi, v_lambda = self.velocity(
                Phi, lambda_, Phi_target, lambda_target, t
            )

            # Update eigenvalues (Euclidean)
            lambda_ = lambda_ + dt * v_lambda

            # Update eigenvectors (Stiefel retraction)
            Phi = self._stiefel_retract(Phi, v_Phi, dt)

            if return_trajectory:
                Phi_traj.append(Phi.clone())
                lambda_traj.append(lambda_.clone())

        if return_trajectory:
            return Phi, lambda_, Phi_traj, lambda_traj

        return Phi, lambda_

    def _stiefel_retract(
        self,
        Phi: torch.Tensor,
        V: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        QR retraction on Stiefel manifold.

        Args:
            Phi: Current point (batch, n, k) or (n, k)
            V: Tangent vector
            dt: Step size

        Returns:
            Retracted point on Stiefel
        """
        Y = Phi + dt * V

        # Handle batched vs unbatched
        if Y.dim() == 2:
            Q, R = torch.linalg.qr(Y)
            # Sign correction for deterministic behavior
            signs = torch.sign(torch.diag(R))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Q = Q * signs.unsqueeze(0)
        else:
            Q, R = torch.linalg.qr(Y)
            diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.sign(diag_R)
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Q = Q * signs.unsqueeze(-2)

        return Q


# =============================================================================
# Stiefel Sampling
# =============================================================================


def uniform_stiefel(
    n: int,
    k: int,
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample uniformly from the Stiefel manifold St(n, k).

    Args:
        n: Ambient dimension
        k: Subspace dimension
        batch_size: Number of samples
        device: Torch device
        dtype: Data type

    Returns:
        Tensor of shape (batch_size, n, k) if batch_size > 1, else (n, k)
    """
    # Sample Gaussian matrix
    Z = torch.randn(batch_size, n, k, device=device, dtype=dtype)

    # QR factorization
    Q, R = torch.linalg.qr(Z)

    # Sign correction
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    signs = torch.sign(diag_R)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Q = Q * signs.unsqueeze(-2)

    if batch_size == 1:
        return Q.squeeze(0)

    return Q


# =============================================================================
# Geodesic Operations (for training reference bridge)
# =============================================================================


def stiefel_log_qr(
    Phi_0: torch.Tensor,
    Phi_1: torch.Tensor,
) -> torch.Tensor:
    """
    QR-retraction inverse: tangent vector V such that retract(Phi_0, V, 1) ≈ Phi_1.

    This is consistent with the QR retraction used in integration.

    Args:
        Phi_0: Base point (batch, n, k) or (n, k)
        Phi_1: Target point

    Returns:
        Tangent vector V at Phi_0
    """
    # A = Phi_0^T @ Phi_1
    if Phi_0.dim() == 2:
        A = Phi_0.T @ Phi_1
        Qa, Ra = torch.linalg.qr(A)
        # Sign correction
        d = torch.sign(torch.diag(Ra))
        d = torch.where(d == 0, torch.ones_like(d), d)
        Ra = d.unsqueeze(0) * Ra
        V = (Phi_1 @ Ra) - Phi_0
    else:
        A = einsum(Phi_0, Phi_1, 'b n k, b n l -> b k l')
        Qa, Ra = torch.linalg.qr(A)
        diag_Ra = torch.diagonal(Ra, dim1=-2, dim2=-1)
        d = torch.sign(diag_Ra)
        d = torch.where(d == 0, torch.ones_like(d), d)
        Ra = d.unsqueeze(-2) * Ra
        V = einsum(Phi_1, Ra, 'b n k, b k l -> b n l') - Phi_0

    return V


def stiefel_geodesic_qr(
    Phi_0: torch.Tensor,
    Phi_1: torch.Tensor,
    s: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Geodesic interpolation using QR retraction.

    Phi_s = retract(Phi_0, s * log(Phi_0, Phi_1))

    Args:
        Phi_0: Start point
        Phi_1: End point
        s: Interpolation parameter in [0, 1]

    Returns:
        Interpolated point Phi_s
    """
    if isinstance(s, float):
        if s <= 0:
            return Phi_0.clone()
        if s >= 1:
            return Phi_1.clone()

    V = stiefel_log_qr(Phi_0, Phi_1)

    # Retract with step s
    Y = Phi_0 + s * V

    if Y.dim() == 2:
        Q, R = torch.linalg.qr(Y)
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
    else:
        Q, R = torch.linalg.qr(Y)
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        signs = torch.sign(diag_R)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(-2)

    return Q
