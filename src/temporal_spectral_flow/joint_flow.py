"""
Joint flow model for eigenvalues and eigenvectors.

Learns dynamics on the product space R^k x V_k(R^n), where eigenvalues
evolve in Euclidean space and eigenvectors evolve on the Stiefel manifold.

The flow is conditioned on eigenvalues - eigenvector dynamics depend on
the geometric scales present.
"""

from typing import Optional, Tuple, Literal, List
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.flow import SinusoidalTimeEmbedding


class EigenvalueVelocityField(nn.Module):
    """
    Neural network for eigenvalue dynamics in R^k.

    Predicts d(lambda)/dt given current eigenvalues and time.
    """

    def __init__(
        self,
        k: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        time_embed_dim: int = 32,
    ):
        """
        Initialize eigenvalue velocity field.

        Args:
            k: Number of eigenvalues
            hidden_dims: Hidden layer dimensions
            time_embed_dim: Dimension of time embedding
        """
        super().__init__()

        self.k = k
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: eigenvalues (k) + time embedding
        in_dim = k + time_embed_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, k))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        lambda_: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict eigenvalue velocity.

        Args:
            lambda_: Eigenvalues (batch, k) or (k,)
            t: Time (batch,) or scalar

        Returns:
            Velocity in R^k
        """
        if lambda_.dim() == 1:
            lambda_ = lambda_.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size = lambda_.shape[0]

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        t_embed = self.time_embed(t)

        x = torch.cat([lambda_, t_embed], dim=-1)
        v = self.net(x)

        if squeeze:
            v = v.squeeze(0)

        return v


class EigenvectorVelocityField(nn.Module):
    """
    Neural network for eigenvector dynamics on Stiefel manifold.

    Predicts d(Phi)/dt given current eigenvectors, eigenvalues, and time.
    Output is projected to tangent space of Stiefel manifold.

    Key: Conditions on eigenvalues so eigenvector dynamics know
    which geometric scales are present.
    """

    def __init__(
        self,
        k: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        time_embed_dim: int = 32,
        lambda_embed_dim: int = 32,
    ):
        """
        Initialize eigenvector velocity field.

        Args:
            k: Number of eigenvectors (columns of Phi)
            hidden_dims: Hidden layer dimensions
            time_embed_dim: Dimension of time embedding
            lambda_embed_dim: Dimension of eigenvalue embedding
        """
        super().__init__()

        self.k = k
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Eigenvalue embedding
        self.lambda_embed = nn.Sequential(
            nn.Linear(k, lambda_embed_dim),
            nn.GELU(),
            nn.Linear(lambda_embed_dim, lambda_embed_dim),
        )

        # Eigenvector features: use Phi^T Phi structure (always identity for Stiefel)
        # and row statistics of Phi
        # Input features: k eigenvalue embed + time embed + k*(k-1)/2 off-diagonal correlations
        feature_dim = lambda_embed_dim + time_embed_dim + k

        layers = []
        in_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Output: k x k matrix that parameterizes tangent vector
        self.output_head = nn.Linear(hidden_dims[-1], k * k)

    def forward(
        self,
        Phi: torch.Tensor,
        lambda_: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict eigenvector velocity (tangent to Stiefel).

        Args:
            Phi: Eigenvectors (batch, N, k) or (N, k)
            lambda_: Eigenvalues (batch, k) or (k,)
            t: Time (batch,) or scalar

        Returns:
            Tangent vector same shape as Phi
        """
        if Phi.dim() == 2:
            Phi = Phi.unsqueeze(0)
            lambda_ = lambda_.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, N, k = Phi.shape

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Embeddings
        t_embed = self.time_embed(t)  # (batch, time_embed_dim)
        lambda_embed = self.lambda_embed(lambda_)  # (batch, lambda_embed_dim)

        # Eigenvector statistics (invariant to node ordering within rows)
        # Use column norms (should be ~1) and column means
        col_norms = torch.sqrt(torch.sum(Phi ** 2, dim=1))  # (batch, k)

        # Combine features
        features = torch.cat([t_embed, lambda_embed, col_norms], dim=-1)

        # Process
        hidden = self.feature_net(features)  # (batch, hidden_dim)

        # Output k x k coefficients
        coeffs = self.output_head(hidden)  # (batch, k*k)
        coeffs = coeffs.view(batch_size, k, k)

        # Construct tangent vector: V = Phi @ A where A is skew-symmetric
        A = (coeffs - coeffs.transpose(-2, -1)) / 2
        V = torch.bmm(Phi, A)  # (batch, N, k)

        if squeeze:
            V = V.squeeze(0)

        return V


class JointSpectralFlow(nn.Module):
    """
    Joint flow model for (lambda, Phi) on R^k x V_k(R^n).

    Learns to transport spectral representations through time,
    respecting the Stiefel geometry for eigenvectors.
    """

    def __init__(
        self,
        k: int,
        eigenvalue_hidden: Tuple[int, ...] = (128, 128),
        eigenvector_hidden: Tuple[int, ...] = (256, 256),
        integration_method: Literal["euler", "midpoint", "rk4"] = "euler",
    ):
        """
        Initialize joint spectral flow model.

        Args:
            k: Number of eigenmodes
            eigenvalue_hidden: Hidden dimensions for eigenvalue network
            eigenvector_hidden: Hidden dimensions for eigenvector network
            integration_method: Integration scheme for flow
        """
        super().__init__()

        self.k = k
        self.integration_method = integration_method

        self.eigenvalue_field = EigenvalueVelocityField(k, eigenvalue_hidden)
        self.eigenvector_field = EigenvectorVelocityField(k, eigenvector_hidden)

    def velocity(
        self,
        lambda_: torch.Tensor,
        Phi: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint velocity.

        Args:
            lambda_: Eigenvalues
            Phi: Eigenvectors
            t: Time

        Returns:
            v_lambda: Eigenvalue velocity
            v_Phi: Eigenvector velocity (tangent to Stiefel)
        """
        v_lambda = self.eigenvalue_field(lambda_, t)
        v_Phi = self.eigenvector_field(Phi, lambda_, t)

        return v_lambda, v_Phi

    def step_euler(
        self,
        lambda_: torch.Tensor,
        Phi: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Euler step with Stiefel retraction."""
        v_lambda, v_Phi = self.velocity(lambda_, Phi, t)

        lambda_new = lambda_ + dt * v_lambda
        Phi_new = self._stiefel_retract(Phi, v_Phi, dt)

        return lambda_new, Phi_new

    def step_midpoint(
        self,
        lambda_: torch.Tensor,
        Phi: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Midpoint integration step."""
        # Half step
        v_lambda1, v_Phi1 = self.velocity(lambda_, Phi, t)
        lambda_mid = lambda_ + (dt / 2) * v_lambda1
        Phi_mid = self._stiefel_retract(Phi, v_Phi1, dt / 2)

        # Full step with midpoint velocity
        v_lambda2, v_Phi2 = self.velocity(lambda_mid, Phi_mid, t + dt / 2)
        lambda_new = lambda_ + dt * v_lambda2
        Phi_new = self._stiefel_retract(Phi, v_Phi2, dt)

        return lambda_new, Phi_new

    def integrate(
        self,
        lambda_start: torch.Tensor,
        Phi_start: torch.Tensor,
        t_start: float,
        t_end: float,
        n_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate flow from t_start to t_end.

        Args:
            lambda_start: Initial eigenvalues
            Phi_start: Initial eigenvectors
            t_start: Start time
            t_end: End time
            n_steps: Number of integration steps

        Returns:
            lambda_end: Final eigenvalues
            Phi_end: Final eigenvectors
        """
        dt = (t_end - t_start) / n_steps

        lambda_ = lambda_start
        Phi = Phi_start
        t = t_start

        step_fn = {
            "euler": self.step_euler,
            "midpoint": self.step_midpoint,
        }.get(self.integration_method, self.step_euler)

        for _ in range(n_steps):
            t_tensor = torch.tensor([t], device=lambda_.device, dtype=lambda_.dtype)
            lambda_, Phi = step_fn(lambda_, Phi, t_tensor, dt)
            t += dt

        return lambda_, Phi

    def trajectory(
        self,
        lambda_start: torch.Tensor,
        Phi_start: torch.Tensor,
        t_start: float,
        t_end: float,
        n_steps: int = 10,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Generate full trajectory with intermediate states.

        Args:
            lambda_start: Initial eigenvalues
            Phi_start: Initial eigenvectors
            t_start: Start time
            t_end: End time
            n_steps: Number of integration steps

        Returns:
            lambda_traj: List of eigenvalue tensors
            Phi_traj: List of eigenvector tensors
            times: List of time values
        """
        dt = (t_end - t_start) / n_steps

        lambda_ = lambda_start
        Phi = Phi_start
        t = t_start

        lambda_traj = [lambda_.clone()]
        Phi_traj = [Phi.clone()]
        times = [t]

        step_fn = {
            "euler": self.step_euler,
            "midpoint": self.step_midpoint,
        }.get(self.integration_method, self.step_euler)

        for _ in range(n_steps):
            t_tensor = torch.tensor([t], device=lambda_.device, dtype=lambda_.dtype)
            lambda_, Phi = step_fn(lambda_, Phi, t_tensor, dt)
            t += dt

            lambda_traj.append(lambda_.clone())
            Phi_traj.append(Phi.clone())
            times.append(t)

        return lambda_traj, Phi_traj, times

    def _stiefel_retract(
        self,
        Phi: torch.Tensor,
        V: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """QR retraction on Stiefel manifold."""
        Y = Phi + dt * V

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


def compute_flow_matching_loss(
    model: JointSpectralFlow,
    lambda_source: torch.Tensor,
    Phi_source: torch.Tensor,
    lambda_target: torch.Tensor,
    Phi_target: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute flow matching loss for training.

    Target velocities are computed from aligned source-target pairs.

    Args:
        model: JointSpectralFlow model
        lambda_source: Source eigenvalues (batch, k)
        Phi_source: Source eigenvectors (batch, N, k)
        lambda_target: Aligned target eigenvalues (batch, k)
        Phi_target: Aligned target eigenvectors (batch, N, k)
        t: Interpolation time (batch,), values in [0, 1]

    Returns:
        loss: Scalar loss
        metrics: Dictionary of component losses
    """
    batch_size = lambda_source.shape[0]

    # Interpolate to time t
    # Linear interpolation for eigenvalues
    lambda_t = (1 - t.unsqueeze(-1)) * lambda_source + t.unsqueeze(-1) * lambda_target

    # For eigenvectors, we should use geodesic interpolation
    # Approximate with linear + retraction for simplicity
    Phi_t_linear = (1 - t.view(-1, 1, 1)) * Phi_source + t.view(-1, 1, 1) * Phi_target

    # Retract to Stiefel
    Phi_t = []
    for i in range(batch_size):
        Q, R = torch.linalg.qr(Phi_t_linear[i])
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Phi_t.append(Q * signs.unsqueeze(0))
    Phi_t = torch.stack(Phi_t)

    # Predict velocity at interpolated point
    v_lambda_pred, v_Phi_pred = model.velocity(lambda_t, Phi_t, t)

    # Target velocity: direction to target
    # For eigenvalues: simple difference
    v_lambda_target = lambda_target - lambda_source

    # For eigenvectors: difference projected to tangent space
    v_Phi_target = Phi_target - Phi_source
    # Project to tangent space at Phi_t
    v_Phi_target = _project_to_tangent_batch(Phi_t, v_Phi_target)

    # Losses
    loss_lambda = F.mse_loss(v_lambda_pred, v_lambda_target)
    loss_Phi = F.mse_loss(v_Phi_pred, v_Phi_target)

    loss = loss_lambda + loss_Phi

    metrics = {
        "loss_total": loss.item(),
        "loss_lambda": loss_lambda.item(),
        "loss_Phi": loss_Phi.item(),
    }

    return loss, metrics


def _project_to_tangent_batch(
    Phi: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Project V to tangent space at Phi (batched)."""
    PhiTV = torch.bmm(Phi.transpose(-2, -1), V)
    sym_PhiTV = (PhiTV + PhiTV.transpose(-2, -1)) / 2
    return V - torch.bmm(Phi, sym_PhiTV)
