"""
Neural velocity field for temporal dynamics on the Stiefel manifold.

This module implements the "outer flow" - a learned model that predicts
how spectral representations evolve over time. The velocity field operates
directly on the Stiefel manifold, respecting its geometric structure.

Key properties:
- Predictions lie in the tangent space at each point
- No alignment machinery needed at inference time
- Can extrapolate into the future
"""

from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_spectral_flow.stiefel import StiefelManifold


class StiefelVelocityField(nn.Module):
    """
    Neural network that predicts tangent vectors on the Stiefel manifold.

    The network takes a point Phi on St(N, k) and time t, and outputs
    a tangent vector in T_Phi St(N, k).

    Architecture ensures output lies in tangent space via projection.
    """

    def __init__(
        self,
        k: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        time_embed_dim: int = 64,
        activation: str = "gelu",
        use_spectral_features: bool = True,
        use_attention: bool = True,
    ):
        """
        Initialize velocity field network.

        Args:
            k: Spectral dimension (number of columns in Phi)
            hidden_dims: Hidden layer dimensions
            time_embed_dim: Dimension of time embedding
            activation: Activation function ('relu', 'gelu', 'silu')
            use_spectral_features: Extract spectral features from input
            use_attention: Use self-attention over spectral dimensions
        """
        super().__init__()

        self.k = k
        self.time_embed_dim = time_embed_dim
        self.use_spectral_features = use_spectral_features
        self.use_attention = use_attention

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Activation
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        self.activation_fn = activations.get(activation, nn.GELU)

        # Feature extraction (operates on Phi^T Phi and diagonal features)
        # Input: k*(k+1)/2 from upper triangle of Gram + k diagonal
        spectral_input_dim = k * (k + 1) // 2 + k
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_input_dim, hidden_dims[0]),
            self.activation_fn(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Self-attention over spectral dimensions
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0],
                num_heads=4,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dims[0])

        # Main processing network
        in_dim = hidden_dims[0] + time_embed_dim
        layers = []
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.activation_fn())
            layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.main_net = nn.Sequential(*layers)

        # Output head: produces k*k matrix that will be reshaped and projected
        self.velocity_head = nn.Linear(hidden_dims[-1], k * k)

    def forward(
        self,
        Phi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity field at (Phi, t).

        Args:
            Phi: Spectral embedding of shape (batch, N, k) or (N, k)
            t: Time of shape (batch,) or scalar

        Returns:
            Velocity in tangent space, same shape as Phi
        """
        # Handle unbatched input
        if Phi.dim() == 2:
            Phi = Phi.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, N, k = Phi.shape

        # Ensure t has correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.shape[0] == 1:
            t = t.expand(batch_size)

        # Time embedding
        t_embed = self.time_embed(t)  # (batch, time_embed_dim)

        # Extract spectral features
        features = self._extract_spectral_features(Phi)  # (batch, spectral_input_dim)
        features = self.spectral_encoder(features)  # (batch, hidden_dim)

        # Optional attention
        if self.use_attention:
            features = features.unsqueeze(1)  # (batch, 1, hidden_dim)
            attn_out, _ = self.attention(features, features, features)
            features = self.attn_norm(features + attn_out)
            features = features.squeeze(1)  # (batch, hidden_dim)

        # Concatenate with time
        combined = torch.cat([features, t_embed], dim=-1)

        # Main network
        hidden = self.main_net(combined)  # (batch, hidden_dim)

        # Predict velocity coefficients
        velocity_coeffs = self.velocity_head(hidden)  # (batch, k*k)
        velocity_coeffs = velocity_coeffs.view(batch_size, k, k)

        # Project to tangent space
        # V = Phi @ A where A is skew-symmetric
        A = velocity_coeffs - velocity_coeffs.transpose(-2, -1)  # Skew-symmetric
        A = A / 2.0

        # Also add component orthogonal to Phi
        # Full tangent space: V = Phi @ A + (I - Phi @ Phi^T) @ B
        # For simplicity, we use the Phi @ A parameterization which is a valid subspace

        # Actually, full tangent space parameterization:
        # Generate orthogonal complement contribution
        V = Phi @ A  # (batch, N, k)

        if squeeze_output:
            V = V.squeeze(0)

        return V

    def _extract_spectral_features(self, Phi: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation-invariant spectral features from Phi.

        Args:
            Phi: Shape (batch, N, k)

        Returns:
            Features of shape (batch, feature_dim)
        """
        batch_size, N, k = Phi.shape

        # Gram matrix (k x k)
        gram = torch.bmm(Phi.transpose(-2, -1), Phi)  # (batch, k, k)

        # Extract upper triangle (including diagonal)
        indices = torch.triu_indices(k, k)
        gram_upper = gram[:, indices[0], indices[1]]  # (batch, k*(k+1)/2)

        # Column norms (should be ~1 for Stiefel points)
        col_norms = torch.sqrt(torch.sum(Phi ** 2, dim=1))  # (batch, k)

        features = torch.cat([gram_upper, col_norms], dim=-1)

        return features


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (as in transformers/diffusion models)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time values.

        Args:
            t: Time values of shape (batch,)

        Returns:
            Embeddings of shape (batch, dim)
        """
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


class SpectralFlowModel(nn.Module):
    """
    Complete spectral flow model for temporal dynamics.

    Combines the velocity field with integration methods and
    manifold-aware operations.
    """

    def __init__(
        self,
        k: int,
        velocity_field: Optional[StiefelVelocityField] = None,
        integration_method: Literal["euler", "rk4", "midpoint"] = "euler",
        **velocity_kwargs,
    ):
        """
        Initialize flow model.

        Args:
            k: Spectral dimension
            velocity_field: Velocity network (created if None)
            integration_method: ODE integration method
            **velocity_kwargs: Arguments for velocity field if created
        """
        super().__init__()

        self.k = k
        self.integration_method = integration_method

        if velocity_field is None:
            velocity_field = StiefelVelocityField(k, **velocity_kwargs)
        self.velocity_field = velocity_field

    def forward(
        self,
        Phi: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Integrate flow from t_start to t_end.

        Args:
            Phi: Initial spectral embedding (batch, N, k) or (N, k)
            t_start: Starting time
            t_end: Ending time
            n_steps: Number of integration steps

        Returns:
            Final spectral embedding at t_end
        """
        if self.integration_method == "euler":
            return self._integrate_euler(Phi, t_start, t_end, n_steps)
        elif self.integration_method == "rk4":
            return self._integrate_rk4(Phi, t_start, t_end, n_steps)
        elif self.integration_method == "midpoint":
            return self._integrate_midpoint(Phi, t_start, t_end, n_steps)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")

    def _integrate_euler(
        self,
        Phi: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Euler integration with retraction."""
        dt = (t_end - t_start) / n_steps
        t = t_start.clone()

        for _ in range(n_steps):
            v = self.velocity_field(Phi, t)
            Phi = self._retract(Phi, v, dt)
            t = t + dt

        return Phi

    def _integrate_midpoint(
        self,
        Phi: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Midpoint integration."""
        dt = (t_end - t_start) / n_steps
        t = t_start.clone()

        for _ in range(n_steps):
            # Half step
            v1 = self.velocity_field(Phi, t)
            Phi_mid = self._retract(Phi, v1, dt / 2)

            # Full step with midpoint velocity
            v2 = self.velocity_field(Phi_mid, t + dt / 2)
            Phi = self._retract(Phi, v2, dt)

            t = t + dt

        return Phi

    def _integrate_rk4(
        self,
        Phi: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """RK4 integration (approximate on manifold)."""
        dt = (t_end - t_start) / n_steps
        t = t_start.clone()

        for _ in range(n_steps):
            k1 = self.velocity_field(Phi, t)

            Phi_2 = self._retract(Phi, k1, dt / 2)
            k2 = self.velocity_field(Phi_2, t + dt / 2)

            Phi_3 = self._retract(Phi, k2, dt / 2)
            k3 = self.velocity_field(Phi_3, t + dt / 2)

            Phi_4 = self._retract(Phi, k3, dt)
            k4 = self.velocity_field(Phi_4, t + dt)

            # Weighted average (approximate on manifold)
            v_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Phi = self._retract(Phi, v_avg, dt)

            t = t + dt

        return Phi

    def _retract(
        self,
        Phi: torch.Tensor,
        V: torch.Tensor,
        dt: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """
        Retract from Phi along V onto the Stiefel manifold.

        Uses QR retraction for efficiency.
        """
        if isinstance(dt, torch.Tensor):
            dt = dt.item() if dt.numel() == 1 else dt

        Y = Phi + dt * V

        # QR retraction
        if Y.dim() == 2:
            Q, R = torch.linalg.qr(Y)
            # Ensure consistent sign
            signs = torch.sign(torch.diag(R))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Q = Q * signs.unsqueeze(0)
        else:
            # Batched QR
            Q, R = torch.linalg.qr(Y)
            diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.sign(diag_R)
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Q = Q * signs.unsqueeze(-2)

        return Q

    def predict_velocity(
        self,
        Phi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at given state and time.

        Args:
            Phi: Spectral embedding
            t: Time

        Returns:
            Predicted tangent vector
        """
        return self.velocity_field(Phi, t)

    def trajectory(
        self,
        Phi: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        n_steps: int = 10,
    ) -> list[torch.Tensor]:
        """
        Generate full trajectory from t_start to t_end.

        Args:
            Phi: Initial embedding
            t_start: Starting time
            t_end: Ending time
            n_steps: Number of steps

        Returns:
            List of embeddings at each time step
        """
        trajectory = [Phi.clone()]

        dt = (t_end - t_start) / n_steps
        t = t_start.clone()

        for _ in range(n_steps):
            v = self.velocity_field(Phi, t)
            Phi = self._retract(Phi, v, dt)
            trajectory.append(Phi.clone())
            t = t + dt

        return trajectory


def numpy_to_torch(
    arr: NDArray[np.floating],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    tensor = torch.from_numpy(arr.astype(np.float32))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_to_numpy(tensor: torch.Tensor) -> NDArray[np.floating]:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()
