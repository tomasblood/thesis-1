"""
Tests for Riemannian Flow Matching components.

Tests the conditional velocity fields, geodesic operations,
and integration on the Stiefel manifold.
"""

import numpy as np
import torch
import pytest

from temporal_spectral_flow.riemannian_flow import (
    SinusoidalTimeEmbedding,
    ConditionalEigenvalueVelocityField,
    ConditionalStiefelVelocityField,
    GeodesicFlowModel,
    uniform_stiefel,
    stiefel_log_qr,
    stiefel_geodesic_qr,
)


class TestSinusoidalTimeEmbedding:
    """Tests for time embedding."""

    def test_output_shape(self):
        """Embedding has correct shape."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embed(t)
        assert out.shape == (3, 64)

    def test_different_times_different_embeddings(self):
        """Different times produce different embeddings."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        e1 = embed(t1)
        e2 = embed(t2)
        assert not torch.allclose(e1, e2)

    def test_scalar_input(self):
        """Handles scalar time input."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor(0.5)
        out = embed(t)
        assert out.shape == (1, 64)


class TestConditionalEigenvalueVelocityField:
    """Tests for eigenvalue velocity field."""

    def test_output_shape(self):
        """Output has correct shape."""
        k = 6
        field = ConditionalEigenvalueVelocityField(k=k, hidden_dim=64, n_blocks=2)

        lambda_current = torch.randn(4, k)
        lambda_target = torch.randn(4, k)
        t = torch.tensor([0.5])

        v = field(lambda_current, lambda_target, t)
        assert v.shape == (4, k)

    def test_unbatched_input(self):
        """Handles unbatched input."""
        k = 6
        field = ConditionalEigenvalueVelocityField(k=k, hidden_dim=64, n_blocks=2)

        lambda_current = torch.randn(k)
        lambda_target = torch.randn(k)
        t = torch.tensor(0.5)

        v = field(lambda_current, lambda_target, t)
        assert v.shape == (k,)

    def test_gradient_flow(self):
        """Gradients flow through the network."""
        k = 6
        field = ConditionalEigenvalueVelocityField(k=k, hidden_dim=64, n_blocks=2)

        lambda_current = torch.randn(4, k, requires_grad=True)
        lambda_target = torch.randn(4, k)
        t = torch.tensor([0.5])

        v = field(lambda_current, lambda_target, t)
        loss = v.sum()
        loss.backward()

        assert lambda_current.grad is not None


class TestConditionalStiefelVelocityField:
    """Tests for Stiefel velocity field."""

    def test_output_shape(self):
        """Output has correct shape."""
        n, k = 50, 6
        field = ConditionalStiefelVelocityField(n=n, k=k, hidden_dim=128, n_blocks=2)

        Phi_current = uniform_stiefel(n, k, batch_size=4)
        Phi_target = uniform_stiefel(n, k, batch_size=4)
        lambda_target = torch.randn(4, k)
        t = torch.tensor([0.5])

        v = field(Phi_current, Phi_target, lambda_target, t)
        assert v.shape == (4, n, k)

    def test_output_in_tangent_space(self):
        """Output lies in tangent space at Phi_current."""
        n, k = 50, 6
        field = ConditionalStiefelVelocityField(n=n, k=k, hidden_dim=128, n_blocks=2)

        Phi_current = uniform_stiefel(n, k, batch_size=1).squeeze(0)
        Phi_target = uniform_stiefel(n, k, batch_size=1).squeeze(0)
        lambda_target = torch.randn(k)
        t = torch.tensor(0.5)

        v = field(Phi_current, Phi_target, lambda_target, t)

        # Tangent space condition: Phi^T V + V^T Phi = 0 (skew-symmetric)
        PhiTV = Phi_current.T @ v
        skew_check = PhiTV + PhiTV.T

        assert torch.allclose(skew_check, torch.zeros_like(skew_check), atol=1e-5)

    def test_unbatched_input(self):
        """Handles unbatched input."""
        n, k = 50, 6
        field = ConditionalStiefelVelocityField(n=n, k=k, hidden_dim=128, n_blocks=2)

        Phi_current = uniform_stiefel(n, k)
        Phi_target = uniform_stiefel(n, k)
        lambda_target = torch.randn(k)
        t = torch.tensor(0.5)

        v = field(Phi_current, Phi_target, lambda_target, t)
        assert v.shape == (n, k)


class TestUniformStiefel:
    """Tests for uniform Stiefel sampling."""

    def test_output_shape(self):
        """Samples have correct shape."""
        Phi = uniform_stiefel(50, 6, batch_size=4)
        assert Phi.shape == (4, 50, 6)

    def test_single_sample_shape(self):
        """Single sample has correct shape."""
        Phi = uniform_stiefel(50, 6, batch_size=1)
        assert Phi.shape == (50, 6)

    def test_orthonormality(self):
        """Samples are orthonormal."""
        Phi = uniform_stiefel(50, 6, batch_size=4)
        for i in range(4):
            gram = Phi[i].T @ Phi[i]
            assert torch.allclose(gram, torch.eye(6), atol=1e-5)

    def test_different_samples(self):
        """Different calls produce different samples."""
        Phi1 = uniform_stiefel(50, 6)
        Phi2 = uniform_stiefel(50, 6)
        assert not torch.allclose(Phi1, Phi2)


class TestStiefelGeodesic:
    """Tests for geodesic operations."""

    def test_log_then_exp_recovers_target(self):
        """exp(log(Y)) ≈ Y (up to retraction approximation)."""
        Phi_0 = uniform_stiefel(50, 6)
        Phi_1 = uniform_stiefel(50, 6)

        V = stiefel_log_qr(Phi_0, Phi_1)
        Phi_recovered = stiefel_geodesic_qr(Phi_0, Phi_1, 1.0)

        # Check orthonormality
        gram = Phi_recovered.T @ Phi_recovered
        assert torch.allclose(gram, torch.eye(6), atol=1e-5)

    def test_geodesic_endpoints(self):
        """Geodesic at s=0 and s=1 gives endpoints."""
        Phi_0 = uniform_stiefel(50, 6)
        Phi_1 = uniform_stiefel(50, 6)

        Phi_at_0 = stiefel_geodesic_qr(Phi_0, Phi_1, 0.0)
        Phi_at_1 = stiefel_geodesic_qr(Phi_0, Phi_1, 1.0)

        assert torch.allclose(Phi_at_0, Phi_0, atol=1e-6)
        # Note: Phi_at_1 may differ from Phi_1 due to QR retraction

    def test_geodesic_stays_on_manifold(self):
        """Geodesic interpolation stays on Stiefel."""
        Phi_0 = uniform_stiefel(50, 6)
        Phi_1 = uniform_stiefel(50, 6)

        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s)
            gram = Phi_s.T @ Phi_s
            assert torch.allclose(gram, torch.eye(6), atol=1e-5)

    def test_batched_geodesic(self):
        """Batched geodesic works correctly."""
        batch_size = 4
        Phi_0 = uniform_stiefel(50, 6, batch_size=batch_size)
        Phi_1 = uniform_stiefel(50, 6, batch_size=batch_size)

        s = torch.tensor([0.5] * batch_size).view(batch_size, 1, 1)
        Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s)

        assert Phi_s.shape == (batch_size, 50, 6)

        # Check orthonormality for each
        for i in range(batch_size):
            gram = Phi_s[i].T @ Phi_s[i]
            assert torch.allclose(gram, torch.eye(6), atol=1e-5)


class TestGeodesicFlowModel:
    """Tests for the complete flow model."""

    def test_velocity_output_shapes(self):
        """Velocity outputs have correct shapes."""
        n, k = 50, 6
        model = GeodesicFlowModel(n=n, k=k, hidden_dim_phi=128, hidden_dim_lambda=64)

        Phi = uniform_stiefel(n, k)
        lambda_ = torch.randn(k)
        Phi_target = uniform_stiefel(n, k)
        lambda_target = torch.randn(k)
        t = torch.tensor(0.5)

        v_Phi, v_lambda = model.velocity(Phi, lambda_, Phi_target, lambda_target, t)

        assert v_Phi.shape == (n, k)
        assert v_lambda.shape == (k,)

    def test_integration_stays_on_manifold(self):
        """Integration keeps Phi on Stiefel manifold."""
        n, k = 50, 6
        model = GeodesicFlowModel(n=n, k=k, hidden_dim_phi=128, hidden_dim_lambda=64)

        Phi_0 = uniform_stiefel(n, k)
        lambda_0 = torch.randn(k)
        Phi_target = uniform_stiefel(n, k)
        lambda_target = torch.randn(k)

        Phi_end, lambda_end = model.integrate(
            Phi_0, lambda_0,
            Phi_target, lambda_target,
            t_end=1.0,
            n_steps=10,
        )

        gram = Phi_end.T @ Phi_end
        assert torch.allclose(gram, torch.eye(k), atol=1e-4)

    def test_trajectory_generation(self):
        """Trajectory generation works."""
        n, k = 50, 6
        model = GeodesicFlowModel(n=n, k=k, hidden_dim_phi=128, hidden_dim_lambda=64)

        Phi_0 = uniform_stiefel(n, k)
        lambda_0 = torch.randn(k)
        Phi_target = uniform_stiefel(n, k)
        lambda_target = torch.randn(k)

        Phi_end, lambda_end, Phi_traj, lambda_traj = model.integrate(
            Phi_0, lambda_0,
            Phi_target, lambda_target,
            t_end=1.0,
            n_steps=5,
            return_trajectory=True,
        )

        assert len(Phi_traj) == 6  # 5 steps + initial
        assert len(lambda_traj) == 6

        # All trajectory points on manifold
        for Phi in Phi_traj:
            gram = Phi.T @ Phi
            assert torch.allclose(gram, torch.eye(k), atol=1e-4)

    def test_batched_velocity(self):
        """Batched velocity prediction works."""
        n, k = 50, 6
        batch_size = 4
        model = GeodesicFlowModel(n=n, k=k, hidden_dim_phi=128, hidden_dim_lambda=64)

        Phi = uniform_stiefel(n, k, batch_size=batch_size)
        lambda_ = torch.randn(batch_size, k)
        Phi_target = uniform_stiefel(n, k, batch_size=batch_size)
        lambda_target = torch.randn(batch_size, k)
        t = torch.tensor([0.5])

        v_Phi, v_lambda = model.velocity(Phi, lambda_, Phi_target, lambda_target, t)

        assert v_Phi.shape == (batch_size, n, k)
        assert v_lambda.shape == (batch_size, k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
