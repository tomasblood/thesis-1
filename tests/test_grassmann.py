"""
Tests for Grassmannian operations.

Verifies that all operations are properly invariant to basis choice:
- Sign flips
- Permutations
- Rotations within O(k)
"""

import numpy as np
import torch
import pytest
from scipy.stats import special_ortho_group

from temporal_spectral_flow.grassmann import (
    to_projection,
    grassmann_distance,
    projection_frobenius,
    projection_frobenius_efficient,
    principal_angles,
    to_projection_torch,
    grassmann_distance_torch,
    projection_frobenius_torch,
    grassmann_loss,
    compute_distance_matrix,
)


class TestProjection:
    """Tests for projection matrix computation."""

    def test_projection_is_symmetric(self):
        """P = Φ Φ^T should be symmetric."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        P = to_projection(Phi)

        assert np.allclose(P, P.T)

    def test_projection_is_idempotent(self):
        """P^2 = P for a projection matrix."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        P = to_projection(Phi)

        assert np.allclose(P @ P, P)

    def test_projection_invariant_to_sign(self):
        """P(Φ) = P(Φ_flipped) for column sign flips."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        # Flip signs of some columns
        signs = np.array([1, -1, 1, -1, -1, 1])
        Phi_flipped = Phi * signs

        P_original = to_projection(Phi)
        P_flipped = to_projection(Phi_flipped)

        assert np.allclose(P_original, P_flipped)

    def test_projection_invariant_to_permutation(self):
        """P(Φ) = P(Φ_permuted) for column permutations."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        # Permute columns
        perm = [3, 1, 5, 0, 2, 4]
        Phi_permuted = Phi[:, perm]

        P_original = to_projection(Phi)
        P_permuted = to_projection(Phi_permuted)

        assert np.allclose(P_original, P_permuted)

    def test_projection_invariant_to_rotation(self):
        """P(Φ) = P(Φ R) for R ∈ O(k)."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        # Random rotation in O(k)
        R = special_ortho_group.rvs(6, random_state=42)
        Phi_rotated = Phi @ R

        P_original = to_projection(Phi)
        P_rotated = to_projection(Phi_rotated)

        assert np.allclose(P_original, P_rotated)


class TestGrassmannDistance:
    """Tests for Grassmann geodesic distance."""

    def test_distance_zero_for_same_subspace(self):
        """d(Φ, Φ) = 0."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        d = grassmann_distance(Phi, Phi)

        assert d == pytest.approx(0.0, abs=1e-6)

    def test_distance_zero_for_rotated_basis(self):
        """d(Φ, Φ R) = 0 for R ∈ O(k) (same subspace)."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        R = special_ortho_group.rvs(6, random_state=42)
        Phi_rotated = Phi @ R

        d = grassmann_distance(Phi, Phi_rotated)

        assert d == pytest.approx(0.0, abs=1e-6)

    def test_distance_symmetric(self):
        """d(Φ_0, Φ_1) = d(Φ_1, Φ_0)."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        d_01 = grassmann_distance(Phi_0, Phi_1)
        d_10 = grassmann_distance(Phi_1, Phi_0)

        assert d_01 == pytest.approx(d_10)

    def test_distance_nonnegative(self):
        """d(Φ_0, Φ_1) >= 0."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        d = grassmann_distance(Phi_0, Phi_1)

        assert d >= 0

    def test_distance_invariant_to_basis_choice(self):
        """d(Φ_0, Φ_1) = d(Φ_0 R_0, Φ_1 R_1) for any R_0, R_1 ∈ O(k)."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        R_0 = special_ortho_group.rvs(6, random_state=42)
        R_1 = special_ortho_group.rvs(6, random_state=43)

        Phi_0_rotated = Phi_0 @ R_0
        Phi_1_rotated = Phi_1 @ R_1

        d_original = grassmann_distance(Phi_0, Phi_1)
        d_rotated = grassmann_distance(Phi_0_rotated, Phi_1_rotated)

        assert d_original == pytest.approx(d_rotated)

    def test_distance_positive_for_different_subspaces(self):
        """d(Φ_0, Φ_1) > 0 for genuinely different subspaces."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        d = grassmann_distance(Phi_0, Phi_1)

        assert d > 0.1  # Should be clearly positive for random subspaces


class TestProjectionFrobenius:
    """Tests for projection Frobenius distance."""

    def test_frobenius_zero_for_same_subspace(self):
        """||P_0 - P_1||_F = 0 for same subspace."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        R = special_ortho_group.rvs(6, random_state=42)
        Phi_rotated = Phi @ R

        d = projection_frobenius(Phi, Phi_rotated)

        assert d == pytest.approx(0.0, abs=1e-10)

    def test_frobenius_efficient_matches_direct(self):
        """Efficient computation matches direct computation."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        d_direct = projection_frobenius(Phi_0, Phi_1)
        d_efficient = projection_frobenius_efficient(Phi_0, Phi_1)

        assert d_direct == pytest.approx(d_efficient)

    def test_frobenius_invariant_to_rotation(self):
        """||P_0 - P_1||_F invariant to basis rotation."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        R_0 = special_ortho_group.rvs(6, random_state=42)
        R_1 = special_ortho_group.rvs(6, random_state=43)

        d_original = projection_frobenius(Phi_0, Phi_1)
        d_rotated = projection_frobenius(Phi_0 @ R_0, Phi_1 @ R_1)

        assert d_original == pytest.approx(d_rotated)


class TestPrincipalAngles:
    """Tests for principal angles computation."""

    def test_angles_zero_for_same_subspace(self):
        """All angles zero for same subspace."""
        rng = np.random.default_rng(42)
        Phi = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        R = special_ortho_group.rvs(6, random_state=42)

        angles = principal_angles(Phi, Phi @ R)

        assert np.allclose(angles, 0, atol=1e-6)

    def test_angles_in_valid_range(self):
        """Principal angles should be in [0, π/2]."""
        rng = np.random.default_rng(42)
        Phi_0 = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1 = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        angles = principal_angles(Phi_0, Phi_1)

        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi / 2 + 1e-10)


class TestTorchImplementations:
    """Tests for PyTorch implementations."""

    def test_torch_projection_matches_numpy(self):
        """PyTorch projection matches NumPy."""
        rng = np.random.default_rng(42)
        Phi_np = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_torch = torch.from_numpy(Phi_np)

        P_np = to_projection(Phi_np)
        P_torch = to_projection_torch(Phi_torch).numpy()

        assert np.allclose(P_np, P_torch)

    def test_torch_grassmann_distance_matches_numpy(self):
        """PyTorch Grassmann distance matches NumPy."""
        rng = np.random.default_rng(42)
        Phi_0_np = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1_np = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        Phi_0_torch = torch.from_numpy(Phi_0_np)
        Phi_1_torch = torch.from_numpy(Phi_1_np)

        d_np = grassmann_distance(Phi_0_np, Phi_1_np)
        d_torch = grassmann_distance_torch(Phi_0_torch, Phi_1_torch).item()

        assert d_np == pytest.approx(d_torch, rel=1e-5)

    def test_torch_frobenius_matches_numpy(self):
        """PyTorch Frobenius distance matches NumPy."""
        rng = np.random.default_rng(42)
        Phi_0_np = np.linalg.qr(rng.standard_normal((50, 6)))[0]
        Phi_1_np = np.linalg.qr(rng.standard_normal((50, 6)))[0]

        Phi_0_torch = torch.from_numpy(Phi_0_np)
        Phi_1_torch = torch.from_numpy(Phi_1_np)

        d_np = projection_frobenius_efficient(Phi_0_np, Phi_1_np) ** 2  # Torch returns squared
        d_torch = projection_frobenius_torch(Phi_0_torch, Phi_1_torch).item()

        assert d_np == pytest.approx(d_torch, rel=1e-5)

    def test_grassmann_loss_differentiable(self):
        """Grassmann loss should be differentiable."""
        rng = np.random.default_rng(42)
        Phi_pred = torch.from_numpy(
            np.linalg.qr(rng.standard_normal((50, 6)))[0]
        ).requires_grad_(True)
        Phi_target = torch.from_numpy(
            np.linalg.qr(rng.standard_normal((50, 6)))[0]
        )

        loss = grassmann_loss(Phi_pred, Phi_target)
        loss.backward()

        assert Phi_pred.grad is not None
        assert not torch.isnan(Phi_pred.grad).any()

    def test_batched_operations(self):
        """Batched operations should work correctly."""
        rng = np.random.default_rng(42)
        batch_size = 4
        n, k = 50, 6

        Phi_0 = torch.from_numpy(
            np.stack([np.linalg.qr(rng.standard_normal((n, k)))[0]
                      for _ in range(batch_size)])
        )
        Phi_1 = torch.from_numpy(
            np.stack([np.linalg.qr(rng.standard_normal((n, k)))[0]
                      for _ in range(batch_size)])
        )

        # Batched distance
        d_batched = grassmann_distance_torch(Phi_0, Phi_1)
        assert d_batched.shape == (batch_size,)

        # Individual distances should match
        for i in range(batch_size):
            d_individual = grassmann_distance_torch(Phi_0[i], Phi_1[i])
            assert d_batched[i] == pytest.approx(d_individual.item(), rel=1e-5)


class TestDistanceMatrix:
    """Tests for distance matrix computation."""

    def test_distance_matrix_symmetric(self):
        """Distance matrix should be symmetric."""
        rng = np.random.default_rng(42)
        sequence = [np.linalg.qr(rng.standard_normal((50, 6)))[0]
                    for _ in range(10)]

        D = compute_distance_matrix(sequence)

        assert np.allclose(D, D.T)

    def test_distance_matrix_diagonal_zero(self):
        """Diagonal should be zero."""
        rng = np.random.default_rng(42)
        sequence = [np.linalg.qr(rng.standard_normal((50, 6)))[0]
                    for _ in range(10)]

        D = compute_distance_matrix(sequence)

        assert np.allclose(np.diag(D), 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
