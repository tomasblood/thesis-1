"""Tests for optimal transport alignment."""

import numpy as np
import pytest

ot = pytest.importorskip("ot")

from temporal_spectral_flow.transport import (
    TransportAlignment,
    AlignmentResult,
    BasisAligner,
)


class TestTransportAlignment:
    """Tests for TransportAlignment class."""

    @pytest.fixture
    def aligner(self):
        """Create default aligner."""
        return TransportAlignment(method="balanced", reg=0.1)

    @pytest.fixture
    def unbalanced_aligner(self):
        """Create unbalanced aligner."""
        return TransportAlignment(method="unbalanced", reg=0.1, reg_m=1.0)

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample spectral embeddings."""
        np.random.seed(42)
        n, k = 50, 5
        Phi1, _ = np.linalg.qr(np.random.randn(n, k))
        Phi2, _ = np.linalg.qr(np.random.randn(n, k))
        return Phi1, Phi2

    @pytest.fixture
    def different_size_embeddings(self):
        """Generate embeddings with different sample sizes."""
        np.random.seed(42)
        k = 5
        Phi1, _ = np.linalg.qr(np.random.randn(50, k))
        Phi2, _ = np.linalg.qr(np.random.randn(60, k))
        return Phi1, Phi2

    def test_euclidean_cost(self, aligner, sample_embeddings):
        """Test Euclidean cost computation."""
        Phi1, Phi2 = sample_embeddings
        C = aligner.compute_cost_matrix(Phi1, Phi2)

        assert C.shape == (50, 50)
        assert np.all(C >= 0)

    def test_spectral_cost(self, sample_embeddings):
        """Test spectral-weighted cost."""
        Phi1, Phi2 = sample_embeddings
        eigenvalues = np.array([0.01, 0.1, 0.5, 1.0, 2.0])

        aligner = TransportAlignment(cost_type="spectral")
        C = aligner.compute_cost_matrix(
            Phi1, Phi2,
            eigenvalues_source=eigenvalues,
        )

        assert C.shape == (50, 50)
        assert np.all(C >= 0)

    def test_balanced_transport(self, aligner, sample_embeddings):
        """Test balanced optimal transport."""
        Phi1, Phi2 = sample_embeddings
        result = aligner.align(Phi1, Phi2)

        assert isinstance(result, AlignmentResult)
        assert result.transport_plan.shape == (50, 50)
        assert result.aligned_target.shape == Phi1.shape

        # Balanced transport: marginals should sum to 1
        assert np.isclose(result.transport_plan.sum(), 1.0, atol=1e-6)

    def test_unbalanced_transport(self, unbalanced_aligner, different_size_embeddings):
        """Test unbalanced transport with different sizes."""
        Phi1, Phi2 = different_size_embeddings
        result = unbalanced_aligner.align(Phi1, Phi2)

        assert result.transport_plan.shape == (50, 60)
        assert result.aligned_target.shape == (50, 5)

    def test_transport_plan_nonnegative(self, aligner, sample_embeddings):
        """Test transport plan is non-negative."""
        Phi1, Phi2 = sample_embeddings
        result = aligner.align(Phi1, Phi2)

        assert np.all(result.transport_plan >= 0)

    def test_aligned_target_reasonable(self, aligner, sample_embeddings):
        """Test aligned target is reasonable (similar structure)."""
        Phi1, Phi2 = sample_embeddings
        result = aligner.align(Phi1, Phi2)

        # Aligned target should have similar column norms to source
        source_norms = np.linalg.norm(Phi1, axis=0)
        aligned_norms = np.linalg.norm(result.aligned_target, axis=0)

        # Allow some deviation but should be same order of magnitude
        ratio = aligned_norms / (source_norms + 1e-10)
        assert np.all(ratio > 0.1)
        assert np.all(ratio < 10)

    def test_partial_transport(self, different_size_embeddings):
        """Test partial optimal transport."""
        Phi1, Phi2 = different_size_embeddings
        aligner = TransportAlignment(method="partial", reg=0.1)
        result = aligner.align(Phi1, Phi2)

        # Partial transport should move less than full mass
        assert result.transport_plan.sum() < 1.0

    def test_mass_change_computation(self, unbalanced_aligner, different_size_embeddings):
        """Test mass change computation."""
        Phi1, Phi2 = different_size_embeddings
        result = unbalanced_aligner.align(Phi1, Phi2)

        mass_created, mass_destroyed = unbalanced_aligner.compute_mass_change(result)

        assert len(mass_created) == Phi2.shape[0]
        assert len(mass_destroyed) == Phi1.shape[0]
        assert np.all(mass_created >= 0)
        assert np.all(mass_destroyed >= 0)

    def test_custom_mass_distributions(self, aligner):
        """Test with custom mass distributions."""
        np.random.seed(42)
        n1, n2, k = 40, 40, 4
        Phi1, _ = np.linalg.qr(np.random.randn(n1, k))
        Phi2, _ = np.linalg.qr(np.random.randn(n2, k))

        # Non-uniform masses
        mass1 = np.random.rand(n1)
        mass1 /= mass1.sum()
        mass2 = np.random.rand(n2)
        mass2 /= mass2.sum()

        result = aligner.align(Phi1, Phi2, mass_source=mass1, mass_target=mass2)

        assert result.transport_plan.shape == (n1, n2)


class TestBasisAligner:
    """Tests for BasisAligner class."""

    @pytest.fixture
    def aligner(self):
        return BasisAligner(allow_reflection=True)

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings with known relationship."""
        np.random.seed(42)
        n, k = 50, 5
        Phi1, _ = np.linalg.qr(np.random.randn(n, k))

        # Create Phi2 as rotated version of Phi1
        R = np.eye(k)
        R[0, 0], R[1, 1] = np.cos(0.5), np.cos(0.5)
        R[0, 1], R[1, 0] = -np.sin(0.5), np.sin(0.5)

        Phi2 = Phi1 @ R

        return Phi1, Phi2, R

    def test_align_bases(self, aligner, sample_embeddings):
        """Test basis alignment recovers rotation."""
        Phi1, Phi2, R = sample_embeddings

        Q, Phi_aligned = aligner.align_bases(Phi1, Phi2)

        # Q should be orthogonal
        assert np.allclose(Q @ Q.T, np.eye(5), atol=1e-6)

        # Aligned should match source
        assert np.allclose(Phi_aligned, Phi1, atol=1e-6)

    def test_align_bases_orthogonal(self, aligner):
        """Test alignment produces orthogonal Q."""
        np.random.seed(42)
        n, k = 50, 5
        Phi1, _ = np.linalg.qr(np.random.randn(n, k))
        Phi2, _ = np.linalg.qr(np.random.randn(n, k))

        Q, _ = aligner.align_bases(Phi1, Phi2)

        assert np.allclose(Q @ Q.T, np.eye(k), atol=1e-6)
        assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-6)

    def test_no_reflection(self):
        """Test alignment without reflections."""
        aligner = BasisAligner(allow_reflection=False)

        np.random.seed(42)
        n, k = 50, 5
        Phi1, _ = np.linalg.qr(np.random.randn(n, k))
        Phi2, _ = np.linalg.qr(np.random.randn(n, k))

        Q, _ = aligner.align_bases(Phi1, Phi2)

        # Determinant should be +1 (no reflection)
        det = np.linalg.det(Q)
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_align_signs(self, aligner):
        """Test simple sign alignment."""
        np.random.seed(42)
        n, k = 50, 5
        Phi1, _ = np.linalg.qr(np.random.randn(n, k))

        # Flip some signs
        signs = np.array([1, -1, 1, -1, 1])
        Phi2 = Phi1 * signs

        Phi_aligned = aligner.align_signs(Phi1, Phi2)

        # Should recover original
        assert np.allclose(np.abs(Phi_aligned), np.abs(Phi1), atol=1e-10)
