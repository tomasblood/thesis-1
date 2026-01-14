"""Tests for Stiefel manifold geometry utilities."""

import numpy as np
import pytest

from temporal_spectral_flow.stiefel import StiefelManifold, StiefelInterpolator


class TestStiefelManifold:
    """Tests for StiefelManifold class."""

    @pytest.fixture
    def manifold(self):
        """Create a Stiefel manifold."""
        return StiefelManifold(n=50, k=5)

    @pytest.fixture
    def random_point(self, manifold):
        """Generate random point on manifold."""
        np.random.seed(42)
        return manifold.random_point()

    def test_is_on_manifold(self, manifold, random_point):
        """Test manifold membership check."""
        assert StiefelManifold.is_on_manifold(random_point)

        # Non-orthonormal matrix should fail
        bad_point = np.random.randn(50, 5)
        assert not StiefelManifold.is_on_manifold(bad_point)

    def test_project_to_manifold(self, manifold):
        """Test projection onto manifold."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X_proj = StiefelManifold.project_to_manifold(X)

        assert StiefelManifold.is_on_manifold(X_proj)

    def test_tangent_projection(self, manifold, random_point):
        """Test tangent space projection."""
        V = np.random.randn(50, 5)
        V_tan = StiefelManifold.project_to_tangent(random_point, V)

        # Check tangent condition: X^T V + V^T X = 0 (skew-symmetric)
        XTV = random_point.T @ V_tan
        assert np.allclose(XTV + XTV.T, 0, atol=1e-10)

    def test_random_tangent(self, manifold, random_point):
        """Test random tangent generation."""
        V = manifold.random_tangent(random_point)

        # Should satisfy tangent condition
        XTV = random_point.T @ V
        assert np.allclose(XTV + XTV.T, 0, atol=1e-10)

    def test_exp_map_stays_on_manifold(self, manifold, random_point):
        """Test that exponential map stays on manifold."""
        V = manifold.random_tangent(random_point)

        for t in [0.1, 0.5, 1.0, 2.0]:
            Y = StiefelManifold.exp_map(random_point, V, t)
            assert StiefelManifold.is_on_manifold(Y, tol=1e-5)

    def test_exp_map_zero(self, manifold, random_point):
        """Test exp_map at t=0 returns starting point."""
        V = manifold.random_tangent(random_point)
        Y = StiefelManifold.exp_map(random_point, V, t=0)

        assert np.allclose(Y, random_point, atol=1e-10)

    def test_log_map_inverse(self, manifold):
        """Test that log is inverse of exp."""
        np.random.seed(42)
        X = manifold.random_point()
        V = 0.5 * manifold.random_tangent(X)  # Small step for accuracy

        Y = StiefelManifold.exp_map(X, V)
        V_recovered = StiefelManifold.log_map(X, Y)

        assert np.allclose(V, V_recovered, atol=1e-4)

    def test_geodesic_distance_symmetric(self, manifold):
        """Test geodesic distance is symmetric."""
        np.random.seed(42)
        X = manifold.random_point()
        Y = manifold.random_point()

        d_XY = StiefelManifold.geodesic_distance(X, Y)
        d_YX = StiefelManifold.geodesic_distance(Y, X)

        assert np.isclose(d_XY, d_YX, atol=1e-10)

    def test_geodesic_distance_zero(self, manifold, random_point):
        """Test distance to self is zero."""
        d = StiefelManifold.geodesic_distance(random_point, random_point)
        assert np.isclose(d, 0, atol=1e-10)

    def test_qr_retraction(self, manifold, random_point):
        """Test QR retraction stays on manifold."""
        V = manifold.random_tangent(random_point)

        for t in [0.1, 0.5, 1.0]:
            Y = StiefelManifold.retract_qr(random_point, V, t)
            assert StiefelManifold.is_on_manifold(Y, tol=1e-10)

    def test_polar_retraction(self, manifold, random_point):
        """Test polar retraction stays on manifold."""
        V = manifold.random_tangent(random_point)

        for t in [0.1, 0.5, 1.0]:
            Y = StiefelManifold.retract_polar(random_point, V, t)
            assert StiefelManifold.is_on_manifold(Y, tol=1e-10)

    def test_parallel_transport(self, manifold):
        """Test parallel transport produces tangent vector."""
        np.random.seed(42)
        X = manifold.random_point()
        Y = manifold.random_point()
        V = manifold.random_tangent(X)

        V_transported = StiefelManifold.parallel_transport(X, Y, V)

        # Check transported vector is tangent at Y
        YTV = Y.T @ V_transported
        assert np.allclose(YTV + YTV.T, 0, atol=1e-10)

    def test_inner_product_symmetric(self, manifold, random_point):
        """Test inner product is symmetric."""
        V1 = manifold.random_tangent(random_point)
        V2 = manifold.random_tangent(random_point)

        ip1 = StiefelManifold.inner_product(random_point, V1, V2)
        ip2 = StiefelManifold.inner_product(random_point, V2, V1)

        assert np.isclose(ip1, ip2, atol=1e-10)

    def test_norm_nonnegative(self, manifold, random_point):
        """Test norm is non-negative."""
        V = manifold.random_tangent(random_point)
        norm = StiefelManifold.norm(random_point, V)

        assert norm >= 0

    def test_norm_zero_for_zero_vector(self, manifold, random_point):
        """Test norm of zero vector is zero."""
        V = np.zeros_like(random_point)
        norm = StiefelManifold.norm(random_point, V)

        assert np.isclose(norm, 0, atol=1e-10)


class TestStiefelInterpolator:
    """Tests for StiefelInterpolator class."""

    @pytest.fixture
    def manifold(self):
        return StiefelManifold(n=30, k=4)

    @pytest.fixture
    def interpolator(self):
        return StiefelInterpolator()

    def test_interpolate_endpoints(self, manifold, interpolator):
        """Test interpolation at endpoints."""
        np.random.seed(42)
        X = manifold.random_point()
        Y = manifold.random_point()

        # t=0 should give X
        result_0 = interpolator.interpolate(X, Y, t=0)
        assert np.allclose(result_0, X, atol=1e-10)

        # t=1 should give Y
        result_1 = interpolator.interpolate(X, Y, t=1)
        assert np.allclose(result_1, Y, atol=1e-4)

    def test_interpolate_midpoint_on_manifold(self, manifold, interpolator):
        """Test midpoint is on manifold."""
        np.random.seed(42)
        X = manifold.random_point()
        Y = manifold.random_point()

        mid = interpolator.interpolate(X, Y, t=0.5)
        assert StiefelManifold.is_on_manifold(mid, tol=1e-5)

    def test_interpolate_sequence(self, manifold, interpolator):
        """Test sequence generation."""
        np.random.seed(42)
        X = manifold.random_point()
        Y = manifold.random_point()

        sequence = interpolator.interpolate_sequence(X, Y, n_steps=5)

        assert len(sequence) == 7  # 5 intermediate + 2 endpoints

        # All points should be on manifold
        for point in sequence:
            assert StiefelManifold.is_on_manifold(point, tol=1e-5)

        # First and last should match endpoints
        assert np.allclose(sequence[0], X, atol=1e-10)
        assert np.allclose(sequence[-1], Y, atol=1e-4)
