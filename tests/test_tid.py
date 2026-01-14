"""Tests for Temporal Intrinsic Dimension computation."""

import numpy as np
import pytest

from temporal_spectral_flow.tid import (
    TemporalIntrinsicDimension,
    DimensionStability,
    TIDResult,
    compute_tid_from_data,
)
from temporal_spectral_flow.spectral import SpectralSnapshot, TemporalSpectralEmbedding
from temporal_spectral_flow.flow import SpectralFlowModel


class TestDimensionStability:
    """Tests for DimensionStability dataclass."""

    def test_creation(self):
        """Test creating stability object."""
        stability = DimensionStability(
            dimension_index=0,
            stability_score=0.8,
            flow_energy=0.9,
            curvature=0.7,
            mass_consistency=0.85,
            eigenvalue_stability=0.75,
        )

        assert stability.dimension_index == 0
        assert stability.stability_score == 0.8


class TestTIDResult:
    """Tests for TIDResult dataclass."""

    def test_creation(self):
        """Test creating TID result."""
        stabilities = [
            DimensionStability(i, 0.8 - i * 0.1, 0.9, 0.8, 0.85, 0.75)
            for i in range(5)
        ]

        result = TIDResult(
            tid_count=3,
            tid_effective=3.5,
            dimension_stabilities=stabilities,
            stability_threshold=0.5,
            overall_stability=0.6,
        )

        assert result.tid_count == 3
        assert result.tid_effective == 3.5
        assert len(result.dimension_stabilities) == 5


class TestTemporalIntrinsicDimension:
    """Tests for TID analyzer."""

    @pytest.fixture
    def sample_snapshots(self):
        """Generate sample spectral snapshots."""
        np.random.seed(42)
        snapshots = []
        n, k = 50, 5

        for t in range(5):
            Phi, _ = np.linalg.qr(np.random.randn(n, k))
            eigenvalues = np.sort(np.abs(np.random.randn(k)) * 0.1)

            snapshot = SpectralSnapshot(
                Phi=Phi,
                eigenvalues=eigenvalues,
                n_samples=n,
                k=k,
            )
            snapshots.append(snapshot)

        return snapshots

    @pytest.fixture
    def evolving_snapshots(self):
        """Generate snapshots with smooth evolution."""
        np.random.seed(42)
        n, k = 50, 5

        # Initial embedding
        Phi0, _ = np.linalg.qr(np.random.randn(n, k))
        eigenvalues = np.array([0.01, 0.05, 0.1, 0.5, 1.0])

        snapshots = []
        for t in range(5):
            # Smooth perturbation
            perturbation = 0.1 * t * np.random.randn(n, k)
            Phi_t, _ = np.linalg.qr(Phi0 + perturbation)

            snapshot = SpectralSnapshot(
                Phi=Phi_t,
                eigenvalues=eigenvalues * (1 + 0.01 * t),
                n_samples=n,
                k=k,
            )
            snapshots.append(snapshot)

        return snapshots

    @pytest.fixture
    def tid_analyzer(self):
        """Create TID analyzer."""
        return TemporalIntrinsicDimension(stability_threshold=0.5)

    def test_compute_tid_basic(self, tid_analyzer, sample_snapshots):
        """Test basic TID computation."""
        result = tid_analyzer.compute_tid(sample_snapshots, use_flow=False)

        assert isinstance(result, TIDResult)
        assert result.tid_count >= 0
        assert result.tid_effective >= 0
        assert len(result.dimension_stabilities) == 5

    def test_stability_scores_bounded(self, tid_analyzer, sample_snapshots):
        """Test that stability scores are in [0, 1]."""
        result = tid_analyzer.compute_tid(sample_snapshots, use_flow=False)

        for stability in result.dimension_stabilities:
            assert 0 <= stability.stability_score <= 1
            assert 0 <= stability.flow_energy <= 1
            assert 0 <= stability.curvature <= 1
            assert 0 <= stability.mass_consistency <= 1
            assert 0 <= stability.eigenvalue_stability <= 1

    def test_tid_count_reasonable(self, tid_analyzer, sample_snapshots):
        """Test TID count is reasonable."""
        result = tid_analyzer.compute_tid(sample_snapshots, use_flow=False)

        # TID should be between 0 and k
        assert 0 <= result.tid_count <= 5

    def test_smooth_evolution_higher_stability(self, tid_analyzer, evolving_snapshots):
        """Test that smooth evolution yields higher stability."""
        result = tid_analyzer.compute_tid(evolving_snapshots, use_flow=False)

        # Smooth evolution should have reasonable overall stability
        assert result.overall_stability > 0.1

    def test_count_aggregation(self, sample_snapshots):
        """Test count aggregation method."""
        analyzer = TemporalIntrinsicDimension(
            stability_threshold=0.3,
            aggregation_method="count",
        )
        result = analyzer.compute_tid(sample_snapshots, use_flow=False)

        # Count should be integer
        assert result.tid_count == result.tid_effective

    def test_soft_aggregation(self, sample_snapshots):
        """Test soft aggregation method."""
        analyzer = TemporalIntrinsicDimension(
            stability_threshold=0.3,
            aggregation_method="soft",
        )
        result = analyzer.compute_tid(sample_snapshots, use_flow=False)

        # Soft aggregation can give fractional effective dimension
        assert result.tid_effective >= 0

    def test_entropy_aggregation(self, sample_snapshots):
        """Test entropy aggregation method."""
        analyzer = TemporalIntrinsicDimension(
            stability_threshold=0.3,
            aggregation_method="entropy",
        )
        result = analyzer.compute_tid(sample_snapshots, use_flow=False)

        # Entropy-based dimension should be between 1 and k
        assert 0 < result.tid_effective <= 5

    def test_tid_over_time(self, tid_analyzer, evolving_snapshots):
        """Test TID computation over sliding windows."""
        tid_values = tid_analyzer.compute_tid_over_time(
            evolving_snapshots,
            window_size=3,
        )

        assert len(tid_values) == 3  # 5 - 3 + 1 windows
        assert np.all(tid_values >= 0)

    def test_insufficient_snapshots(self, tid_analyzer):
        """Test error on insufficient snapshots."""
        np.random.seed(42)
        Phi, _ = np.linalg.qr(np.random.randn(50, 5))
        snapshot = SpectralSnapshot(
            Phi=Phi,
            eigenvalues=np.zeros(5),
            n_samples=50,
            k=5,
        )

        with pytest.raises(ValueError, match="at least 2"):
            tid_analyzer.compute_tid([snapshot])


class TestComputeTIDFromData:
    """Tests for convenience function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample temporal data."""
        np.random.seed(42)
        return [np.random.randn(80, 20) for _ in range(5)]

    def test_compute_tid_from_data(self, sample_data):
        """Test computing TID directly from data."""
        result = compute_tid_from_data(
            sample_data,
            k=5,
            stability_threshold=0.3,
        )

        assert isinstance(result, TIDResult)
        assert result.tid_count >= 0
        assert len(result.dimension_stabilities) == 5
