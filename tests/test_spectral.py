"""Tests for spectral embedding computation."""

import numpy as np
import pytest

from temporal_spectral_flow.spectral import (
    SpectralEmbedding,
    SpectralSnapshot,
    TemporalSpectralEmbedding,
)
from temporal_spectral_flow.graph import GraphConstructor


class TestSpectralSnapshot:
    """Tests for SpectralSnapshot dataclass."""

    def test_valid_snapshot(self):
        """Test creation of valid snapshot."""
        np.random.seed(42)
        n, k = 50, 5
        Phi, _ = np.linalg.qr(np.random.randn(n, k))
        eigenvalues = np.sort(np.random.rand(k))

        snapshot = SpectralSnapshot(
            Phi=Phi,
            eigenvalues=eigenvalues,
            n_samples=n,
            k=k,
        )

        assert snapshot.n_samples == n
        assert snapshot.k == k
        assert snapshot.Phi.shape == (n, k)

    def test_invalid_orthonormality(self):
        """Test that non-orthonormal Phi raises error."""
        n, k = 50, 5
        Phi = np.random.randn(n, k)  # Not orthonormal
        eigenvalues = np.zeros(k)

        with pytest.raises(ValueError, match="not orthonormal"):
            SpectralSnapshot(
                Phi=Phi,
                eigenvalues=eigenvalues,
                n_samples=n,
                k=k,
            )


class TestSpectralEmbedding:
    """Tests for SpectralEmbedding class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        return np.random.randn(100, 20)

    @pytest.fixture
    def embedder(self):
        """Create default embedder."""
        return SpectralEmbedding(k=10)

    def test_embed_basic(self, embedder, sample_data):
        """Test basic embedding."""
        snapshot = embedder.embed(sample_data)

        assert isinstance(snapshot, SpectralSnapshot)
        assert snapshot.n_samples == 100
        assert snapshot.k == 10
        assert snapshot.Phi.shape == (100, 10)

    def test_orthonormality(self, embedder, sample_data):
        """Test that embedding produces orthonormal columns."""
        snapshot = embedder.embed(sample_data)

        gram = snapshot.Phi.T @ snapshot.Phi
        assert np.allclose(gram, np.eye(10), atol=1e-6)

    def test_eigenvalues_sorted(self, embedder, sample_data):
        """Test that eigenvalues are sorted ascending."""
        snapshot = embedder.embed(sample_data)

        # Eigenvalues should be non-negative and sorted
        assert np.all(snapshot.eigenvalues >= -1e-10)
        assert np.all(np.diff(snapshot.eigenvalues) >= -1e-10)

    def test_return_graph(self, embedder, sample_data):
        """Test that graph can be returned."""
        snapshot = embedder.embed(sample_data, return_graph=True)

        assert snapshot.graph_weights is not None
        assert snapshot.laplacian is not None

    def test_skip_first_eigenvector(self, sample_data):
        """Test skipping the first (trivial) eigenvector."""
        embedder_skip = SpectralEmbedding(k=5, skip_first=True)
        embedder_noskip = SpectralEmbedding(k=5, skip_first=False)

        snapshot_skip = embedder_skip.embed(sample_data)
        snapshot_noskip = embedder_noskip.embed(sample_data)

        # First eigenvalue without skip should be ~0 (trivial mode)
        assert snapshot_noskip.eigenvalues[0] < 0.01

        # With skip, first eigenvalue should be larger
        assert snapshot_skip.eigenvalues[0] > snapshot_noskip.eigenvalues[0]

    def test_different_k_values(self, sample_data):
        """Test with different spectral dimensions."""
        for k in [3, 5, 10, 20]:
            embedder = SpectralEmbedding(k=k)
            snapshot = embedder.embed(sample_data)

            assert snapshot.k == k
            assert snapshot.Phi.shape == (100, k)
            assert len(snapshot.eigenvalues) == k

    def test_eigenvalue_gaps(self, embedder, sample_data):
        """Test eigenvalue gap computation."""
        snapshot = embedder.embed(sample_data)
        gaps = embedder.compute_eigenvalue_gaps(snapshot.eigenvalues)

        assert len(gaps) == len(snapshot.eigenvalues) - 1
        assert np.all(gaps >= -1e-10)

    def test_intrinsic_dimension_estimate(self, embedder, sample_data):
        """Test intrinsic dimension estimation."""
        snapshot = embedder.embed(sample_data)
        dim = embedder.estimate_intrinsic_dimension(snapshot.eigenvalues)

        assert 1 <= dim <= len(snapshot.eigenvalues)


class TestTemporalSpectralEmbedding:
    """Tests for temporal spectral embedding."""

    @pytest.fixture
    def temporal_data(self):
        """Generate temporal sequence of data."""
        np.random.seed(42)
        snapshots = []
        for t in range(5):
            # Slowly evolving data
            base = np.random.randn(80 + t * 5, 15)
            snapshots.append(base)
        return snapshots

    def test_process_sequence(self, temporal_data):
        """Test processing a sequence of snapshots."""
        embedder = TemporalSpectralEmbedding(k=8)
        snapshots = embedder.process_sequence(temporal_data)

        assert len(snapshots) == 5
        for i, snap in enumerate(snapshots):
            assert snap.k == 8
            assert snap.n_samples == 80 + i * 5

    def test_embedding_sequence(self, temporal_data):
        """Test extracting embedding sequence."""
        embedder = TemporalSpectralEmbedding(k=8)
        embedder.process_sequence(temporal_data)

        embeddings = embedder.get_embedding_sequence()
        assert len(embeddings) == 5

        eigenvalues = embedder.get_eigenvalue_sequence()
        assert len(eigenvalues) == 5
