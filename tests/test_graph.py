"""Tests for graph construction and Laplacian computation."""

import numpy as np
import pytest
from scipy import sparse

from temporal_spectral_flow.graph import GraphConstructor


class TestGraphConstructor:
    """Tests for GraphConstructor class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 10)

    @pytest.fixture
    def small_data(self):
        """Small dataset for detailed checks."""
        np.random.seed(42)
        return np.random.randn(20, 5)

    def test_knn_graph_construction(self, sample_data):
        """Test kNN graph construction."""
        gc = GraphConstructor(method="knn", n_neighbors=10)
        W = gc.construct_graph(sample_data)

        # Check output type and shape
        assert sparse.issparse(W)
        assert W.shape == (100, 100)

        # Check symmetry
        diff = W - W.T
        assert np.allclose(diff.toarray(), 0, atol=1e-10)

        # Check non-negativity
        assert np.all(W.toarray() >= 0)

        # Check no self-loops
        assert np.allclose(W.diagonal(), 0)

    def test_kernel_graph_construction(self, small_data):
        """Test Gaussian kernel graph construction."""
        gc = GraphConstructor(method="kernel", n_neighbors=5)
        W = gc.construct_graph(small_data)

        assert sparse.issparse(W)
        assert W.shape == (20, 20)
        assert np.all(W.toarray() >= 0)

    def test_fuzzy_simplicial_graph(self, small_data):
        """Test UMAP-style fuzzy simplicial set construction."""
        gc = GraphConstructor(method="fuzzy_simplicial", n_neighbors=5)
        W = gc.construct_graph(small_data)

        assert sparse.issparse(W)
        assert W.shape == (20, 20)
        # Weights should be in [0, 1] for fuzzy sets
        assert np.all(W.toarray() >= 0)
        assert np.all(W.toarray() <= 1 + 1e-6)

    def test_unnormalized_laplacian(self, small_data):
        """Test unnormalized Laplacian construction."""
        gc = GraphConstructor(laplacian_type="unnormalized")
        W = gc.construct_graph(small_data)
        L = gc.compute_laplacian(W)

        # Check shape
        assert L.shape == W.shape

        # Laplacian should be symmetric
        diff = L - L.T
        assert np.allclose(diff.toarray(), 0, atol=1e-10)

        # Row sums should be approximately 0 for unnormalized Laplacian
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 0, atol=1e-10)

    def test_symmetric_laplacian(self, small_data):
        """Test symmetric normalized Laplacian."""
        gc = GraphConstructor(laplacian_type="symmetric")
        W = gc.construct_graph(small_data)
        L = gc.compute_laplacian(W)

        # Check symmetry
        diff = L - L.T
        assert np.allclose(diff.toarray(), 0, atol=1e-10)

        # Eigenvalues should be in [0, 2] for normalized Laplacian
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigenvalues >= -1e-10)
        assert np.all(eigenvalues <= 2 + 1e-10)

    def test_random_walk_laplacian(self, small_data):
        """Test random walk Laplacian."""
        gc = GraphConstructor(laplacian_type="random_walk")
        W = gc.construct_graph(small_data)
        L = gc.compute_laplacian(W)

        # Random walk Laplacian is L = I - D^{-1}W
        # Eigenvalues should be in [0, 2]
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigenvalues >= -1e-10)
        assert np.all(eigenvalues <= 2 + 1e-10)

    def test_process_snapshot(self, sample_data):
        """Test full snapshot processing."""
        gc = GraphConstructor()
        W, L = gc.process_snapshot(sample_data)

        assert sparse.issparse(W)
        assert sparse.issparse(L)
        assert W.shape == L.shape == (100, 100)

    def test_return_distances(self, small_data):
        """Test distance matrix return."""
        gc = GraphConstructor()
        W, distances = gc.construct_graph(small_data, return_distances=True)

        assert distances.shape == (20, 20)
        assert np.all(distances >= 0)
        assert np.allclose(distances, distances.T)
        assert np.allclose(np.diag(distances), 0)

    def test_different_neighbor_counts(self, sample_data):
        """Test with different numbers of neighbors."""
        for k in [5, 10, 20]:
            gc = GraphConstructor(n_neighbors=k)
            W = gc.construct_graph(sample_data)
            assert sparse.issparse(W)
            # Each row should have at least k non-zero entries
            # (may have more due to symmetrization)
            nnz_per_row = np.diff(W.tocsr().indptr)
            assert np.all(nnz_per_row >= k)
