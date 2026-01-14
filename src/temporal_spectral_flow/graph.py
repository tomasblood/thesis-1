"""
Graph construction and Laplacian computation for temporal snapshots.

This module handles the conversion of raw high-dimensional data into
neighborhood graphs and their associated Laplacian operators, which
approximate diffusion on the underlying data manifold.
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


class GraphConstructor:
    """
    Constructs neighborhood graphs and Laplacian operators from data snapshots.

    The graph encodes local connectivity structure, not global linear structure.
    The Laplacian serves as a discrete geometric operator approximating diffusion
    on the data manifold.

    Attributes:
        method: Graph construction method ('knn', 'kernel', 'fuzzy_simplicial')
        n_neighbors: Number of neighbors for kNN-based methods
        kernel_bandwidth: Bandwidth for kernel-based methods (auto-computed if None)
        laplacian_type: Type of Laplacian normalization
    """

    def __init__(
        self,
        method: Literal["knn", "kernel", "fuzzy_simplicial"] = "knn",
        n_neighbors: int = 15,
        kernel_bandwidth: Optional[float] = None,
        laplacian_type: Literal["unnormalized", "symmetric", "random_walk"] = "symmetric",
    ):
        """
        Initialize the graph constructor.

        Args:
            method: Graph construction method
                - 'knn': k-nearest neighbors with symmetric edges
                - 'kernel': Gaussian kernel with bandwidth
                - 'fuzzy_simplicial': UMAP-style fuzzy simplicial set
            n_neighbors: Number of neighbors for kNN-based methods
            kernel_bandwidth: Bandwidth for kernel methods (auto if None)
            laplacian_type: Laplacian normalization type
                - 'unnormalized': L = D - W
                - 'symmetric': L = I - D^{-1/2} W D^{-1/2}
                - 'random_walk': L = I - D^{-1} W
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.kernel_bandwidth = kernel_bandwidth
        self.laplacian_type = laplacian_type

    def construct_graph(
        self,
        X: NDArray[np.floating],
        return_distances: bool = False,
    ) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, NDArray]]:
        """
        Construct a neighborhood graph from data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            return_distances: Whether to return distance matrix

        Returns:
            W: Sparse affinity/weight matrix of shape (n_samples, n_samples)
            distances: Optional distance matrix if return_distances=True
        """
        n_samples = X.shape[0]

        if self.method == "knn":
            W, distances = self._construct_knn_graph(X)
        elif self.method == "kernel":
            W, distances = self._construct_kernel_graph(X)
        elif self.method == "fuzzy_simplicial":
            W, distances = self._construct_fuzzy_simplicial_graph(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_distances:
            return W, distances
        return W

    def _construct_knn_graph(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[sparse.csr_matrix, NDArray]:
        """Construct k-nearest neighbors graph with Gaussian weights."""
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        # Find k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Remove self-connections (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Compute adaptive bandwidth (local scaling)
        sigma = distances[:, -1]  # Distance to k-th neighbor
        sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero

        # Construct sparse affinity matrix with Gaussian weights
        rows = np.repeat(np.arange(n_samples), k)
        cols = indices.ravel()

        # Compute weights: exp(-d^2 / (sigma_i * sigma_j))
        sigma_i = sigma[rows]
        sigma_j = sigma[cols]
        weights = np.exp(-distances.ravel() ** 2 / (sigma_i * sigma_j))

        W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))

        # Symmetrize: W = (W + W^T) / 2
        W = (W + W.T) / 2

        # Full distance matrix for potential later use
        full_distances = squareform(pdist(X, metric="euclidean"))

        return W, full_distances

    def _construct_kernel_graph(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[sparse.csr_matrix, NDArray]:
        """Construct Gaussian kernel graph."""
        # Compute pairwise distances
        distances = squareform(pdist(X, metric="euclidean"))

        # Auto-compute bandwidth if not specified
        if self.kernel_bandwidth is None:
            # Use median heuristic
            bandwidth = np.median(distances[distances > 0])
        else:
            bandwidth = self.kernel_bandwidth

        # Compute Gaussian kernel
        W = np.exp(-distances ** 2 / (2 * bandwidth ** 2))

        # Zero out self-connections
        np.fill_diagonal(W, 0)

        # Sparsify: keep only k-nearest neighbor connections
        k = min(self.n_neighbors, X.shape[0] - 1)
        W_sparse = self._sparsify_by_knn(W, k)

        return W_sparse, distances

    def _construct_fuzzy_simplicial_graph(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[sparse.csr_matrix, NDArray]:
        """
        Construct fuzzy simplicial set (UMAP-style).

        This creates a graph where edge weights represent membership strengths
        in a fuzzy topological structure.
        """
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        # Find k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Remove self-connections
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Compute local connectivity (distance to nearest neighbor)
        rho = distances[:, 0]

        # Binary search for sigma to achieve target sum of weights
        target_sum = np.log2(k)
        sigma = self._compute_sigma_fuzzy(distances, rho, target_sum)

        # Compute fuzzy set membership strengths
        rows = np.repeat(np.arange(n_samples), k)
        cols = indices.ravel()

        d = distances.ravel()
        rho_expanded = np.repeat(rho, k)
        sigma_expanded = np.repeat(sigma, k)

        weights = np.exp(-np.maximum(0, d - rho_expanded) / sigma_expanded)

        W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))

        # Fuzzy union: W_sym = W + W^T - W * W^T (probabilistic OR)
        W_T = W.T.tocsr()
        W_product = W.multiply(W_T)
        W = W + W_T - W_product

        full_distances = squareform(pdist(X, metric="euclidean"))

        return W, full_distances

    def _compute_sigma_fuzzy(
        self,
        distances: NDArray[np.floating],
        rho: NDArray[np.floating],
        target: float,
        n_iter: int = 64,
        tol: float = 1e-5,
    ) -> NDArray[np.floating]:
        """Binary search for sigma values to achieve target sum."""
        n_samples = distances.shape[0]
        sigma = np.ones(n_samples)

        for i in range(n_samples):
            lo, hi = 1e-10, 1000.0
            mid = 1.0

            for _ in range(n_iter):
                psum = np.sum(np.exp(-np.maximum(0, distances[i] - rho[i]) / mid))

                if np.abs(psum - target) < tol:
                    break

                if psum > target:
                    hi = mid
                else:
                    lo = mid

                mid = (lo + hi) / 2

            sigma[i] = mid

        return sigma

    def _sparsify_by_knn(
        self,
        W: NDArray[np.floating],
        k: int,
    ) -> sparse.csr_matrix:
        """Keep only k-nearest neighbor edges."""
        n = W.shape[0]

        # For each row, keep only top k values
        W_sparse = np.zeros_like(W)
        for i in range(n):
            idx = np.argpartition(W[i], -k)[-k:]
            W_sparse[i, idx] = W[i, idx]

        # Symmetrize
        W_sparse = (W_sparse + W_sparse.T) / 2

        return sparse.csr_matrix(W_sparse)

    def compute_laplacian(
        self,
        W: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Compute the graph Laplacian from an affinity matrix.

        Args:
            W: Sparse affinity matrix

        Returns:
            L: Sparse Laplacian matrix
        """
        n = W.shape[0]

        # Compute degree matrix
        degrees = np.asarray(W.sum(axis=1)).ravel()
        degrees = np.maximum(degrees, 1e-10)  # Avoid division by zero

        if self.laplacian_type == "unnormalized":
            # L = D - W
            D = sparse.diags(degrees)
            L = D - W

        elif self.laplacian_type == "symmetric":
            # L = I - D^{-1/2} W D^{-1/2}
            d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
            L = sparse.eye(n) - d_inv_sqrt @ W @ d_inv_sqrt

        elif self.laplacian_type == "random_walk":
            # L = I - D^{-1} W
            d_inv = sparse.diags(1.0 / degrees)
            L = sparse.eye(n) - d_inv @ W

        else:
            raise ValueError(f"Unknown laplacian_type: {self.laplacian_type}")

        return L.tocsr()

    def process_snapshot(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Process a single temporal snapshot: construct graph and Laplacian.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            W: Sparse affinity matrix
            L: Sparse Laplacian matrix
        """
        W = self.construct_graph(X)
        L = self.compute_laplacian(W)
        return W, L
