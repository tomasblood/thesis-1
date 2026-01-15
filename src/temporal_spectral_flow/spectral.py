"""
Spectral embedding computation for temporal snapshots.

This module computes the leading eigenvectors of the graph Laplacian,
producing points on the Stiefel manifold St(N, k) that serve as
geometry-aware, coordinate-free representations of each snapshot.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh

from temporal_spectral_flow.graph import GraphConstructor


@dataclass
class SpectralSnapshot:
    """
    Container for spectral representation of a temporal snapshot.

    Attributes:
        Phi: Spectral embedding matrix of shape (N, k), point on St(N, k)
        eigenvalues: Corresponding eigenvalues of shape (k,)
        n_samples: Number of samples in the snapshot
        k: Number of spectral dimensions
        graph_weights: Optional affinity matrix
        laplacian: Optional Laplacian matrix
    """

    Phi: NDArray[np.floating]
    eigenvalues: NDArray[np.floating]
    n_samples: int
    k: int
    graph_weights: Optional[sparse.csr_matrix] = None
    laplacian: Optional[sparse.csr_matrix] = None

    def __post_init__(self):
        """Validate that Phi is approximately orthonormal."""
        gram = self.Phi.T @ self.Phi
        identity = np.eye(self.k)
        if not np.allclose(gram, identity, atol=1e-6):
            raise ValueError(
                f"Phi columns not orthonormal. "
                f"Max deviation: {np.max(np.abs(gram - identity)):.2e}"
            )


class SpectralEmbedding:
    """
    Computes spectral embeddings from data snapshots.

    Each snapshot is mapped to a point on the Stiefel manifold St(N, k),
    where columns are the leading eigenvectors of the graph Laplacian.
    This representation:
    - Performs geometry-aware dimensionality reduction
    - Removes arbitrary linear coordinate systems
    - Captures global geometric modes (low-frequency diffusion modes)

    Attributes:
        k: Number of spectral dimensions to compute
        graph_constructor: GraphConstructor instance for building graphs
        skip_first: Whether to skip the trivial constant eigenvector
    """

    def __init__(
        self,
        k: int = 10,
        graph_constructor: Optional[GraphConstructor] = None,
        skip_first: bool = True,
        normalize_eigenvectors: bool = True,
    ):
        """
        Initialize spectral embedding.

        Args:
            k: Number of spectral dimensions (eigenvectors) to compute
            graph_constructor: GraphConstructor for building neighborhood graphs.
                If None, uses default kNN construction.
            skip_first: Whether to skip the first (trivial) eigenvector.
                For symmetric normalized Laplacian, this is constant.
            normalize_eigenvectors: Ensure orthonormality via QR if needed
        """
        self.k = k
        self.graph_constructor = graph_constructor or GraphConstructor()
        self.skip_first = skip_first
        self.normalize_eigenvectors = normalize_eigenvectors

    def embed(
        self,
        X: NDArray[np.floating],
        return_graph: bool = False,
    ) -> SpectralSnapshot:
        """
        Compute spectral embedding from raw data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            return_graph: Whether to include graph/Laplacian in output

        Returns:
            SpectralSnapshot containing the embedding and metadata
        """
        n_samples = X.shape[0]

        # Construct graph and Laplacian
        W, L = self.graph_constructor.process_snapshot(X)

        # Compute embedding
        Phi, eigenvalues = self.embed_from_laplacian(L, n_samples)

        return SpectralSnapshot(
            Phi=Phi,
            eigenvalues=eigenvalues,
            n_samples=n_samples,
            k=self.k,
            graph_weights=W if return_graph else None,
            laplacian=L if return_graph else None,
        )

    def embed_from_laplacian(
        self,
        L: sparse.csr_matrix,
        n_samples: Optional[int] = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute spectral embedding from a precomputed Laplacian.

        Args:
            L: Sparse Laplacian matrix of shape (N, N)
            n_samples: Number of samples (inferred from L if None)

        Returns:
            Phi: Eigenvector matrix of shape (N, k), orthonormal columns
            eigenvalues: Corresponding eigenvalues of shape (k,)
        """
        if n_samples is None:
            n_samples = L.shape[0]

        # Number of eigenvectors to compute
        n_components = self.k + (1 if self.skip_first else 0)
        n_components = min(n_components, n_samples - 1)

        # Compute smallest eigenvalues/eigenvectors
        # We want low-frequency modes (smallest eigenvalues of Laplacian)
        # Use small positive sigma to avoid singularity issues when L has zero eigenvalue
        eigenvalues, eigenvectors = eigsh(
            L.astype(np.float64),
            k=n_components,
            which="LM",  # Largest magnitude after shift-invert
            sigma=1e-6,  # Small positive shift to avoid singularity
            maxiter=1000,
            tol=1e-8,
        )

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip first eigenvector if requested (trivial constant mode)
        if self.skip_first:
            eigenvalues = eigenvalues[1 : self.k + 1]
            eigenvectors = eigenvectors[:, 1 : self.k + 1]
        else:
            eigenvalues = eigenvalues[: self.k]
            eigenvectors = eigenvectors[:, : self.k]

        # Ensure we have exactly k columns
        actual_k = eigenvectors.shape[1]
        if actual_k < self.k:
            # Pad with zeros if not enough eigenvectors
            padding = np.zeros((n_samples, self.k - actual_k))
            eigenvectors = np.hstack([eigenvectors, padding])
            eigenvalues = np.concatenate([eigenvalues, np.zeros(self.k - actual_k)])

        # Ensure orthonormality
        if self.normalize_eigenvectors:
            eigenvectors = self._ensure_orthonormal(eigenvectors)

        return eigenvectors, eigenvalues

    def _ensure_orthonormal(
        self,
        V: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Ensure matrix has orthonormal columns via QR decomposition.

        Args:
            V: Matrix of shape (N, k)

        Returns:
            Q: Orthonormal matrix of shape (N, k)
        """
        Q, R = np.linalg.qr(V, mode="reduced")

        # Ensure consistent sign (positive diagonal of R)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs

        return Q

    def compute_eigenvalue_gaps(
        self,
        eigenvalues: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute gaps between consecutive eigenvalues.

        Large gaps indicate natural boundaries between geometric scales.

        Args:
            eigenvalues: Array of eigenvalues

        Returns:
            gaps: Array of gaps (length k-1)
        """
        return np.diff(eigenvalues)

    def estimate_intrinsic_dimension(
        self,
        eigenvalues: NDArray[np.floating],
        threshold: float = 0.1,
    ) -> int:
        """
        Estimate intrinsic dimension from eigenvalue gaps.

        Args:
            eigenvalues: Array of eigenvalues
            threshold: Gap threshold as fraction of total spectral range

        Returns:
            Estimated intrinsic dimension
        """
        if len(eigenvalues) < 2:
            return 1

        gaps = self.compute_eigenvalue_gaps(eigenvalues)
        spectral_range = eigenvalues[-1] - eigenvalues[0]

        if spectral_range < 1e-10:
            return len(eigenvalues)

        normalized_gaps = gaps / spectral_range

        # Find first gap exceeding threshold
        large_gap_idx = np.where(normalized_gaps > threshold)[0]

        if len(large_gap_idx) == 0:
            return len(eigenvalues)

        return large_gap_idx[0] + 1


class TemporalSpectralEmbedding:
    """
    Computes spectral embeddings for a sequence of temporal snapshots.

    This class manages the embedding of multiple snapshots while ensuring
    consistent spectral dimension across time.
    """

    def __init__(
        self,
        k: int = 10,
        graph_constructor: Optional[GraphConstructor] = None,
        **kwargs,
    ):
        """
        Initialize temporal spectral embedding.

        Args:
            k: Number of spectral dimensions
            graph_constructor: GraphConstructor for building graphs
            **kwargs: Additional arguments passed to SpectralEmbedding
        """
        self.k = k
        self.embedder = SpectralEmbedding(
            k=k,
            graph_constructor=graph_constructor,
            **kwargs,
        )
        self.snapshots: list[SpectralSnapshot] = []

    def process_sequence(
        self,
        snapshots: list[NDArray[np.floating]],
        return_graphs: bool = False,
    ) -> list[SpectralSnapshot]:
        """
        Process a sequence of temporal snapshots.

        Args:
            snapshots: List of data matrices, each of shape (N_t, d)
            return_graphs: Whether to store graph structures

        Returns:
            List of SpectralSnapshot objects
        """
        self.snapshots = []

        for X in snapshots:
            snapshot = self.embedder.embed(X, return_graph=return_graphs)
            self.snapshots.append(snapshot)

        return self.snapshots

    def get_embedding_sequence(self) -> list[NDArray[np.floating]]:
        """
        Extract the sequence of Phi matrices.

        Returns:
            List of spectral embedding matrices
        """
        return [s.Phi for s in self.snapshots]

    def get_eigenvalue_sequence(self) -> list[NDArray[np.floating]]:
        """
        Extract the sequence of eigenvalue arrays.

        Returns:
            List of eigenvalue arrays
        """
        return [s.eigenvalues for s in self.snapshots]
