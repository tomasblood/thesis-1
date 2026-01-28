"""
Data generators for temporal spectral alignment toy examples.

Provides functions to generate temporal sequences of spectral embeddings
for demonstrating eigenvalue tracking, sign conventions, and alignment.
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass
class SpectralFrame:
    """
    Spectral embedding at a single timestep.

    Attributes:
        Phi: Eigenvector matrix (n, k)
        eigenvalues: Eigenvalue array (k,)
        t: Time value in [0, 1]
    """
    Phi: NDArray
    eigenvalues: NDArray
    t: float


def generate_merging_clusters(
    n_points_per_cluster: int = 50,
    cluster_std: float = 0.3,
    separation_start: float = 3.0,
    separation_end: float = 0.0,
    k_neighbors: int = 10,
    n_eigenvectors: int = 6,
    n_timesteps: int = 50,
    seed: int = 42,
) -> Tuple[List[SpectralFrame], NDArray]:
    """
    Generate temporal sequence of two Gaussian clusters merging.

    At t=0, clusters are well-separated (near-disconnected graph).
    At t=1, clusters have merged (single connected component).

    Spectral signature:
    - lambda_2 near 0 when separated (Fiedler value of near-disconnected graph)
    - lambda_2 increases as clusters merge
    - Eigenvectors transition from localized to global modes

    Args:
        n_points_per_cluster: Points in each cluster.
        cluster_std: Gaussian standard deviation.
        separation_start: Initial distance between cluster centers.
        separation_end: Final distance (0 = fully merged).
        k_neighbors: For k-NN graph construction.
        n_eigenvectors: Number of spectral dimensions.
        n_timesteps: Number of temporal snapshots.
        seed: Random seed for reproducibility.

    Returns:
        frames: List of SpectralFrame for each timestep.
        labels: Cluster membership array (n_total,).
    """
    rng = np.random.default_rng(seed)
    n_total = 2 * n_points_per_cluster

    # Fixed point positions within each cluster (consistent across time)
    cluster1_local = cluster_std * rng.standard_normal((n_points_per_cluster, 2))
    cluster2_local = cluster_std * rng.standard_normal((n_points_per_cluster, 2))

    labels = np.array([0] * n_points_per_cluster + [1] * n_points_per_cluster)

    frames = []

    for t_idx in range(n_timesteps):
        t = t_idx / max(n_timesteps - 1, 1)

        # Interpolate separation
        separation = separation_start * (1 - t) + separation_end * t

        # Position clusters
        center1 = np.array([-separation / 2, 0.0])
        center2 = np.array([separation / 2, 0.0])

        X = np.vstack([
            cluster1_local + center1,
            cluster2_local + center2,
        ])

        # Build k-NN graph and compute Laplacian
        L = _build_knn_laplacian(X, k_neighbors)

        # Compute spectral embedding
        Phi, eigenvalues = _compute_spectral_embedding(L, n_eigenvectors)

        frames.append(SpectralFrame(Phi=Phi, eigenvalues=eigenvalues, t=t))

    return frames, labels


def generate_evolving_graph(
    n_nodes_ring1: int = 30,
    n_nodes_ring2: int = 20,
    coupling_start: float = 0.0,
    coupling_end: float = 1.0,
    coupling_schedule: Literal["linear", "exponential", "step"] = "linear",
    n_eigenvectors: int = 6,
    n_timesteps: int = 50,
) -> Tuple[List[SpectralFrame], NDArray]:
    """
    Generate temporal sequence of two ring graphs coupling.

    At t=0, two disconnected rings with their own spectra.
    At t=1, strongly coupled rings with shared global modes.

    Spectral signature:
    - Disconnected: degenerate eigenvalues (two copies of ring spectrum)
    - Coupling creates eigenvalue crossings
    - New global modes emerge spanning both rings
    - Asymmetric ring sizes create clear crossing patterns

    Args:
        n_nodes_ring1: Nodes in first ring.
        n_nodes_ring2: Nodes in second ring (different size for crossings).
        coupling_start: Initial coupling weight.
        coupling_end: Final coupling weight.
        coupling_schedule: How coupling increases ("linear", "exponential", "step").
        n_eigenvectors: Number of spectral dimensions.
        n_timesteps: Number of temporal snapshots.

    Returns:
        frames: List of SpectralFrame for each timestep.
        labels: Ring membership array (n_total,).
    """
    n_total = n_nodes_ring1 + n_nodes_ring2
    labels = np.array([0] * n_nodes_ring1 + [1] * n_nodes_ring2)

    frames = []

    for t_idx in range(n_timesteps):
        t = t_idx / max(n_timesteps - 1, 1)

        # Compute coupling strength based on schedule
        if coupling_schedule == "linear":
            coupling = coupling_start + (coupling_end - coupling_start) * t
        elif coupling_schedule == "exponential":
            # Slow start, fast end
            coupling = coupling_start + (coupling_end - coupling_start) * (t ** 2)
        elif coupling_schedule == "step":
            # Sudden connection at t=0.5
            coupling = coupling_end if t >= 0.5 else coupling_start
        else:
            raise ValueError(f"Unknown coupling_schedule: {coupling_schedule}")

        # Build coupled ring Laplacian
        L = _build_coupled_rings_laplacian(n_nodes_ring1, n_nodes_ring2, coupling)

        # Compute spectral embedding
        Phi, eigenvalues = _compute_spectral_embedding(L, n_eigenvectors)

        frames.append(SpectralFrame(Phi=Phi, eigenvalues=eigenvalues, t=t))

    return frames, labels


def _build_knn_laplacian(
    X: NDArray,
    k: int,
) -> sparse.csr_matrix:
    """
    Build symmetric normalized Laplacian from k-NN graph.

    Args:
        X: Data points (n, d).
        k: Number of neighbors.

    Returns:
        Normalized Laplacian matrix.
    """
    from scipy.spatial.distance import cdist

    n = X.shape[0]
    distances = cdist(X, X)

    # Find k nearest neighbors for each point
    rows, cols, data = [], [], []

    for i in range(n):
        # Get k nearest neighbors (excluding self)
        dists_i = distances[i].copy()
        dists_i[i] = np.inf
        neighbors = np.argsort(dists_i)[:k]

        for j in neighbors:
            rows.append(i)
            cols.append(j)
            # Use Gaussian kernel weight
            sigma = np.median(distances[i, neighbors])
            if sigma > 0:
                weight = np.exp(-distances[i, j]**2 / (2 * sigma**2))
            else:
                weight = 1.0
            data.append(weight)

    # Make symmetric
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2

    # Compute normalized Laplacian
    degrees = np.asarray(W.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-10)
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

    L = sparse.eye(n) - d_inv_sqrt @ W @ d_inv_sqrt

    return L.tocsr()


def _build_coupled_rings_laplacian(
    n1: int,
    n2: int,
    coupling: float,
) -> sparse.csr_matrix:
    """
    Build Laplacian for two coupled ring graphs.

    Args:
        n1: Nodes in first ring.
        n2: Nodes in second ring.
        coupling: Weight of coupling edge(s) between rings.

    Returns:
        Normalized Laplacian matrix.
    """
    total_n = n1 + n2
    rows, cols, data = [], [], []

    # First ring edges
    for i in range(n1):
        # Connect to neighbors in ring
        left = (i - 1) % n1
        right = (i + 1) % n1
        rows.extend([i, i])
        cols.extend([left, right])
        data.extend([1.0, 1.0])

    # Second ring edges
    for i in range(n2):
        node_idx = n1 + i
        left = n1 + (i - 1) % n2
        right = n1 + (i + 1) % n2
        rows.extend([node_idx, node_idx])
        cols.extend([left, right])
        data.extend([1.0, 1.0])

    # Coupling edge(s) between rings
    if coupling > 0:
        # Connect node 0 of ring 1 to node 0 of ring 2
        rows.extend([0, n1])
        cols.extend([n1, 0])
        data.extend([coupling, coupling])

        # Optional: add second coupling for stronger connection
        if n1 > 1 and n2 > 1:
            mid1 = n1 // 2
            mid2 = n1 + n2 // 2
            rows.extend([mid1, mid2])
            cols.extend([mid2, mid1])
            data.extend([coupling * 0.5, coupling * 0.5])

    W = sparse.csr_matrix((data, (rows, cols)), shape=(total_n, total_n))

    # Compute normalized Laplacian
    degrees = np.asarray(W.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-10)
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

    L = sparse.eye(total_n) - d_inv_sqrt @ W @ d_inv_sqrt

    return L.tocsr()


def _compute_spectral_embedding(
    L: sparse.csr_matrix,
    k: int,
) -> Tuple[NDArray, NDArray]:
    """
    Compute k smallest eigenvectors of Laplacian (excluding trivial).

    Args:
        L: Laplacian matrix.
        k: Number of eigenvectors.

    Returns:
        Phi: Eigenvector matrix (n, k).
        eigenvalues: Eigenvalue array (k,).
    """
    n = L.shape[0]
    n_components = min(k + 1, n - 1)

    try:
        eigenvalues, eigenvectors = eigsh(
            L.astype(np.float64),
            k=n_components,
            which="SM",
            sigma=1e-10,
            maxiter=1000,
            tol=1e-8,
        )
    except Exception:
        # Fallback to dense computation for small matrices
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip trivial eigenvector (constant, eigenvalue ~0)
    eigenvalues = eigenvalues[1:k+1]
    eigenvectors = eigenvectors[:, 1:k+1]

    # Ensure consistent sign convention (max entry positive)
    for j in range(eigenvectors.shape[1]):
        if np.abs(eigenvectors[:, j]).max() > 1e-10:
            max_idx = np.argmax(np.abs(eigenvectors[:, j]))
            if eigenvectors[max_idx, j] < 0:
                eigenvectors[:, j] *= -1

    return eigenvectors, eigenvalues


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test merging clusters
    print("Testing merging clusters...")
    frames_mc, labels_mc = generate_merging_clusters(n_timesteps=20)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, t_idx in enumerate([0, 10, 19]):
        frame = frames_mc[t_idx]
        ax = axes[i]
        colors = ["blue" if l == 0 else "red" for l in labels_mc]
        ax.scatter(frame.Phi[:, 0], frame.Phi[:, 1], c=colors, alpha=0.6)
        ax.set_title(f"t = {frame.t:.2f}")
        ax.set_xlabel("$\\phi_1$")
        ax.set_ylabel("$\\phi_2$")

    plt.suptitle("Merging Clusters - Spectral Embedding")
    plt.tight_layout()
    plt.show()

    # Test evolving graph
    print("Testing evolving graph...")
    frames_eg, labels_eg = generate_evolving_graph(n_timesteps=20)

    # Plot eigenvalue evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    n_eigs = frames_eg[0].eigenvalues.shape[0]
    t_values = [f.t for f in frames_eg]

    for j in range(n_eigs):
        eig_j = [f.eigenvalues[j] for f in frames_eg]
        ax.plot(t_values, eig_j, label=f"$\\lambda_{{{j+1}}}$")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Evolving Graph - Eigenvalue Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
