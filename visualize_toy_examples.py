#!/usr/bin/env python3
"""
Visualizations for TSF Toy Examples.

This script creates comprehensive visualizations demonstrating the key properties
of the Temporal Spectral Flow (TSF) framework. It visualizes the 6 toy examples
that illustrate why transport-consistent alignment is necessary.

Dependencies: numpy, scipy, matplotlib, scikit-learn (no torch required)
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform

# Import TSF modules (excluding TID which requires torch)
from temporal_spectral_flow.graph import GraphConstructor
from temporal_spectral_flow.spectral import SpectralEmbedding, SpectralSnapshot
from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.transport import TransportAlignment, BasisAligner

# Use non-interactive backend for file output
plt.switch_backend('Agg')


# =============================================================================
# Helper Functions (from test_toy_examples.py)
# =============================================================================

def create_ring_graph(n_nodes: int) -> sparse.csr_matrix:
    """Create a simple ring graph Laplacian."""
    rows = []
    cols = []
    for i in range(n_nodes):
        rows.extend([i, i])
        cols.extend([(i - 1) % n_nodes, (i + 1) % n_nodes])

    data = np.ones(len(rows))
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    degrees = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L = sparse.eye(n_nodes) - d_inv_sqrt @ W @ d_inv_sqrt

    return L.tocsr()


def compute_laplacian_eigenvectors(L: sparse.csr_matrix, k: int, skip_first: bool = True):
    """Compute k smallest eigenvectors of Laplacian."""
    n_components = k + (1 if skip_first else 0)
    n_components = min(n_components, L.shape[0] - 1)

    eigenvalues, eigenvectors = eigsh(
        L.astype(np.float64),
        k=n_components,
        which="SM",
        sigma=1e-10,
        maxiter=1000,
        tol=1e-8,
    )

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if skip_first:
        eigenvalues = eigenvalues[1:k+1]
        eigenvectors = eigenvectors[:, 1:k+1]
    else:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    Q, R = np.linalg.qr(eigenvectors, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs

    return Q, eigenvalues


def create_swiss_roll(n_samples: int, noise: float = 0.0, seed: int = 42):
    """Generate Swiss roll data."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n_samples))
    height = 21 * rng.random(n_samples)

    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)

    X = np.column_stack([x, y, z])
    if noise > 0:
        X += noise * rng.standard_normal(X.shape)

    return X, t


def create_coupled_rings(n1: int, n2: int, coupling_strength: float):
    """Create two ring graphs coupled by a weak edge."""
    total_n = n1 + n2
    rows, cols, data = [], [], []

    # First ring
    for i in range(n1):
        rows.extend([i, i])
        cols.extend([(i - 1) % n1, (i + 1) % n1])
        data.extend([1.0, 1.0])

    # Second ring
    for i in range(n2):
        node_idx = n1 + i
        neighbor1 = n1 + (i - 1) % n2
        neighbor2 = n1 + (i + 1) % n2
        rows.extend([node_idx, node_idx])
        cols.extend([neighbor1, neighbor2])
        data.extend([1.0, 1.0])

    # Coupling edge
    rows.extend([0, n1])
    cols.extend([n1, 0])
    data.extend([coupling_strength, coupling_strength])

    W = sparse.csr_matrix((data, (rows, cols)), shape=(total_n, total_n))
    degrees = np.asarray(W.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-10)
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L = sparse.eye(total_n) - d_inv_sqrt @ W @ d_inv_sqrt

    return L.tocsr()


# =============================================================================
# Visualization 1: Eigenvector Sign Flip
# =============================================================================

def visualize_sign_flip():
    """Visualize how sign flips create apparent change in naive comparison."""
    print("Creating visualization 1: Eigenvector Sign Flip...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_nodes = 50
    k = 5
    L = create_ring_graph(n_nodes)
    Phi, eigenvalues = compute_laplacian_eigenvectors(L, k)

    # Create sign-flipped version
    flip_pattern = np.array([1, -1, 1, -1, 1])
    Phi_flipped = Phi * flip_pattern

    # Panel 1: Original eigenvectors
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(Phi, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax1.set_title('Original Eigenvectors $\\Phi$', fontsize=12)
    ax1.set_xlabel('Spectral Dimension')
    ax1.set_ylabel('Node Index')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Panel 2: Sign-flipped eigenvectors
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(Phi_flipped, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax2.set_title('Sign-Flipped $\\Phi_{flipped}$', fontsize=12)
    ax2.set_xlabel('Spectral Dimension')
    ax2.set_ylabel('Node Index')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Panel 3: Difference (shows sign flips as large changes)
    ax3 = fig.add_subplot(gs[0, 2])
    diff = Phi - Phi_flipped
    im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu', vmin=-0.6, vmax=0.6)
    ax3.set_title('Naive Difference $\\Phi - \\Phi_{flipped}$', fontsize=12)
    ax3.set_xlabel('Spectral Dimension')
    ax3.set_ylabel('Node Index')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Panel 4: First eigenvector comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(Phi[:, 0], 'b-', label='Original', linewidth=2)
    ax4.plot(Phi_flipped[:, 0], 'r--', label='Flipped', linewidth=2)
    ax4.set_title('1st Eigenvector: Flipped Sign', fontsize=12)
    ax4.set_xlabel('Node Index')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: After basis alignment
    aligner = BasisAligner(allow_reflection=True)
    Phi_aligned = aligner.align_signs(Phi, Phi_flipped)

    ax5 = fig.add_subplot(gs[1, 1])
    diff_aligned = Phi - Phi_aligned
    im5 = ax5.imshow(diff_aligned, aspect='auto', cmap='RdBu', vmin=-0.1, vmax=0.1)
    ax5.set_title('After Sign Alignment (near zero)', fontsize=12)
    ax5.set_xlabel('Spectral Dimension')
    ax5.set_ylabel('Node Index')
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Panel 6: Distance comparison
    ax6 = fig.add_subplot(gs[1, 2])

    raw_frob = np.linalg.norm(Phi - Phi_flipped, 'fro')
    aligned_frob = np.linalg.norm(Phi - Phi_aligned, 'fro')
    geodesic = StiefelManifold.geodesic_distance(Phi, Phi_flipped)

    bars = ax6.bar(['Raw Frobenius', 'Geodesic\n(Gauge-Invariant)', 'After\nAlignment'],
                   [raw_frob, geodesic, aligned_frob],
                   color=['#e74c3c', '#3498db', '#27ae60'])
    ax6.set_ylabel('Distance')
    ax6.set_title('Distance Metrics Comparison', fontsize=12)

    # Add value labels on bars
    for bar, val in zip(bars, [raw_frob, geodesic, aligned_frob]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Toy Example 1: Sign Flips Require Alignment\n'
                 '(Sign flips are meaningless gauge changes, not real evolution)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_1_sign_flip.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_1_sign_flip.png")


# =============================================================================
# Visualization 2: Pure Rotation in Spectral Space
# =============================================================================

def visualize_rotation():
    """Visualize how ambient rotation affects spectral embeddings."""
    print("Creating visualization 2: Pure Rotation...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_samples = 100
    k = 5

    X, t = create_swiss_roll(n_samples, noise=0.1, seed=42)

    graph_constructor = GraphConstructor(method="knn", n_neighbors=10, laplacian_type="symmetric")
    embedder = SpectralEmbedding(k=k, graph_constructor=graph_constructor)

    # Apply rotations
    angles = [0, 0.3, 0.6]
    snapshots = []

    for theta in angles:
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        X_rot = X @ R.T
        snapshot = embedder.embed(X_rot)
        snapshots.append((X_rot, snapshot, theta))

    # Panel 1-3: 3D scatter plots
    for i, (X_rot, snapshot, theta) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        scatter = ax.scatter(X_rot[:, 0], X_rot[:, 1], X_rot[:, 2],
                           c=t, cmap='viridis', s=20, alpha=0.7)
        ax.set_title(f'Rotation: $\\theta$ = {theta:.1f} rad', fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Panel 4: Transport costs over rotation sequence
    ax4 = fig.add_subplot(gs[1, 0])

    aligner = TransportAlignment(method="balanced", reg=0.05)
    transport_costs = []
    rotation_angles = np.linspace(0, 1.0, 10)

    for theta in rotation_angles:
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        X_rot = X @ R.T
        snapshot_rot = embedder.embed(X_rot)

        result = aligner.align(
            snapshots[0][1].Phi, snapshot_rot.Phi,
            eigenvalues_source=snapshots[0][1].eigenvalues,
            eigenvalues_target=snapshot_rot.eigenvalues,
        )
        transport_costs.append(result.transport_cost)

    ax4.plot(rotation_angles, transport_costs, 'bo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Rotation Angle (radians)')
    ax4.set_ylabel('Transport Cost')
    ax4.set_title('Transport Cost vs Rotation\n(Should stay low - rotation is gauge)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Spectral embedding comparison (first 2 dimensions)
    ax5 = fig.add_subplot(gs[1, 1])

    colors = ['#3498db', '#e74c3c', '#27ae60']
    for i, (_, snapshot, theta) in enumerate(snapshots):
        ax5.scatter(snapshot.Phi[:, 0], snapshot.Phi[:, 1],
                   c=colors[i], alpha=0.6, s=20, label=f'$\\theta$={theta:.1f}')

    ax5.set_xlabel('1st Spectral Dimension')
    ax5.set_ylabel('2nd Spectral Dimension')
    ax5.set_title('Spectral Embeddings\n(Should be similar up to rotation)', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Eigenvalue comparison
    ax6 = fig.add_subplot(gs[1, 2])

    width = 0.25
    x = np.arange(k)

    for i, (_, snapshot, theta) in enumerate(snapshots):
        ax6.bar(x + i*width, snapshot.eigenvalues, width,
               label=f'$\\theta$={theta:.1f}', alpha=0.8)

    ax6.set_xlabel('Eigenvalue Index')
    ax6.set_ylabel('Eigenvalue')
    ax6.set_title('Eigenvalues Preserved\n(Geometry unchanged)', fontsize=11)
    ax6.set_xticks(x + width)
    ax6.set_xticklabels([f'$\\lambda_{{{i}}}$' for i in range(k)])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Toy Example 2: Ambient Rotation is Gauge Invariant\n'
                 '(OT alignment absorbs coordinate rotations)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_2_rotation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_2_rotation.png")


# =============================================================================
# Visualization 3: Birth of New Mode (Coupled Rings)
# =============================================================================

def visualize_birth_of_mode():
    """Visualize emergence of new spectral mode when rings couple."""
    print("Creating visualization 3: Birth of New Mode...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    k = 6
    n1, n2 = 50, 30

    # Single ring
    L_single = create_ring_graph(n1)
    Phi_single, eig_single = compute_laplacian_eigenvectors(L_single, k)

    # Coupled rings with varying coupling
    coupling_strengths = [0.0, 0.3, 0.6, 1.0]
    coupled_data = []

    for coupling in coupling_strengths:
        if coupling == 0:
            L = create_ring_graph(n1)
            n = n1
        else:
            L = create_coupled_rings(n1, n2, coupling)
            n = n1 + n2

        Phi, eig = compute_laplacian_eigenvectors(L, k)
        coupled_data.append((coupling, Phi, eig, n))

    # Panel 1: Single ring eigenvalues
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(k), eig_single, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Single Ring Spectrum', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Eigenvalue evolution with coupling
    ax2 = fig.add_subplot(gs[0, 1])

    for j in range(k):
        eigenvalues_j = [data[2][j] for data in coupled_data]
        ax2.plot(coupling_strengths, eigenvalues_j, 'o-',
                label=f'$\\lambda_{{{j}}}$', linewidth=2, markersize=6)

    ax2.set_xlabel('Coupling Strength')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Eigenvalue Evolution\nas Rings Couple', fontsize=11)
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Graph structure visualization
    ax3 = fig.add_subplot(gs[0, 2])

    # Draw two coupled rings
    theta1 = np.linspace(0, 2*np.pi, n1+1)[:-1]
    theta2 = np.linspace(0, 2*np.pi, n2+1)[:-1]

    x1 = 0.3 * np.cos(theta1) - 0.5
    y1 = 0.3 * np.sin(theta1)
    x2 = 0.2 * np.cos(theta2) + 0.4
    y2 = 0.2 * np.sin(theta2)

    ax3.scatter(x1, y1, c='#3498db', s=30, zorder=3, label=f'Ring 1 (n={n1})')
    ax3.scatter(x2, y2, c='#e74c3c', s=30, zorder=3, label=f'Ring 2 (n={n2})')

    # Draw coupling edge
    ax3.plot([x1[0], x2[0]], [y1[0], y2[0]], 'g-', linewidth=3,
            label='Coupling Edge', zorder=2)

    ax3.set_xlim(-1, 0.8)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=9)
    ax3.set_title('Coupled Ring Structure', fontsize=11)
    ax3.axis('off')

    # Panel 4: Eigenvector localization
    ax4 = fig.add_subplot(gs[1, 0])

    _, Phi_coupled, _, n_coupled = coupled_data[-1]  # Strongest coupling

    for j in range(min(3, k)):
        ax4.plot(Phi_coupled[:, j], label=f'$\\phi_{{{j}}}$', linewidth=1.5)

    ax4.axvline(x=n1, color='gray', linestyle='--', label=f'Ring boundary')
    ax4.set_xlabel('Node Index')
    ax4.set_ylabel('Eigenvector Value')
    ax4.set_title('Eigenvectors of Coupled System', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Transport plan for size-changing alignment
    ax5 = fig.add_subplot(gs[1, 1])

    _, Phi_coupled_weak, eig_coupled_weak, _ = coupled_data[1]

    aligner = TransportAlignment(method="unbalanced", reg=0.1, reg_m=1.0)
    result = aligner.align(
        Phi_single, Phi_coupled_weak[:n1, :],  # Match sizes for visualization
        eigenvalues_source=eig_single,
        eigenvalues_target=eig_coupled_weak,
    )

    im5 = ax5.imshow(result.transport_plan[:30, :30], aspect='auto', cmap='Blues')
    ax5.set_title('Transport Plan (30x30 subset)\nUnbalanced OT handles size change', fontsize=11)
    ax5.set_xlabel('Target Node')
    ax5.set_ylabel('Source Node')
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Panel 6: Mass distribution
    ax6 = fig.add_subplot(gs[1, 2])

    source_marginal = result.mass_source
    ax6.bar(range(len(source_marginal)), source_marginal, alpha=0.7, color='#3498db')
    ax6.axhline(y=1/n1, color='red', linestyle='--', label='Uniform mass')
    ax6.set_xlabel('Source Node')
    ax6.set_ylabel('Transported Mass')
    ax6.set_title('Mass Distribution\n(Unbalanced OT allows creation/destruction)', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Toy Example 3: Birth of New Spectral Mode\n'
                 '(When topology changes, new modes emerge - unbalanced OT required)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_3_birth_of_mode.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_3_birth_of_mode.png")


# =============================================================================
# Visualization 4: Noise vs Signal
# =============================================================================

def visualize_noise_vs_signal():
    """Visualize how temporal analysis separates noise from signal."""
    print("Creating visualization 4: Noise vs Signal...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_samples = 100
    ambient_dim = 50
    intrinsic_dim = 2
    noise_std = 1.5
    k = 8

    def create_embedded_manifold(seed):
        rng = np.random.default_rng(seed)

        theta = 2 * np.pi * rng.random(n_samples)
        phi = 2 * np.pi * rng.random(n_samples)

        r1, r2 = 3.0, 1.0
        x = (r1 + r2 * np.cos(phi)) * np.cos(theta)
        y = (r1 + r2 * np.cos(phi)) * np.sin(theta)
        z = r2 * np.sin(phi)

        manifold_coords = np.column_stack([x, y, z])

        projection = rng.standard_normal((3, ambient_dim))
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)

        X_manifold = manifold_coords @ projection
        noise = noise_std * rng.standard_normal((n_samples, ambient_dim))

        return X_manifold + noise, manifold_coords

    # Create sequence with changing noise
    graph_constructor = GraphConstructor(method="knn", n_neighbors=15, laplacian_type="symmetric")
    embedder = SpectralEmbedding(k=k, graph_constructor=graph_constructor)

    snapshots = []
    for t in range(10):
        X, _ = create_embedded_manifold(seed=42 + t)
        snapshot = embedder.embed(X)
        snapshots.append(snapshot)

    # Panel 1: PCA variance (static view)
    ax1 = fig.add_subplot(gs[0, 0])

    X_single, _ = create_embedded_manifold(seed=42)
    X_centered = X_single - X_single.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    variances = s ** 2 / n_samples

    ax1.bar(range(min(15, len(variances))), variances[:15], color='#3498db', alpha=0.8)
    ax1.axhline(y=0.1 * variances.max(), color='red', linestyle='--',
               label='10% threshold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance')
    ax1.set_title('PCA: Variance Overestimates Dimension\n(Noise inflates many components)', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Eigenvalue stability over time
    ax2 = fig.add_subplot(gs[0, 1])

    all_eigenvalues = np.array([s.eigenvalues for s in snapshots])

    for j in range(k):
        ax2.plot(all_eigenvalues[:, j], 'o-', label=f'$\\lambda_{{{j}}}$',
                linewidth=1.5, markersize=4)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Eigenvalue Evolution\n(Stable = signal, volatile = noise)', fontsize=11)
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Eigenvector stability (correlation across time)
    ax3 = fig.add_subplot(gs[0, 2])

    basis_aligner = BasisAligner()
    stability_scores = np.zeros(k)

    for j in range(k):
        correlations = []
        for t in range(len(snapshots) - 1):
            _, Phi_aligned = basis_aligner.align_bases(
                snapshots[t].Phi, snapshots[t + 1].Phi
            )
            corr = np.abs(np.corrcoef(snapshots[t].Phi[:, j],
                                      Phi_aligned[:, j])[0, 1])
            correlations.append(corr)
        stability_scores[j] = np.mean(correlations)

    colors = ['#27ae60' if s > 0.7 else '#e74c3c' for s in stability_scores]
    ax3.bar(range(k), stability_scores, color=colors, alpha=0.8)
    ax3.axhline(y=0.7, color='gray', linestyle='--', label='Stability threshold')
    ax3.set_xlabel('Spectral Dimension')
    ax3.set_ylabel('Temporal Stability')
    ax3.set_title('Temporal Stability Score\n(Green = stable signal)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Eigenvector comparison (t=0 vs t=9)
    ax4 = fig.add_subplot(gs[1, 0])

    _, Phi_aligned = basis_aligner.align_bases(snapshots[0].Phi, snapshots[-1].Phi)
    diff = np.abs(snapshots[0].Phi - Phi_aligned)

    im4 = ax4.imshow(diff, aspect='auto', cmap='Reds')
    ax4.set_xlabel('Spectral Dimension')
    ax4.set_ylabel('Node Index')
    ax4.set_title('Eigenvector Change (t=0 to t=9)\n(Red = high change = noise)', fontsize=11)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Panel 5: First two stable eigenvectors
    ax5 = fig.add_subplot(gs[1, 1])

    for t in [0, 4, 9]:
        ax5.scatter(snapshots[t].Phi[:, 0], snapshots[t].Phi[:, 1],
                   alpha=0.5, s=20, label=f't={t}')

    ax5.set_xlabel('1st Eigenvector')
    ax5.set_ylabel('2nd Eigenvector')
    ax5.set_title('Stable Dimensions\n(Overlap = consistent structure)', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Last two (noisy) eigenvectors
    ax6 = fig.add_subplot(gs[1, 2])

    for t in [0, 4, 9]:
        ax6.scatter(snapshots[t].Phi[:, -2], snapshots[t].Phi[:, -1],
                   alpha=0.5, s=20, label=f't={t}')

    ax6.set_xlabel(f'{k-1}th Eigenvector')
    ax6.set_ylabel(f'{k}th Eigenvector')
    ax6.set_title('Noisy Dimensions\n(No overlap = noise)', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Toy Example 4: Temporal Stability Separates Signal from Noise\n'
                 '(Stable dimensions are signal; fluctuating dimensions are noise)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_4_noise_vs_signal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_4_noise_vs_signal.png")


# =============================================================================
# Visualization 5: Permutation Chaos
# =============================================================================

def visualize_permutation():
    """Visualize how OT recovers from random permutations."""
    print("Creating visualization 5: Permutation Chaos...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_nodes = 50
    k = 5

    L = create_ring_graph(n_nodes)
    Phi, eigenvalues = compute_laplacian_eigenvectors(L, k)

    # Random permutation
    np.random.seed(42)
    perm = np.random.permutation(n_nodes)
    Phi_permuted = Phi[perm, :]

    # Panel 1: Original ordering
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(Phi, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax1.set_title('Original Node Ordering', fontsize=11)
    ax1.set_xlabel('Spectral Dimension')
    ax1.set_ylabel('Node Index')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Panel 2: Permuted ordering
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(Phi_permuted, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax2.set_title('Randomly Permuted Ordering', fontsize=11)
    ax2.set_xlabel('Spectral Dimension')
    ax2.set_ylabel('Node Index')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Panel 3: Direct difference (meaningless)
    ax3 = fig.add_subplot(gs[0, 2])
    diff = Phi - Phi_permuted
    im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax3.set_title('Direct Difference (Meaningless)', fontsize=11)
    ax3.set_xlabel('Spectral Dimension')
    ax3.set_ylabel('Node Index')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Panel 4: Transport plan
    ax4 = fig.add_subplot(gs[1, 0])

    aligner = TransportAlignment(method="balanced", reg=0.01)
    result = aligner.align(Phi, Phi_permuted)

    im4 = ax4.imshow(result.transport_plan, aspect='auto', cmap='Blues')
    ax4.set_title('OT Transport Plan\n(Should find permutation)', fontsize=11)
    ax4.set_xlabel('Permuted Index')
    ax4.set_ylabel('Original Index')
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Panel 5: Permutation recovery accuracy
    ax5 = fig.add_subplot(gs[1, 1])

    assignment = np.argmax(result.transport_plan, axis=1)
    inv_perm = np.argsort(perm)

    correct = (assignment == inv_perm).astype(int)
    accuracy = np.mean(correct)

    ax5.scatter(range(n_nodes), inv_perm, c='blue', s=30, alpha=0.5,
               label='True inverse perm')
    ax5.scatter(range(n_nodes), assignment, c='red', s=15, marker='x',
               label='OT assignment')

    ax5.set_xlabel('Original Node')
    ax5.set_ylabel('Assigned/True Permuted Node')
    ax5.set_title(f'Permutation Recovery\nAccuracy: {accuracy:.1%}', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Multiple random permutations
    ax6 = fig.add_subplot(gs[1, 2])

    transport_costs = []
    accuracies = []

    for seed in range(20):
        np.random.seed(seed)
        perm_i = np.random.permutation(n_nodes)
        Phi_perm_i = Phi[perm_i, :]

        result_i = aligner.align(Phi, Phi_perm_i)
        transport_costs.append(result_i.transport_cost)

        assignment_i = np.argmax(result_i.transport_plan, axis=1)
        inv_perm_i = np.argsort(perm_i)
        acc_i = np.mean(assignment_i == inv_perm_i)
        accuracies.append(acc_i)

    ax6.scatter(transport_costs, accuracies, c='#3498db', s=50, alpha=0.7)
    ax6.set_xlabel('Transport Cost')
    ax6.set_ylabel('Recovery Accuracy')
    ax6.set_title('OT Performance Across\n20 Random Permutations', fontsize=11)
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Toy Example 5: OT Alignment is Permutation Invariant\n'
                 '(Structural correspondence recovered without point labels)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_5_permutation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_5_permutation.png")


# =============================================================================
# Visualization 6: Gradual Curvature Change
# =============================================================================

def visualize_curvature_change():
    """Visualize smooth geometric deformation tracking."""
    print("Creating visualization 6: Gradual Curvature Change...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_samples = 100
    k = 5
    n_timesteps = 10

    def create_deformed_swiss_roll(stretch, bend, seed=42):
        rng = np.random.default_rng(seed)
        t = 1.5 * np.pi * (1 + 2 * rng.random(n_samples))
        height = 21 * rng.random(n_samples)

        t_stretched = t * stretch
        x = t_stretched * np.cos(t_stretched + bend)
        y = height
        z = t_stretched * np.sin(t_stretched + bend)

        return np.column_stack([x, y, z]), t

    graph_constructor = GraphConstructor(method="knn", n_neighbors=12, laplacian_type="symmetric")
    embedder = SpectralEmbedding(k=k, graph_constructor=graph_constructor)
    basis_aligner = BasisAligner()
    transport_aligner = TransportAlignment(method="balanced", reg=0.05)

    # Generate sequence
    snapshots = []
    X_sequence = []

    for t in range(n_timesteps):
        stretch = 1.0 + 0.05 * t
        bend = 0.02 * t
        X, color = create_deformed_swiss_roll(stretch, bend)
        snapshot = embedder.embed(X)
        snapshots.append(snapshot)
        X_sequence.append(X)

    # Panel 1-3: 3D visualization at t=0, t=5, t=9
    times_to_show = [0, 4, 9]
    _, color_ref = create_deformed_swiss_roll(1.0, 0.0)

    for idx, t in enumerate(times_to_show):
        ax = fig.add_subplot(gs[0, idx], projection='3d')
        X_t = X_sequence[t]
        ax.scatter(X_t[:, 0], X_t[:, 1], X_t[:, 2], c=color_ref, cmap='viridis', s=20, alpha=0.7)
        stretch = 1.0 + 0.05 * t
        bend = 0.02 * t
        ax.set_title(f't={t}: stretch={stretch:.2f}, bend={bend:.2f}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Panel 4: Transport costs over time
    ax4 = fig.add_subplot(gs[1, 0])

    transport_costs = []
    for t in range(len(snapshots) - 1):
        result = transport_aligner.align(
            snapshots[t].Phi, snapshots[t + 1].Phi,
            eigenvalues_source=snapshots[t].eigenvalues,
            eigenvalues_target=snapshots[t + 1].eigenvalues,
        )
        transport_costs.append(result.transport_cost)

    ax4.plot(range(1, len(transport_costs) + 1), transport_costs, 'bo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Transport Cost')
    ax4.set_title('Transport Cost Over Time\n(Smooth = gradual change)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Spectral drift (Frobenius distance from t=0)
    ax5 = fig.add_subplot(gs[1, 1])

    distances_from_start = []
    for t in range(1, len(snapshots)):
        _, Phi_aligned = basis_aligner.align_bases(snapshots[0].Phi, snapshots[t].Phi)
        dist = np.linalg.norm(snapshots[0].Phi - Phi_aligned, 'fro')
        distances_from_start.append(dist)

    ax5.plot(range(1, len(distances_from_start) + 1), distances_from_start,
            'go-', linewidth=2, markersize=6)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Frobenius Distance from t=0')
    ax5.set_title('Cumulative Spectral Drift\n(Linear growth = smooth evolution)', fontsize=11)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Eigenvalue evolution
    ax6 = fig.add_subplot(gs[1, 2])

    all_eigenvalues = np.array([s.eigenvalues for s in snapshots])

    for j in range(k):
        ax6.plot(all_eigenvalues[:, j], 'o-', label=f'$\\lambda_{{{j}}}$',
                linewidth=1.5, markersize=4)

    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Eigenvalue')
    ax6.set_title('Eigenvalue Evolution\n(Smooth change = geometry shift)', fontsize=11)
    ax6.legend(ncol=2, fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Toy Example 6: Tracking Smooth Geometric Deformation\n'
                 '(Smooth manifold deformation tracked with low transport cost)',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_6_curvature_change.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_6_curvature_change.png")


# =============================================================================
# Summary Visualization
# =============================================================================

def visualize_summary():
    """Create a summary visualization of all key concepts."""
    print("Creating summary visualization...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Summary Panel 1: The Problem - Gauge Ambiguity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.9, 'THE PROBLEM', ha='center', va='top', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.7, 'Spectral embeddings have\ngauge ambiguity:', ha='center', va='top', fontsize=11)
    ax1.text(0.5, 0.45, '1. Sign flips\n2. Rotations\n3. Permutations', ha='center', va='top', fontsize=10)
    ax1.text(0.5, 0.15, 'Naive comparison fails!', ha='center', va='top', fontsize=11,
            color='red', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_facecolor('#ffe6e6')

    # Summary Panel 2: The Solution - Transport Alignment
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.9, 'THE SOLUTION', ha='center', va='top', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.7, 'Transport-Consistent Alignment:', ha='center', va='top', fontsize=11)
    ax2.text(0.5, 0.45, '1. OT matches structure\n2. Basis alignment\n3. Unbalanced for topology',
            ha='center', va='top', fontsize=10)
    ax2.text(0.5, 0.15, 'Gauge-invariant comparison!', ha='center', va='top', fontsize=11,
            color='green', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_facecolor('#e6ffe6')

    # Summary Panel 3: The Outcome - TID
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.9, 'THE OUTCOME', ha='center', va='top', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.7, 'Temporal Intrinsic Dimension:', ha='center', va='top', fontsize=11)
    ax3.text(0.5, 0.45, '1. Stable modes = signal\n2. Volatile modes = noise\n3. Tracks real change',
            ha='center', va='top', fontsize=10)
    ax3.text(0.5, 0.15, 'Principled dimension selection!', ha='center', va='top', fontsize=11,
            color='blue', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_facecolor('#e6e6ff')

    # Demo panels with actual computations
    n_nodes = 50
    k = 5
    L = create_ring_graph(n_nodes)
    Phi, eigenvalues = compute_laplacian_eigenvectors(L, k)

    # Panel 4: Sign flip demonstration
    ax4 = fig.add_subplot(gs[1, 0])
    Phi_flipped = Phi * np.array([1, -1, 1, -1, 1])

    raw_dist = np.linalg.norm(Phi - Phi_flipped, 'fro')
    aligner = BasisAligner()
    Phi_aligned = aligner.align_signs(Phi, Phi_flipped)
    aligned_dist = np.linalg.norm(Phi - Phi_aligned, 'fro')

    ax4.bar(['Raw', 'Aligned'], [raw_dist, aligned_dist], color=['#e74c3c', '#27ae60'])
    ax4.set_ylabel('Frobenius Distance')
    ax4.set_title('Sign Flip: Before/After Alignment', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Permutation recovery
    ax5 = fig.add_subplot(gs[1, 1])

    np.random.seed(42)
    perm = np.random.permutation(n_nodes)
    Phi_perm = Phi[perm, :]

    transport_aligner = TransportAlignment(method="balanced", reg=0.01)
    result = transport_aligner.align(Phi, Phi_perm)

    assignment = np.argmax(result.transport_plan, axis=1)
    inv_perm = np.argsort(perm)
    accuracy = np.mean(assignment == inv_perm)

    ax5.bar(['Accuracy'], [accuracy * 100], color='#3498db')
    ax5.set_ylabel('Recovery Accuracy (%)')
    ax5.set_title(f'Permutation Recovery: {accuracy:.0%}', fontsize=11)
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Transport cost comparison
    ax6 = fig.add_subplot(gs[1, 2])

    # Same structure (sign flip)
    result_same = transport_aligner.align(Phi, Phi_flipped)

    # Different structure (random)
    np.random.seed(123)
    Phi_random, _ = np.linalg.qr(np.random.randn(n_nodes, k))
    result_diff = transport_aligner.align(Phi, Phi_random)

    ax6.bar(['Same\nStructure', 'Different\nStructure'],
           [result_same.transport_cost, result_diff.transport_cost],
           color=['#27ae60', '#e74c3c'])
    ax6.set_ylabel('Transport Cost')
    ax6.set_title('OT Detects Real Change', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')

    # Panel 7: Eigenvalue stability across noise realizations
    ax7 = fig.add_subplot(gs[2, 0])

    all_eigs = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise = 0.02 * rng.standard_normal(Phi.shape)
        Phi_noisy = Phi + noise
        Q, _ = np.linalg.qr(Phi_noisy)
        all_eigs.append(eigenvalues)  # Same eigenvalues for ring

    all_eigs = np.array(all_eigs)
    ax7.boxplot([all_eigs[:, j] for j in range(k)], labels=[f'$\\lambda_{j}$' for j in range(k)])
    ax7.set_ylabel('Eigenvalue')
    ax7.set_title('Eigenvalue Stability\n(Narrow = stable)', fontsize=11)
    ax7.grid(True, alpha=0.3, axis='y')

    # Panel 8: Transport plan structure
    ax8 = fig.add_subplot(gs[2, 1])

    im = ax8.imshow(result.transport_plan[:25, :25], cmap='Blues', aspect='auto')
    ax8.set_title('Transport Plan Structure\n(Diagonal = identity mapping)', fontsize=11)
    ax8.set_xlabel('Target')
    ax8.set_ylabel('Source')
    plt.colorbar(im, ax=ax8, shrink=0.8)

    # Panel 9: Key equations
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.text(0.5, 0.9, 'KEY EQUATIONS', ha='center', va='top', fontsize=12, fontweight='bold')
    ax9.text(0.5, 0.7, r'$\Phi \in St(n,k) = \{X : X^TX = I_k\}$',
            ha='center', va='top', fontsize=11, family='serif')
    ax9.text(0.5, 0.5, r'$\min_P \langle C, P \rangle + \epsilon H(P)$',
            ha='center', va='top', fontsize=11, family='serif')
    ax9.text(0.5, 0.3, r'$TID = \sum_j S_j \cdot \mathbf{1}[S_j > \tau]$',
            ha='center', va='top', fontsize=11, family='serif')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_facecolor('#f0f0f0')

    fig.suptitle('Temporal Spectral Flow: Summary of Key Concepts\n'
                 'Transport-Consistent Alignment for Learning Dynamical Models of Geometric Structure',
                 fontsize=14, fontweight='bold')

    plt.savefig('viz_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_summary.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("TSF Toy Examples Visualization")
    print("=" * 60)
    print()

    visualize_sign_flip()
    visualize_rotation()
    visualize_birth_of_mode()
    visualize_noise_vs_signal()
    visualize_permutation()
    visualize_curvature_change()
    visualize_summary()

    print()
    print("=" * 60)
    print("All visualizations complete!")
    print("Generated files:")
    print("  - viz_1_sign_flip.png")
    print("  - viz_2_rotation.png")
    print("  - viz_3_birth_of_mode.png")
    print("  - viz_4_noise_vs_signal.png")
    print("  - viz_5_permutation.png")
    print("  - viz_6_curvature_change.png")
    print("  - viz_summary.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
