"""
Eigenvector heatmap panel for temporal spectral visualization.

Shows Phi_t (nodes x spectral dimensions) as a heatmap at the
current timestep, with optional side-by-side raw vs aligned comparison.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.styles import VizStyle, apply_style


def plot_eigenvector_heatmap(
    Phi: NDArray,
    Phi_aligned: Optional[NDArray] = None,
    show_comparison: bool = False,
    sign_flips: Optional[NDArray] = None,
    permutation: Optional[NDArray] = None,
    style: Optional[VizStyle] = None,
    max_nodes_display: int = 100,
) -> plt.Figure:
    """
    Plot eigenvector matrix as heatmap.

    Args:
        Phi: Eigenvector matrix at current time, shape (n, k).
        Phi_aligned: Aligned eigenvectors. If None, uses Phi.
        show_comparison: If True, show side-by-side raw vs aligned.
        sign_flips: Optional array of signs applied during alignment.
        permutation: Optional permutation applied during alignment.
        style: Visualization style configuration.
        max_nodes_display: Subsample if more nodes than this.

    Returns:
        Matplotlib figure with eigenvector heatmap(s).
    """
    if style is None:
        style = VizStyle()

    apply_style(style)

    n_nodes, k = Phi.shape

    # Subsample for display if too many nodes
    if n_nodes > max_nodes_display:
        indices = np.linspace(0, n_nodes - 1, max_nodes_display, dtype=int)
        Phi_display = Phi[indices]
        if Phi_aligned is not None:
            Phi_aligned_display = Phi_aligned[indices]
        else:
            Phi_aligned_display = Phi_display
        ylabel = f"Node Index (subsampled from {n_nodes})"
    else:
        Phi_display = Phi
        Phi_aligned_display = Phi_aligned if Phi_aligned is not None else Phi
        ylabel = "Node Index"

    # Determine color scale
    vmax = max(np.abs(Phi_display).max(), 0.01)
    if Phi_aligned is not None:
        vmax = max(vmax, np.abs(Phi_aligned_display).max())
    vmin = -vmax

    if show_comparison and Phi_aligned is not None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(style.fig_width * 1.2, style.fig_height * 0.6),
            sharey=True,
        )

        # Raw eigenvectors
        im1 = ax1.imshow(
            Phi_display,
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax1.set_xlabel("Eigenvector Index", fontsize=style.label_size)
        ax1.set_ylabel(ylabel, fontsize=style.label_size)
        ax1.set_title("Raw $\\Phi_t$", fontsize=style.title_size)
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f"$\\phi_{{{j+1}}}$" for j in range(k)])

        # Aligned eigenvectors
        im2 = ax2.imshow(
            Phi_aligned_display,
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax2.set_xlabel("Eigenvector Index", fontsize=style.label_size)
        ax2.set_title("Aligned $\\Phi_t$", fontsize=style.title_size)
        ax2.set_xticks(range(k))
        ax2.set_xticklabels([f"$\\phi_{{{j+1}}}$" for j in range(k)])

        # Add colorbar
        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Value")

        # Annotate sign flips and permutations
        if sign_flips is not None:
            flip_str = ", ".join([f"{'+' if s > 0 else '-'}" for s in sign_flips])
            ax2.set_xlabel(
                f"Eigenvector Index\nSigns: [{flip_str}]",
                fontsize=style.label_size,
            )

    else:
        # Single heatmap
        display_data = Phi_aligned_display if Phi_aligned is not None else Phi_display

        fig, ax = plt.subplots(
            figsize=(style.fig_width * 0.7, style.fig_height * 0.6)
        )

        im = ax.imshow(
            display_data,
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("Eigenvector Index", fontsize=style.label_size)
        ax.set_ylabel(ylabel, fontsize=style.label_size)
        ax.set_title("Eigenvector Matrix $\\Phi_t$", fontsize=style.title_size)
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"$\\phi_{{{j+1}}}$" for j in range(k)])

        fig.colorbar(im, ax=ax, shrink=0.8, label="Value")

    fig.tight_layout()
    return fig


def plot_eigenvector_evolution(
    Phi_sequence: list,
    eigenvector_idx: int = 0,
    style: Optional[VizStyle] = None,
    max_nodes_display: int = 50,
) -> plt.Figure:
    """
    Plot how a single eigenvector evolves over time.

    Args:
        Phi_sequence: List of eigenvector matrices over time.
        eigenvector_idx: Which eigenvector to show (0-indexed).
        style: Visualization style.
        max_nodes_display: Subsample nodes if needed.

    Returns:
        Matplotlib figure showing eigenvector evolution.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)

    n_timesteps = len(Phi_sequence)
    n_nodes = Phi_sequence[0].shape[0]

    # Subsample nodes if needed
    if n_nodes > max_nodes_display:
        indices = np.linspace(0, n_nodes - 1, max_nodes_display, dtype=int)
    else:
        indices = np.arange(n_nodes)

    # Extract single eigenvector over time: (n_timesteps, n_nodes_display)
    evolution = np.array([
        Phi[indices, eigenvector_idx] for Phi in Phi_sequence
    ])

    vmax = np.abs(evolution).max()
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(style.fig_width * 0.8, style.fig_height * 0.5))

    im = ax.imshow(
        evolution.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[0, 1, n_nodes if n_nodes <= max_nodes_display else max_nodes_display, 0],
    )

    ax.set_xlabel("Time $t$", fontsize=style.label_size)
    ax.set_ylabel("Node Index", fontsize=style.label_size)
    ax.set_title(f"Evolution of $\\phi_{{{eigenvector_idx + 1}}}$", fontsize=style.title_size)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Value")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Test with mock data
    np.random.seed(42)
    n_nodes = 50
    k = 6

    # Create orthonormal eigenvectors
    random_matrix = np.random.randn(n_nodes, k)
    Phi, _ = np.linalg.qr(random_matrix)

    # Create "aligned" version with sign flips
    sign_flips = np.array([1, -1, 1, -1, 1, -1])
    Phi_aligned = Phi * sign_flips

    # Test single heatmap
    fig1 = plot_eigenvector_heatmap(Phi)
    plt.show()

    # Test comparison view
    fig2 = plot_eigenvector_heatmap(
        Phi * np.array([1, -1, 1, 1, -1, 1]),  # Raw with random signs
        Phi_aligned=Phi,                        # Aligned
        show_comparison=True,
        sign_flips=sign_flips,
    )
    plt.show()
