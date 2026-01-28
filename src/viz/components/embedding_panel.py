"""
Spectral embedding scatter panel for temporal visualization.

Shows 2D scatter plot using first two eigenvectors as coordinates,
with particle trails showing temporal evolution.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.styles import VizStyle, ColorPalette, apply_style, get_time_colors


def plot_spectral_embedding(
    Phi_sequence: List[NDArray],
    current_idx: int,
    labels: Optional[NDArray] = None,
    trail_length: int = 5,
    show_velocity: bool = False,
    velocity: Optional[NDArray] = None,
    dims: tuple = (0, 1),
    style: Optional[VizStyle] = None,
) -> plt.Figure:
    """
    Plot 2D spectral embedding with temporal trails.

    Args:
        Phi_sequence: List of eigenvector matrices, shape (n, k) each.
        current_idx: Current timestep index.
        labels: Optional cluster/component labels for coloring, shape (n,).
        trail_length: Number of past positions to show as trails.
        show_velocity: If True, show velocity arrows.
        velocity: Velocity vectors at current time, shape (n, k).
        dims: Which eigenvector dimensions to use for x, y (default: 0, 1).
        style: Visualization style configuration.

    Returns:
        Matplotlib figure with 2D spectral embedding.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    n_timesteps = len(Phi_sequence)
    n_nodes = Phi_sequence[0].shape[0]
    dim_x, dim_y = dims

    # Get current embedding
    current_idx = min(current_idx, n_timesteps - 1)
    Phi_current = Phi_sequence[current_idx]
    x_current = Phi_current[:, dim_x]
    y_current = Phi_current[:, dim_y]

    fig, ax = plt.subplots(figsize=(style.fig_width * 0.8, style.fig_height * 0.7))

    # Determine coloring
    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 2)))
        point_colors = cluster_colors[labels.astype(int) % len(cluster_colors)]
    else:
        point_colors = palette.source

    # Draw trails
    if trail_length > 0 and current_idx > 0:
        trail_start = max(0, current_idx - trail_length)
        trail_alphas = np.linspace(0.1, 0.5, current_idx - trail_start)

        for i in range(n_nodes):
            # Extract trail for this node
            trail_x = [Phi_sequence[t][i, dim_x] for t in range(trail_start, current_idx + 1)]
            trail_y = [Phi_sequence[t][i, dim_y] for t in range(trail_start, current_idx + 1)]

            if len(trail_x) > 1:
                points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Color by cluster if available
                if labels is not None:
                    trail_color = cluster_colors[int(labels[i]) % len(cluster_colors)]
                else:
                    trail_color = palette.trajectory

                lc = LineCollection(
                    segments,
                    colors=[(*trail_color[:3], a) for a in trail_alphas],
                    linewidths=1.5,
                )
                ax.add_collection(lc)

    # Draw current points
    scatter = ax.scatter(
        x_current,
        y_current,
        c=point_colors if labels is not None else [palette.source],
        s=style.scatter_size * 1.5,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )

    # Draw velocity arrows if requested
    if show_velocity and velocity is not None:
        vx = velocity[:, dim_x]
        vy = velocity[:, dim_y]

        # Normalize for display
        v_mag = np.sqrt(vx**2 + vy**2)
        v_mag_max = v_mag.max() if v_mag.max() > 0 else 1
        scale = 0.1 / v_mag_max  # Scale to reasonable arrow length

        ax.quiver(
            x_current,
            y_current,
            vx * scale,
            vy * scale,
            color=palette.vector_field,
            alpha=0.7,
            scale=1,
            scale_units="xy",
            width=0.005,
            zorder=9,
        )

    # Ghost of initial position
    if current_idx > 0:
        Phi_initial = Phi_sequence[0]
        ax.scatter(
            Phi_initial[:, dim_x],
            Phi_initial[:, dim_y],
            c=point_colors if labels is not None else [palette.source],
            s=style.scatter_size * 0.5,
            alpha=0.15,
            marker="o",
            zorder=1,
        )

    # Set axis limits with padding
    all_x = np.concatenate([Phi[:, dim_x] for Phi in Phi_sequence])
    all_y = np.concatenate([Phi[:, dim_y] for Phi in Phi_sequence])
    x_margin = (all_x.max() - all_x.min()) * 0.1 + 0.01
    y_margin = (all_y.max() - all_y.min()) * 0.1 + 0.01

    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    ax.set_xlabel(f"$\\phi_{{{dim_x + 1}}}$", fontsize=style.label_size)
    ax.set_ylabel(f"$\\phi_{{{dim_y + 1}}}$", fontsize=style.label_size)

    current_t = current_idx / max(n_timesteps - 1, 1)
    ax.set_title(f"Spectral Embedding ($t = {current_t:.2f}$)", fontsize=style.title_size)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend for clusters
    if labels is not None:
        for label_idx in unique_labels:
            ax.scatter(
                [], [],
                c=[cluster_colors[int(label_idx) % len(cluster_colors)]],
                s=style.scatter_size,
                label=f"Cluster {int(label_idx) + 1}",
            )
        ax.legend(
            loc="upper right",
            fontsize=style.legend_size,
            framealpha=0.9,
        )

    fig.tight_layout()
    return fig


def plot_embedding_trajectory(
    Phi_sequence: List[NDArray],
    node_indices: Optional[List[int]] = None,
    dims: tuple = (0, 1),
    style: Optional[VizStyle] = None,
) -> plt.Figure:
    """
    Plot full trajectories of selected nodes through spectral space.

    Args:
        Phi_sequence: List of eigenvector matrices over time.
        node_indices: Which nodes to highlight. If None, samples a few.
        dims: Eigenvector dimensions for x, y.
        style: Visualization style.

    Returns:
        Matplotlib figure with trajectory paths.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    n_timesteps = len(Phi_sequence)
    n_nodes = Phi_sequence[0].shape[0]
    dim_x, dim_y = dims

    if node_indices is None:
        # Sample a few representative nodes
        node_indices = np.linspace(0, n_nodes - 1, min(10, n_nodes), dtype=int).tolist()

    fig, ax = plt.subplots(figsize=(style.fig_width * 0.8, style.fig_height * 0.7))

    colors = get_time_colors(n_timesteps)
    node_colors = plt.cm.Set1(np.linspace(0, 1, len(node_indices)))

    for i, node_idx in enumerate(node_indices):
        # Extract trajectory for this node
        traj_x = [Phi_sequence[t][node_idx, dim_x] for t in range(n_timesteps)]
        traj_y = [Phi_sequence[t][node_idx, dim_y] for t in range(n_timesteps)]

        # Plot trajectory line
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            colors=colors[:-1],
            linewidths=2,
            alpha=0.8,
        )
        ax.add_collection(lc)

        # Mark start and end
        ax.scatter(
            [traj_x[0]], [traj_y[0]],
            c=[palette.source],
            s=80,
            marker="o",
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )
        ax.scatter(
            [traj_x[-1]], [traj_y[-1]],
            c=[palette.target],
            s=80,
            marker="s",
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

    ax.set_xlabel(f"$\\phi_{{{dim_x + 1}}}$", fontsize=style.label_size)
    ax.set_ylabel(f"$\\phi_{{{dim_y + 1}}}$", fontsize=style.label_size)
    ax.set_title("Node Trajectories in Spectral Space", fontsize=style.title_size)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Simple legend
    ax.scatter([], [], c=palette.source, s=60, marker="o", label="$t=0$")
    ax.scatter([], [], c=palette.target, s=60, marker="s", label="$t=1$")
    ax.legend(loc="upper right", fontsize=style.legend_size)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Test with mock data
    np.random.seed(42)
    n_nodes = 100
    k = 6
    n_timesteps = 30

    # Create evolving eigenvectors (two clusters that merge)
    Phi_sequence = []
    labels = np.array([0] * 50 + [1] * 50)

    for t in range(n_timesteps):
        progress = t / (n_timesteps - 1)

        # Two clusters moving together
        cluster1_center = np.array([-0.2 * (1 - progress), 0.1])
        cluster2_center = np.array([0.2 * (1 - progress), -0.1])

        Phi = np.zeros((n_nodes, k))
        Phi[:50, :2] = cluster1_center + 0.05 * np.random.randn(50, 2)
        Phi[50:, :2] = cluster2_center + 0.05 * np.random.randn(50, 2)

        # Add some structure to other dimensions
        for j in range(2, k):
            Phi[:, j] = 0.1 * np.random.randn(n_nodes)

        Phi_sequence.append(Phi)

    # Test current embedding plot
    fig1 = plot_spectral_embedding(
        Phi_sequence,
        current_idx=15,
        labels=labels,
        trail_length=10,
    )
    plt.show()

    # Test trajectory plot
    fig2 = plot_embedding_trajectory(
        Phi_sequence,
        node_indices=[0, 25, 50, 75],
    )
    plt.show()
