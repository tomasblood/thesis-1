"""
Eigenvalue trajectory panel for temporal spectral visualization.

Shows eigenvalues lambda_1, lambda_2, ..., lambda_k as trajectories
over time, with vertical line indicating current timestep.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.styles import VizStyle, ColorPalette, apply_style


def plot_eigenvalue_trajectories(
    lambda_sequence: List[NDArray],
    current_idx: int,
    lambda_aligned: Optional[List[NDArray]] = None,
    show_raw: bool = False,
    highlight_crossings: bool = True,
    style: Optional[VizStyle] = None,
) -> plt.Figure:
    """
    Plot eigenvalue trajectories over time.

    Args:
        lambda_sequence: List of eigenvalue arrays, one per timestep.
            Each array has shape (k,) where k is number of eigenvectors.
        current_idx: Current timestep index to highlight.
        lambda_aligned: Optional aligned eigenvalues. If provided and
            show_raw=True, shows both raw (faded) and aligned (solid).
        show_raw: If True and lambda_aligned provided, show raw as background.
        highlight_crossings: Mark points where eigenvalues cross.
        style: Visualization style configuration.

    Returns:
        Matplotlib figure with eigenvalue trajectories.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    n_timesteps = len(lambda_sequence)
    k = len(lambda_sequence[0])
    t_values = np.linspace(0, 1, n_timesteps)

    # Stack eigenvalues into array (n_timesteps, k)
    lambda_array = np.array(lambda_sequence)

    if lambda_aligned is not None:
        lambda_aligned_array = np.array(lambda_aligned)
    else:
        lambda_aligned_array = lambda_array

    fig, ax = plt.subplots(figsize=(style.fig_width * 0.8, style.fig_height * 0.6))

    # Color palette for eigenvalue modes
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))

    # Plot raw eigenvalues as faded background if requested
    if show_raw and lambda_aligned is not None:
        for j in range(k):
            ax.plot(
                t_values,
                lambda_array[:, j],
                color=colors[j],
                alpha=0.2,
                linewidth=style.trajectory_lw,
                linestyle="--",
            )

    # Plot main eigenvalue trajectories
    for j in range(k):
        ax.plot(
            t_values,
            lambda_aligned_array[:, j],
            color=colors[j],
            alpha=0.9,
            linewidth=style.trajectory_lw * 1.2,
            label=f"$\\lambda_{{{j+1}}}$",
        )

    # Highlight eigenvalue crossings
    if highlight_crossings:
        crossings = _detect_crossings(lambda_aligned_array)
        for t_idx, mode_i, mode_j in crossings:
            ax.scatter(
                [t_values[t_idx]],
                [lambda_aligned_array[t_idx, mode_i]],
                color=palette.highlight,
                s=80,
                zorder=10,
                marker="o",
                edgecolors="white",
                linewidths=1,
            )

    # Vertical line at current time
    current_t = t_values[min(current_idx, n_timesteps - 1)]
    ax.axvline(
        x=current_t,
        color=palette.target,
        linestyle="-",
        linewidth=2,
        alpha=0.8,
        label=f"$t = {current_t:.2f}$",
    )

    # Mark current eigenvalues
    current_lambdas = lambda_aligned_array[min(current_idx, n_timesteps - 1)]
    ax.scatter(
        [current_t] * k,
        current_lambdas,
        color=colors,
        s=60,
        zorder=11,
        edgecolors="white",
        linewidths=1.5,
    )

    ax.set_xlabel("Time $t$", fontsize=style.label_size)
    ax.set_ylabel("Eigenvalue $\\lambda$", fontsize=style.label_size)
    ax.set_title("Eigenvalue Trajectories", fontsize=style.title_size)
    ax.set_xlim(0, 1)

    # Legend outside plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=style.legend_size,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def _detect_crossings(
    lambda_array: NDArray,
    threshold: float = 0.01,
) -> List[tuple]:
    """
    Detect eigenvalue crossings in temporal sequence.

    Args:
        lambda_array: Eigenvalues of shape (n_timesteps, k).
        threshold: Minimum relative change to consider a crossing.

    Returns:
        List of (timestep, mode_i, mode_j) tuples where crossings occur.
    """
    crossings = []
    n_timesteps, k = lambda_array.shape

    for t in range(n_timesteps - 1):
        # Check ordering at t and t+1
        order_t = np.argsort(lambda_array[t])
        order_t1 = np.argsort(lambda_array[t + 1])

        # If orderings differ, there's a crossing
        if not np.array_equal(order_t, order_t1):
            # Find which modes crossed
            for i in range(k):
                for j in range(i + 1, k):
                    diff_t = lambda_array[t, i] - lambda_array[t, j]
                    diff_t1 = lambda_array[t + 1, i] - lambda_array[t + 1, j]
                    # Sign change indicates crossing
                    if diff_t * diff_t1 < 0:
                        crossings.append((t + 1, i, j))

    return crossings


if __name__ == "__main__":
    # Test with mock data
    np.random.seed(42)
    n_timesteps = 50
    k = 5

    # Create eigenvalues that cross
    t = np.linspace(0, 1, n_timesteps)
    lambda_sequence = []
    for i in range(n_timesteps):
        lambdas = np.array([
            0.1 + 0.02 * i,                          # Slowly increasing
            0.3 - 0.01 * i + 0.1 * np.sin(4 * t[i]), # Oscillating
            0.5 + 0.03 * i,                          # Faster increase
            0.7,                                      # Constant
            0.9 - 0.02 * i,                          # Decreasing
        ])
        lambda_sequence.append(lambdas)

    fig = plot_eigenvalue_trajectories(
        lambda_sequence,
        current_idx=25,
        highlight_crossings=True,
    )
    plt.show()
