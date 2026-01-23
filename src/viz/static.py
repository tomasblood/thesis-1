"""
Static plotting functions for Flow Matching / Optimal Transport visualizations.

Provides publication-ready figures following the Diffusion Explorer aesthetic.
All functions return matplotlib Figure objects for flexibility in saving/display.

Key Functions:
    plot_flow_trajectories: Visualize particle paths through the flow
    plot_vector_field: Show velocity field v_t(x) at a given time
    plot_density_evolution: Multi-panel density evolution over time
    plot_comparison_panel: Side-by-side comparison (e.g., Flow vs Rectified Flow)
    plot_transport_plan: Visualize optimal transport couplings
"""

from typing import Callable, Literal, Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from viz.styles import (
    ColorPalette,
    VizStyle,
    add_time_annotation,
    apply_style,
    create_figure,
    format_axis_labels,
    get_cmap_flow,
    get_time_colors,
)


# Type aliases for clarity
Array2D = NDArray[np.floating]  # Shape (N, 2)
ArrayND = NDArray[np.floating]


class VelocityModel(Protocol):
    """Protocol for models that predict velocity."""

    def __call__(self, x: ArrayND, t: float) -> ArrayND:
        """Predict velocity at points x and time t."""
        ...


# =============================================================================
# Core Integration Utilities
# =============================================================================


def integrate_ode(
    velocity_fn: Callable[[Array2D, float], Array2D],
    x0: Array2D,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_steps: int = 50,
    method: Literal["euler", "rk4", "midpoint"] = "euler",
) -> list[Array2D]:
    """
    Integrate ODE dx/dt = v_t(x) to generate trajectories.

    Args:
        velocity_fn: Function (x, t) -> velocity, where x has shape (N, d).
        x0: Initial points of shape (N, d).
        t_start: Starting time.
        t_end: Ending time.
        n_steps: Number of integration steps.
        method: Integration method.

    Returns:
        List of arrays [x_t0, x_t1, ..., x_tn], each of shape (N, d).
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    dt = (t_end - t_start) / n_steps

    for i in range(n_steps):
        t = t_start + i * dt

        if method == "euler":
            v = velocity_fn(x, t)
            x = x + dt * v

        elif method == "midpoint":
            v1 = velocity_fn(x, t)
            x_mid = x + 0.5 * dt * v1
            v2 = velocity_fn(x_mid, t + 0.5 * dt)
            x = x + dt * v2

        elif method == "rk4":
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        trajectory.append(x.copy())

    return trajectory


def sample_from_gaussian_mixture(
    n_samples: int,
    centers: Array2D,
    stds: ArrayND | float = 0.3,
    weights: ArrayND | None = None,
    seed: int | None = None,
) -> Array2D:
    """
    Sample from a Gaussian mixture model.

    Args:
        n_samples: Number of samples.
        centers: Cluster centers of shape (K, d).
        stds: Standard deviations (scalar or shape (K,)).
        weights: Mixture weights (uniform if None).
        seed: Random seed.

    Returns:
        Samples of shape (n_samples, d).
    """
    rng = np.random.default_rng(seed)
    n_centers, d = centers.shape

    if weights is None:
        weights = np.ones(n_centers) / n_centers

    if isinstance(stds, (int, float)):
        stds = np.full(n_centers, stds)

    # Sample cluster assignments
    assignments = rng.choice(n_centers, size=n_samples, p=weights)

    # Sample from each cluster
    samples = np.zeros((n_samples, d))
    for k in range(n_centers):
        mask = assignments == k
        count = mask.sum()
        if count > 0:
            samples[mask] = centers[k] + stds[k] * rng.standard_normal((count, d))

    return samples


# =============================================================================
# Main Visualization Functions
# =============================================================================


def plot_flow_trajectories(
    velocity_fn: Callable[[Array2D, float], Array2D],
    x0: Array2D,
    steps: int = 50,
    style: VizStyle | None = None,
    show_endpoints: bool = True,
    color_by_time: bool = True,
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    title: str = "Flow Trajectories",
) -> plt.Figure:
    """
    Visualize flow trajectories from initial points.

    Integrates dx/dt = v_t(x) from t=0 to t=1 and plots full particle paths.
    Inspired by Diffusion Explorer "Path" view.

    Args:
        velocity_fn: Velocity field function (x, t) -> v.
        x0: Initial points of shape (N, 2).
        steps: Number of integration steps.
        style: Visualization style.
        show_endpoints: Show start/end markers.
        color_by_time: Color trajectories by time (else uniform).
        xlim: X-axis limits.
        ylim: Y-axis limits.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    # Integrate trajectories
    trajectory = integrate_ode(velocity_fn, x0, n_steps=steps)
    n_particles = x0.shape[0]

    fig, ax = plt.subplots(figsize=(style.fig_width, style.fig_height))

    if color_by_time:
        # Create colored line segments
        colors = get_time_colors(steps + 1)

        for i in range(n_particles):
            # Extract single particle trajectory
            points = np.array([traj[i] for traj in trajectory])
            segments = np.array([
                [points[j], points[j + 1]]
                for j in range(len(points) - 1)
            ])

            lc = LineCollection(
                segments,
                colors=colors[:-1],
                linewidths=style.trajectory_lw,
                alpha=palette.trajectory_alpha * 1.5,  # Slightly brighter
            )
            ax.add_collection(lc)
    else:
        # Uniform color trajectories
        for i in range(n_particles):
            points = np.array([traj[i] for traj in trajectory])
            ax.plot(
                points[:, 0], points[:, 1],
                color=palette.trajectory,
                linewidth=style.trajectory_lw,
                alpha=palette.trajectory_alpha,
            )

    if show_endpoints:
        # Source points (t=0)
        ax.scatter(
            x0[:, 0], x0[:, 1],
            c=palette.source,
            s=style.scatter_size,
            alpha=palette.scatter_alpha,
            label="$x_0$ (source)",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

        # Target points (t=1)
        x1 = trajectory[-1]
        ax.scatter(
            x1[:, 0], x1[:, 1],
            c=palette.target,
            s=style.scatter_size,
            alpha=palette.scatter_alpha,
            label="$x_1$ (target)",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9)
    format_axis_labels(ax, title=title, style=style)

    fig.tight_layout()
    return fig


def plot_vector_field(
    velocity_fn: Callable[[Array2D, float], Array2D],
    t: float,
    grid_size: int = 20,
    style: VizStyle | None = None,
    method: Literal["quiver", "streamplot"] = "streamplot",
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    title: str | None = None,
    density: float = 1.0,
    color_by_magnitude: bool = True,
) -> plt.Figure:
    """
    Visualize velocity field v_t(x) at a specific time.

    Creates a meshgrid and computes velocity at each point, then
    visualizes using quiver or streamplot.

    Args:
        velocity_fn: Velocity field function (x, t) -> v.
        t: Time at which to visualize field.
        grid_size: Number of points per axis.
        style: Visualization style.
        method: "quiver" for arrows, "streamplot" for streamlines.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        title: Plot title (auto-generated if None).
        density: Stream density (for streamplot).
        color_by_magnitude: Color by velocity magnitude.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    if title is None:
        title = f"Vector Field at $t = {t:.2f}$"

    # Create meshgrid
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute velocities at grid points
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    velocities = velocity_fn(grid_points, t)
    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(X.shape)

    # Compute magnitude for coloring
    magnitude = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(style.fig_width, style.fig_height))

    if method == "quiver":
        if color_by_magnitude:
            quiver = ax.quiver(
                X, Y, U, V, magnitude,
                cmap=get_cmap_flow(),
                scale=style.arrow_scale,
                width=0.004,
                alpha=palette.vector_alpha,
            )
            plt.colorbar(quiver, ax=ax, label="Velocity Magnitude", shrink=0.8)
        else:
            ax.quiver(
                X, Y, U, V,
                color=palette.vector_field,
                scale=style.arrow_scale,
                width=0.004,
                alpha=palette.vector_alpha,
            )

    elif method == "streamplot":
        if color_by_magnitude:
            stream = ax.streamplot(
                X, Y, U, V,
                color=magnitude,
                cmap=get_cmap_flow(),
                density=density,
                linewidth=style.vector_lw,
                arrowsize=1.2,
            )
            plt.colorbar(stream.lines, ax=ax, label="Velocity Magnitude", shrink=0.8)
        else:
            ax.streamplot(
                X, Y, U, V,
                color=palette.vector_field,
                density=density,
                linewidth=style.vector_lw,
                arrowsize=1.2,
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    format_axis_labels(ax, title=title, style=style)
    add_time_annotation(ax, t, style)

    fig.tight_layout()
    return fig


def plot_density_evolution(
    velocity_fn: Callable[[Array2D, float], Array2D],
    x0: Array2D,
    t_steps: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    style: VizStyle | None = None,
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    n_integration_steps: int = 100,
    show_kde: bool = True,
    kde_levels: int = 10,
) -> plt.Figure:
    """
    Visualize density evolution over time.

    Shows the distribution of particles at multiple time points,
    optionally with KDE contours.

    Args:
        velocity_fn: Velocity field function (x, t) -> v.
        x0: Initial samples of shape (N, 2).
        t_steps: Time points to visualize.
        style: Visualization style.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        n_integration_steps: Total integration steps from t=0 to t=1.
        show_kde: Show kernel density estimate contours.
        kde_levels: Number of contour levels.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette
    n_panels = len(t_steps)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(style.fig_width * 0.5 * n_panels, style.fig_height * 0.6),
        sharex=True, sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    # Pre-compute full trajectory
    trajectory = integrate_ode(
        velocity_fn, x0,
        t_start=0.0, t_end=1.0,
        n_steps=n_integration_steps,
    )

    # Get colors for time interpolation
    time_colors = get_time_colors(n_panels)

    for i, t in enumerate(t_steps):
        ax = axes[i]

        # Find closest trajectory point
        idx = int(t * n_integration_steps)
        idx = min(idx, len(trajectory) - 1)
        x_t = trajectory[idx]

        # Scatter plot
        ax.scatter(
            x_t[:, 0], x_t[:, 1],
            c=[time_colors[i]],
            s=style.scatter_size * 0.5,
            alpha=palette.scatter_alpha,
            edgecolors="white",
            linewidths=0.3,
        )

        # KDE contours
        if show_kde and len(x_t) > 10:
            try:
                kde = gaussian_kde(x_t.T)
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z = kde(positions).reshape(xx.shape)

                ax.contour(
                    xx, yy, z,
                    levels=kde_levels,
                    colors=[time_colors[i]],
                    alpha=palette.contour_alpha,
                    linewidths=0.8,
                )
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip KDE if degenerate

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"$t = {t:.2f}$", fontsize=style.label_size)

        if i == 0:
            ax.set_ylabel("$x_2$", fontsize=style.label_size)
        ax.set_xlabel("$x_1$", fontsize=style.label_size)

    fig.suptitle("Density Evolution", fontsize=style.title_size, y=1.02)
    fig.tight_layout()
    return fig


def plot_comparison_panel(
    velocity_fn_left: Callable[[Array2D, float], Array2D],
    velocity_fn_right: Callable[[Array2D, float], Array2D],
    x0: Array2D,
    steps: int = 50,
    style: VizStyle | None = None,
    title_left: str = "Method A",
    title_right: str = "Method B",
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
) -> plt.Figure:
    """
    Side-by-side comparison of two flow methods.

    Useful for comparing Flow Matching vs Rectified Flow, different
    number of Euler steps, etc.

    Args:
        velocity_fn_left: Left panel velocity function.
        velocity_fn_right: Right panel velocity function.
        x0: Initial points (shared by both methods).
        steps: Integration steps.
        style: Visualization style.
        title_left: Left panel title.
        title_right: Right panel title.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(style.fig_width * 1.5, style.fig_height * 0.8),
        sharex=True, sharey=True,
    )

    # Compute trajectories
    traj_left = integrate_ode(velocity_fn_left, x0, n_steps=steps)
    traj_right = integrate_ode(velocity_fn_right, x0, n_steps=steps)

    colors = get_time_colors(steps + 1)
    n_particles = x0.shape[0]

    for ax, trajectory, title in [
        (ax_left, traj_left, title_left),
        (ax_right, traj_right, title_right),
    ]:
        # Draw trajectories
        for i in range(n_particles):
            points = np.array([traj[i] for traj in trajectory])
            segments = np.array([
                [points[j], points[j + 1]]
                for j in range(len(points) - 1)
            ])
            lc = LineCollection(
                segments,
                colors=colors[:-1],
                linewidths=style.trajectory_lw,
                alpha=palette.trajectory_alpha * 1.5,
            )
            ax.add_collection(lc)

        # Source points
        ax.scatter(
            x0[:, 0], x0[:, 1],
            c=palette.source,
            s=style.scatter_size,
            alpha=palette.scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

        # Target points
        x1 = trajectory[-1]
        ax.scatter(
            x1[:, 0], x1[:, 1],
            c=palette.target,
            s=style.scatter_size,
            alpha=palette.scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=style.title_size)
        ax.set_xlabel("$x_1$", fontsize=style.label_size)

    ax_left.set_ylabel("$x_2$", fontsize=style.label_size)

    fig.tight_layout()
    return fig


def plot_transport_plan(
    x_source: Array2D,
    x_target: Array2D,
    transport_matrix: ArrayND,
    style: VizStyle | None = None,
    threshold: float = 0.01,
    show_marginals: bool = True,
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    title: str = "Optimal Transport Plan",
) -> plt.Figure:
    """
    Visualize optimal transport coupling between distributions.

    Shows source and target distributions with connecting lines
    weighted by transport plan mass.

    Args:
        x_source: Source samples of shape (N, 2).
        x_target: Target samples of shape (M, 2).
        transport_matrix: Transport plan of shape (N, M).
        style: Visualization style.
        threshold: Minimum mass to draw connection.
        show_marginals: Show marginal KDE contours.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette

    fig, ax = plt.subplots(figsize=(style.fig_width, style.fig_height))

    # Normalize transport matrix for visualization
    T_norm = transport_matrix / transport_matrix.max()

    # Draw transport connections
    for i in range(len(x_source)):
        for j in range(len(x_target)):
            if T_norm[i, j] > threshold:
                ax.plot(
                    [x_source[i, 0], x_target[j, 0]],
                    [x_source[i, 1], x_target[j, 1]],
                    color=palette.trajectory,
                    alpha=T_norm[i, j] * 0.5,
                    linewidth=T_norm[i, j] * 2,
                    zorder=1,
                )

    # Source distribution
    ax.scatter(
        x_source[:, 0], x_source[:, 1],
        c=palette.source,
        s=style.scatter_size,
        alpha=palette.scatter_alpha,
        label="Source $\\mu$",
        edgecolors="white",
        linewidths=0.5,
        zorder=10,
    )

    # Target distribution
    ax.scatter(
        x_target[:, 0], x_target[:, 1],
        c=palette.target,
        s=style.scatter_size,
        alpha=palette.scatter_alpha,
        label="Target $\\nu$",
        edgecolors="white",
        linewidths=0.5,
        zorder=10,
    )

    # Marginal KDEs
    if show_marginals and len(x_source) > 10 and len(x_target) > 10:
        try:
            for x, color in [(x_source, palette.source), (x_target, palette.target)]:
                kde = gaussian_kde(x.T)
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, z, levels=5, colors=[color], alpha=0.3, linewidths=0.8)
        except (np.linalg.LinAlgError, ValueError):
            pass

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9)
    format_axis_labels(ax, title=title, style=style)

    fig.tight_layout()
    return fig


def plot_euler_comparison(
    velocity_fn: Callable[[Array2D, float], Array2D],
    x0: Array2D,
    step_counts: Sequence[int] = (1, 4, 16, 64),
    style: VizStyle | None = None,
    reference_steps: int = 256,
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
) -> plt.Figure:
    """
    Compare trajectories with different numbers of Euler steps.

    Shows how trajectory accuracy improves with more integration steps.
    Reference trajectory uses many steps as ground truth.

    Args:
        velocity_fn: Velocity field function.
        x0: Initial points.
        step_counts: Different step counts to compare.
        style: Visualization style.
        reference_steps: Steps for reference trajectory.
        xlim: X-axis limits.
        ylim: Y-axis limits.

    Returns:
        Matplotlib figure.
    """
    if style is None:
        style = VizStyle()

    apply_style(style)
    palette = style.palette
    n_panels = len(step_counts)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(style.fig_width * 0.5 * n_panels, style.fig_height * 0.6),
        sharex=True, sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    # Reference trajectory
    ref_trajectory = integrate_ode(velocity_fn, x0, n_steps=reference_steps)
    x_ref = ref_trajectory[-1]

    for i, n_steps in enumerate(step_counts):
        ax = axes[i]

        # Compute trajectory with this step count
        trajectory = integrate_ode(velocity_fn, x0, n_steps=n_steps)
        x_final = trajectory[-1]
        n_particles = x0.shape[0]

        # Draw trajectories
        colors = get_time_colors(n_steps + 1)
        for j in range(n_particles):
            points = np.array([traj[j] for traj in trajectory])
            segments = np.array([
                [points[k], points[k + 1]]
                for k in range(len(points) - 1)
            ])
            lc = LineCollection(
                segments,
                colors=colors[:-1],
                linewidths=style.trajectory_lw,
                alpha=palette.trajectory_alpha * 1.5,
            )
            ax.add_collection(lc)

        # Draw error lines to reference
        for j in range(n_particles):
            ax.plot(
                [x_final[j, 0], x_ref[j, 0]],
                [x_final[j, 1], x_ref[j, 1]],
                color=palette.error,
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

        # Reference points (ground truth)
        ax.scatter(
            x_ref[:, 0], x_ref[:, 1],
            c=palette.success,
            s=style.scatter_size * 0.5,
            alpha=0.7,
            marker="x",
            linewidths=1,
            label="Reference" if i == 0 else None,
            zorder=11,
        )

        # Endpoints from this step count
        ax.scatter(
            x_final[:, 0], x_final[:, 1],
            c=palette.target,
            s=style.scatter_size * 0.7,
            alpha=palette.scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

        # Start points
        ax.scatter(
            x0[:, 0], x0[:, 1],
            c=palette.source,
            s=style.scatter_size * 0.7,
            alpha=palette.scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

        # Compute error
        error = np.mean(np.linalg.norm(x_final - x_ref, axis=1))
        ax.set_title(f"$N = {n_steps}$ steps\n(error: {error:.3f})", fontsize=style.label_size)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$", fontsize=style.label_size)

        if i == 0:
            ax.set_ylabel("$x_2$", fontsize=style.label_size)

    fig.suptitle("Euler Step Comparison", fontsize=style.title_size, y=1.05)
    fig.tight_layout()
    return fig


# =============================================================================
# Utility Functions
# =============================================================================


def create_toy_velocity_field(
    mode: Literal["linear", "rotation", "swiss_roll", "two_moons"] = "linear"
) -> Callable[[Array2D, float], Array2D]:
    """
    Create a toy velocity field for testing visualizations.

    Args:
        mode: Type of velocity field.

    Returns:
        Velocity function (x, t) -> v.
    """
    if mode == "linear":
        # Linear interpolation: v = x_1 - x_0 (optimal transport for Gaussians)
        def velocity(x: Array2D, t: float) -> Array2D:
            # Move from source centered at (-1, 0) to target at (1, 0)
            target = np.array([1.0, 0.0])
            source = np.array([-1.0, 0.0])
            return (target - source) * np.ones_like(x)
        return velocity

    elif mode == "rotation":
        # Rotating flow
        def velocity(x: Array2D, t: float) -> Array2D:
            omega = 2 * np.pi  # One full rotation
            vx = -omega * x[:, 1]
            vy = omega * x[:, 0]
            return np.stack([vx, vy], axis=1)
        return velocity

    elif mode == "swiss_roll":
        # Spiraling inward flow
        def velocity(x: Array2D, t: float) -> Array2D:
            r = np.linalg.norm(x, axis=1, keepdims=True)
            r = np.maximum(r, 0.1)  # Avoid division by zero
            # Inward radial + rotation
            radial = -x / r
            tangent = np.stack([-x[:, 1], x[:, 0]], axis=1) / r
            return 0.5 * radial + 0.3 * tangent
        return velocity

    elif mode == "two_moons":
        # Flow between two crescents
        def velocity(x: Array2D, t: float) -> Array2D:
            # Simple gradient descent to two moon centers
            center1 = np.array([0.0, 0.5])
            center2 = np.array([1.0, -0.5])

            # Weighted combination based on position
            weight = 1 / (1 + np.exp(-x[:, 0:1]))
            target = (1 - weight) * center1 + weight * center2
            return (target - x) * 2
        return velocity

    raise ValueError(f"Unknown mode: {mode}")
