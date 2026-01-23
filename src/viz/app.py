"""
Streamlit interactive dashboard for Flow Matching visualization.

Mimics the "Play" functionality of Diffusion Explorer while running
your actual PyTorch research code.

Features:
    - Time slider to scrub through flow evolution
    - Scatter plot of particles at time t
    - Toggle vector field overlay
    - Play/Pause animation button
    - Side-by-side method comparison

Usage:
    cd src && streamlit run viz/app.py

    Or with custom model:
    streamlit run viz/app.py -- --model_path path/to/checkpoint.pt
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from viz.styles import ColorPalette, VizStyle, get_time_colors
from viz.static import integrate_ode, sample_from_gaussian_mixture


# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "time" not in st.session_state:
        st.session_state.time = 0.0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "x0" not in st.session_state:
        st.session_state.x0 = None
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = None
    if "velocity_mode" not in st.session_state:
        st.session_state.velocity_mode = "gaussian_to_gaussian"


# =============================================================================
# Velocity Field Definitions
# =============================================================================


def create_velocity_field(mode: str):
    """Create velocity field based on selected mode."""
    if mode == "gaussian_to_gaussian":
        # Simple translation flow
        source_center = np.array([-1.5, 0.0])
        target_center = np.array([1.5, 0.0])

        def velocity(x: np.ndarray, t: float) -> np.ndarray:
            # Constant velocity field: v = target - source
            direction = target_center - source_center
            return np.tile(direction, (len(x), 1))

        return velocity, source_center, target_center

    elif mode == "rotation":
        # Rotating flow (non-straight paths)
        def velocity(x: np.ndarray, t: float) -> np.ndarray:
            omega = np.pi  # Half rotation
            vx = -omega * x[:, 1]
            vy = omega * x[:, 0]
            return np.stack([vx, vy], axis=1)

        return velocity, np.array([1.0, 0.0]), np.array([-1.0, 0.0])

    elif mode == "two_moons":
        # Two crescent shapes
        source_center = np.array([-0.5, 0.5])
        target_center = np.array([0.5, -0.5])

        def velocity(x: np.ndarray, t: float) -> np.ndarray:
            # Flow toward target with some curvature
            direction = target_center - x
            # Add rotation component for curved paths
            rotation = np.stack([-direction[:, 1], direction[:, 0]], axis=1) * 0.3
            return direction + rotation * (1 - t)

        return velocity, source_center, target_center

    elif mode == "swiss_roll_unroll":
        # Unrolling spiral motion
        def velocity(x: np.ndarray, t: float) -> np.ndarray:
            r = np.linalg.norm(x, axis=1, keepdims=True)
            r = np.maximum(r, 0.1)
            # Outward spiral
            radial = x / r * 0.5
            tangent = np.stack([x[:, 1], -x[:, 0]], axis=1) / r * 0.3
            return radial + tangent

        return velocity, np.array([0.0, 0.0]), np.array([2.0, 0.0])

    else:
        raise ValueError(f"Unknown velocity mode: {mode}")


def generate_source_samples(mode: str, n_samples: int = 200, seed: int = 42) -> np.ndarray:
    """Generate source distribution samples based on mode."""
    rng = np.random.default_rng(seed)

    if mode == "gaussian_to_gaussian":
        center = np.array([[-1.5, 0.0]])
        return sample_from_gaussian_mixture(n_samples, center, stds=0.4, seed=seed)

    elif mode == "rotation":
        # Ring distribution
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        r = 1.5 + 0.2 * rng.standard_normal(n_samples)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    elif mode == "two_moons":
        # Upper crescent
        theta = rng.uniform(0, np.pi, n_samples)
        x = np.cos(theta) - 0.5
        y = np.sin(theta) + 0.5
        noise = 0.1 * rng.standard_normal((n_samples, 2))
        return np.stack([x, y], axis=1) + noise

    elif mode == "swiss_roll_unroll":
        # Tight spiral
        t = 1.5 * np.pi + 2 * np.pi * rng.uniform(0, 1, n_samples)
        r = 0.3 * t / (2 * np.pi)
        noise = 0.05 * rng.standard_normal((n_samples, 2))
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1) + noise

    else:
        return sample_from_gaussian_mixture(
            n_samples,
            np.array([[0.0, 0.0]]),
            stds=1.0,
            seed=seed,
        )


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_particles_at_time(
    trajectory: list[np.ndarray],
    t: float,
    n_steps: int,
    palette: ColorPalette,
    show_trails: bool = True,
    trail_length: int = 10,
) -> plt.Figure:
    """Plot particles at a specific time with optional trails."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get current frame index
    idx = int(t * n_steps)
    idx = min(idx, len(trajectory) - 1)
    x_t = trajectory[idx]
    x_0 = trajectory[0]
    x_1 = trajectory[-1]

    # Draw trails if enabled
    if show_trails and idx > 0:
        colors = get_time_colors(idx + 1)
        trail_start = max(0, idx - trail_length)

        for i in range(len(x_0)):
            points = np.array([traj[i] for traj in trajectory[trail_start:idx + 1]])
            if len(points) > 1:
                segments = np.array([
                    [points[j], points[j + 1]]
                    for j in range(len(points) - 1)
                ])
                lc = LineCollection(
                    segments,
                    colors=colors[trail_start:idx],
                    linewidths=1.5,
                    alpha=0.4,
                )
                ax.add_collection(lc)

    # Get time-based color
    time_color = get_time_colors(2)[0] if t < 0.5 else get_time_colors(2)[1]

    # Draw current particles
    ax.scatter(
        x_t[:, 0], x_t[:, 1],
        c=[time_color],
        s=50,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
        zorder=10,
        label=f"Particles at $t={t:.2f}$",
    )

    # Draw ghost of source (faded)
    if t > 0.1:
        ax.scatter(
            x_0[:, 0], x_0[:, 1],
            c=palette.source,
            s=30,
            alpha=0.2,
            marker="o",
            zorder=1,
        )

    # Draw ghost of target (faded)
    if t < 0.9:
        ax.scatter(
            x_1[:, 0], x_1[:, 1],
            c=palette.target,
            s=30,
            alpha=0.2,
            marker="o",
            zorder=1,
        )

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(f"Flow at $t = {t:.2f}$", fontsize=14)
    ax.legend(loc="upper right")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_vector_field_overlay(
    velocity_fn,
    t: float,
    trajectory: list[np.ndarray],
    n_steps: int,
    palette: ColorPalette,
    grid_size: int = 15,
) -> plt.Figure:
    """Plot particles with vector field overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get current particles
    idx = int(t * n_steps)
    idx = min(idx, len(trajectory) - 1)
    x_t = trajectory[idx]

    # Create vector field grid
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Compute velocities
    velocities = velocity_fn(grid_points, t)
    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(X.shape)
    magnitude = np.sqrt(U**2 + V**2)

    # Normalize for display
    max_mag = magnitude.max() if magnitude.max() > 0 else 1
    U_norm = U / max_mag
    V_norm = V / max_mag

    # Draw vector field
    ax.quiver(
        X, Y, U_norm, V_norm, magnitude,
        cmap="viridis",
        alpha=0.6,
        scale=25,
        width=0.004,
    )

    # Draw particles
    time_color = get_time_colors(2)[0] if t < 0.5 else get_time_colors(2)[1]
    ax.scatter(
        x_t[:, 0], x_t[:, 1],
        c=[time_color],
        s=60,
        alpha=0.9,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(f"Vector Field at $t = {t:.2f}$", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# Main Streamlit App
# =============================================================================


def main():
    """Main Streamlit application."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: uv add streamlit")
        return

    st.set_page_config(
        page_title="Flow Matching Explorer",
        page_icon="🌊",
        layout="wide",
    )

    init_session_state()

    # Header
    st.title("🌊 Flow Matching Explorer")
    st.markdown(
        "Interactive visualization of flow matching and optimal transport. "
        "Inspired by [Diffusion Explorer](https://github.com/helblazer811/Diffusion-Explorer)."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Flow type selection
        flow_type = st.selectbox(
            "Flow Type",
            options=[
                "gaussian_to_gaussian",
                "rotation",
                "two_moons",
                "swiss_roll_unroll",
            ],
            format_func=lambda x: {
                "gaussian_to_gaussian": "Gaussian → Gaussian (Linear OT)",
                "rotation": "Rotation (Curved Paths)",
                "two_moons": "Two Moons",
                "swiss_roll_unroll": "Swiss Roll Unroll",
            }.get(x, x),
        )

        # Number of particles
        n_particles = st.slider("Number of Particles", 50, 500, 200, 50)

        # Integration steps
        n_steps = st.slider("Integration Steps", 10, 200, 50, 10)

        # Visualization options
        st.subheader("Visualization")
        show_vector_field = st.checkbox("Show Vector Field", value=False)
        show_trails = st.checkbox("Show Particle Trails", value=True)
        trail_length = st.slider("Trail Length", 5, 50, 15, 5) if show_trails else 0

        # Regenerate button
        if st.button("🔄 Regenerate Samples"):
            st.session_state.x0 = None
            st.session_state.trajectory = None

    # Generate/cache trajectory
    palette = ColorPalette()

    if (
        st.session_state.x0 is None
        or st.session_state.velocity_mode != flow_type
        or len(st.session_state.x0) != n_particles
    ):
        with st.spinner("Computing trajectory..."):
            velocity_fn, _, _ = create_velocity_field(flow_type)
            x0 = generate_source_samples(flow_type, n_particles)
            trajectory = integrate_ode(velocity_fn, x0, n_steps=n_steps)

            st.session_state.x0 = x0
            st.session_state.trajectory = trajectory
            st.session_state.velocity_mode = flow_type

    trajectory = st.session_state.trajectory
    velocity_fn, _, _ = create_velocity_field(flow_type)

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Time slider
        t = st.slider(
            "Time $t$",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.time,
            step=0.01,
            key="time_slider",
        )
        st.session_state.time = t

        # Animation controls
        col_play, col_reset, col_speed = st.columns(3)

        with col_play:
            if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
                st.session_state.playing = not st.session_state.playing

        with col_reset:
            if st.button("⏮️ Reset"):
                st.session_state.time = 0.0
                st.session_state.playing = False

        with col_speed:
            speed = st.select_slider(
                "Speed",
                options=[0.5, 1.0, 2.0, 4.0],
                value=1.0,
            )

        # Plot
        if show_vector_field:
            fig = plot_vector_field_overlay(
                velocity_fn, t, trajectory, n_steps, palette
            )
        else:
            fig = plot_particles_at_time(
                trajectory, t, n_steps, palette,
                show_trails=show_trails,
                trail_length=trail_length,
            )

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Info panel
        st.subheader("Info")

        idx = int(t * len(trajectory))
        idx = min(idx, len(trajectory) - 1)
        x_t = trajectory[idx]
        x_0 = trajectory[0]
        x_1 = trajectory[-1]

        # Statistics
        st.metric("Time $t$", f"{t:.3f}")

        # Distance metrics
        if t > 0:
            displacement = np.mean(np.linalg.norm(x_t - x_0, axis=1))
            st.metric("Mean Displacement", f"{displacement:.3f}")

        if t < 1:
            remaining = np.mean(np.linalg.norm(x_1 - x_t, axis=1))
            st.metric("Distance to Target", f"{remaining:.3f}")

        # Flow straightness (for Flow Matching this should be ~1)
        if t > 0.1 and t < 0.9:
            total_path = np.mean(np.linalg.norm(x_1 - x_0, axis=1))
            current_path = np.mean(np.linalg.norm(x_t - x_0, axis=1))
            expected = t * total_path
            straightness = expected / (current_path + 1e-6)
            st.metric("Path Straightness", f"{straightness:.3f}")

        st.markdown("---")
        st.markdown(
            """
            **Legend:**
            - 🔵 Source distribution ($t=0$)
            - 🔴 Target distribution ($t=1$)
            - Colored trails show trajectory history
            """
        )

    # Animation loop
    if st.session_state.playing:
        new_t = st.session_state.time + 0.02 * speed
        if new_t >= 1.0:
            new_t = 0.0
            st.session_state.playing = False
        st.session_state.time = new_t
        time.sleep(0.05)
        st.rerun()


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()
