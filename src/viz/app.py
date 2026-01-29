"""
Streamlit interactive dashboard for Temporal Geodesic Flow Matching.

Visualizes the spectral flow model training and inference:
- Training on consecutive spectral pairs (Phi_t, lambda_t) -> (Phi_{t+1}, lambda_{t+1})
- One-step integration showing predicted vs true evolution
- Grassmann loss (subspace distance), eigenvalue loss, energy regularization
- NO sampling from noise - we learn the actual temporal dynamics

Usage:
    cd src && streamlit run viz/app.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from viz.styles import ColorPalette, VizStyle, get_time_colors
from viz.generators import generate_merging_clusters, generate_evolving_graph

# Import temporal spectral flow machinery
from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.alignment import SpectralAligner, SpectralMatcher, SignConvention


@dataclass
class FlowPrediction:
    """Prediction from one-step flow integration."""
    Phi_pred: NDArray[np.floating]  # Predicted eigenvectors
    lambda_pred: NDArray[np.floating]  # Predicted eigenvalues
    velocity_Phi: NDArray[np.floating]  # Velocity on Stiefel
    velocity_lambda: NDArray[np.floating]  # Velocity for eigenvalues


@dataclass
class FlowLosses:
    """Losses for temporal geodesic flow matching."""
    grassmann_loss: float  # Subspace distance
    eigenvalue_loss: float  # |lambda_pred - lambda_true|
    energy_regularization: float  # ||velocity||^2
    total_loss: float


def compute_grassmann_distance(
    Phi1: NDArray[np.floating],
    Phi2: NDArray[np.floating],
) -> float:
    """
    Compute Grassmann distance between subspaces.

    The Grassmann manifold Gr(n, k) is the space of k-dimensional subspaces
    in R^n. Distance is based on principal angles between subspaces.
    """
    # SVD of Phi1^T @ Phi2 gives cos of principal angles
    _, s, _ = np.linalg.svd(Phi1.T @ Phi2, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)

    # Principal angles
    angles = np.arccos(s)

    # Grassmann distance (chordal metric)
    return float(np.linalg.norm(np.sin(angles)))


def compute_stiefel_distance(
    Phi1: NDArray[np.floating],
    Phi2: NDArray[np.floating],
) -> float:
    """Compute geodesic distance on Stiefel manifold."""
    return StiefelManifold.geodesic_distance(Phi1, Phi2)


def simulate_flow_prediction(
    Phi_t: NDArray[np.floating],
    lambda_t: NDArray[np.floating],
    Phi_tp1_true: NDArray[np.floating],
    lambda_tp1_true: NDArray[np.floating],
    noise_level: float = 0.1,
    trained_epochs: int = 0,
) -> FlowPrediction:
    """
    Simulate a flow model prediction.

    As training progresses (trained_epochs increases), the prediction
    improves toward the true target.
    """
    # Compute true velocity (target)
    velocity_Phi_true = Phi_tp1_true - Phi_t
    velocity_lambda_true = lambda_tp1_true - lambda_t

    # Simulate model prediction quality based on training progress
    # Early training: mostly noise, later: converges to true
    quality = min(1.0, trained_epochs / 100.0)  # saturates at 100 epochs

    rng = np.random.default_rng(42 + trained_epochs)

    # Predicted velocity = quality * true_velocity + (1-quality) * noise
    noise_Phi = rng.standard_normal(Phi_t.shape) * noise_level
    noise_lambda = rng.standard_normal(lambda_t.shape) * noise_level

    velocity_Phi = quality * velocity_Phi_true + (1 - quality) * noise_Phi
    velocity_lambda = quality * velocity_lambda_true + (1 - quality) * noise_lambda

    # Project velocity to tangent space of Stiefel
    velocity_Phi = StiefelManifold.project_to_tangent(Phi_t, velocity_Phi)

    # Integrate one step using QR retraction
    Phi_pred = StiefelManifold.retract_qr(Phi_t, velocity_Phi, t=1.0)
    lambda_pred = lambda_t + velocity_lambda

    return FlowPrediction(
        Phi_pred=Phi_pred,
        lambda_pred=lambda_pred,
        velocity_Phi=velocity_Phi,
        velocity_lambda=velocity_lambda,
    )


def compute_losses(
    prediction: FlowPrediction,
    Phi_tp1_true: NDArray[np.floating],
    lambda_tp1_true: NDArray[np.floating],
    grassmann_weight: float = 1.0,
    eigenvalue_weight: float = 1.0,
    energy_weight: float = 0.01,
) -> FlowLosses:
    """
    Compute all losses for temporal geodesic flow matching.

    Losses:
    1. Grassmann loss: Distance between predicted and true subspaces
    2. Eigenvalue loss: MSE between predicted and true eigenvalues
    3. Energy regularization: Penalize large velocities
    """
    # Grassmann loss (subspace distance)
    grassmann_loss = compute_grassmann_distance(
        prediction.Phi_pred, Phi_tp1_true
    )

    # Eigenvalue loss
    eigenvalue_loss = float(np.mean((prediction.lambda_pred - lambda_tp1_true) ** 2))

    # Energy regularization
    energy_Phi = float(np.sum(prediction.velocity_Phi ** 2))
    energy_lambda = float(np.sum(prediction.velocity_lambda ** 2))
    energy_regularization = energy_Phi + energy_lambda

    # Total loss
    total_loss = (
        grassmann_weight * grassmann_loss +
        eigenvalue_weight * eigenvalue_loss +
        energy_weight * energy_regularization
    )

    return FlowLosses(
        grassmann_loss=grassmann_loss,
        eigenvalue_loss=eigenvalue_loss,
        energy_regularization=energy_regularization,
        total_loss=total_loss,
    )


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "time_idx": 0,
        "playing": False,
        "data": None,
        "params_hash": None,
        "trained_epochs": 0,
        "loss_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_spectral_sequence(
    dataset: str,
    n_timesteps: int,
    **kwargs,
) -> tuple[list, NDArray, dict]:
    """Generate a temporal spectral sequence based on dataset type."""
    if dataset == "merging_clusters":
        frames, labels = generate_merging_clusters(
            n_points_per_cluster=kwargs.get("n_points", 50),
            cluster_std=kwargs.get("cluster_std", 0.3),
            separation_start=kwargs.get("separation_start", 3.0),
            separation_end=0.0,
            k_neighbors=kwargs.get("k_neighbors", 10),
            n_eigenvectors=kwargs.get("n_eigenvectors", 6),
            n_timesteps=n_timesteps,
        )
        info = {"type": "merging_clusters"}
    else:
        frames, labels = generate_evolving_graph(
            n_nodes_per_ring=kwargs.get("n_nodes", 20),
            coupling_start=kwargs.get("coupling_start", 0.0),
            coupling_end=kwargs.get("coupling_end", 1.0),
            n_eigenvectors=kwargs.get("n_eigenvectors", 6),
            n_timesteps=n_timesteps,
        )
        info = {"type": "evolving_graph"}

    # Align the sequence
    aligner = SpectralAligner(
        matcher=SpectralMatcher(cost_type="absolute"),
        sign_convention=SignConvention(method="max_entry"),
    )

    Phi_sequence = [f.Phi for f in frames]
    lambda_sequence = [f.eigenvalues for f in frames]

    aligned_pairs = aligner.align_sequence(Phi_sequence, lambda_sequence)

    # Build aligned sequences
    Phi_aligned = [Phi_sequence[0]]
    lambda_aligned = [lambda_sequence[0]]

    for pair in aligned_pairs:
        Phi_aligned.append(pair.Phi_target_aligned)
        lambda_aligned.append(pair.lambda_target_aligned)

    return frames, labels, {
        "Phi_aligned": Phi_aligned,
        "lambda_aligned": lambda_aligned,
        "aligned_pairs": aligned_pairs,
        **info,
    }


def plot_flow_step(
    Phi_t: NDArray[np.floating],
    Phi_tp1_true: NDArray[np.floating],
    Phi_tp1_pred: NDArray[np.floating],
    labels: Optional[NDArray] = None,
    palette: Optional[ColorPalette] = None,
    title: str = "One-Step Flow Integration",
) -> plt.Figure:
    """
    Plot the one-step flow integration showing:
    - Current state (Phi_t)
    - True next state (Phi_{t+1})
    - Predicted next state (Phi_pred)
    """
    if palette is None:
        palette = ColorPalette()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Use first two eigenvectors for 2D visualization
    def scatter_embedding(ax, Phi, title_text, color_idx=0):
        if labels is not None:
            colors = [palette.source if l == 0 else palette.target for l in labels]
        else:
            colors = get_time_colors(len(Phi))[color_idx] if isinstance(color_idx, int) else color_idx

        ax.scatter(Phi[:, 0], Phi[:, 1], c=colors, s=30, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_xlabel(r"$\phi_1$", fontsize=10)
        ax.set_ylabel(r"$\phi_2$", fontsize=10)
        ax.set_title(title_text, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    scatter_embedding(axes[0], Phi_t, r"$\Phi_t$ (Current)")
    scatter_embedding(axes[1], Phi_tp1_true, r"$\Phi_{t+1}$ (True)")
    scatter_embedding(axes[2], Phi_tp1_pred, r"$\hat{\Phi}_{t+1}$ (Predicted)")

    # Compute and show distances
    grassmann_dist = compute_grassmann_distance(Phi_tp1_pred, Phi_tp1_true)
    axes[2].text(
        0.02, 0.98, f"Grassmann dist: {grassmann_dist:.4f}",
        transform=axes[2].transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_eigenvalue_evolution(
    lambda_sequence: list[NDArray[np.floating]],
    lambda_pred: Optional[NDArray[np.floating]] = None,
    current_idx: int = 0,
    style: Optional[VizStyle] = None,
) -> plt.Figure:
    """Plot eigenvalue trajectories with optional predicted values."""
    if style is None:
        style = VizStyle(mode="screen")

    fig, ax = plt.subplots(figsize=(8, 4))

    n_timesteps = len(lambda_sequence)
    n_eigenvalues = lambda_sequence[0].shape[0]

    # Stack into array for plotting
    lambda_array = np.array(lambda_sequence)

    # Plot each eigenvalue trajectory
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_eigenvalues))
    for i in range(n_eigenvalues):
        ax.plot(
            range(n_timesteps), lambda_array[:, i],
            color=colors[i], linewidth=1.5, alpha=0.7,
            label=rf"$\lambda_{{{i+1}}}$",
        )

    # Mark current timestep
    ax.axvline(current_idx, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current $t$')

    # Plot predicted eigenvalues if available
    if lambda_pred is not None and current_idx < n_timesteps - 1:
        for i in range(n_eigenvalues):
            ax.scatter(
                [current_idx + 1], [lambda_pred[i]],
                color=colors[i], s=100, marker='x', linewidths=2,
                zorder=10,
            )
        ax.scatter([], [], color='gray', s=100, marker='x', linewidths=2, label=r"$\hat{\lambda}$ (pred)")

    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel(r"Eigenvalue $\lambda$", fontsize=11)
    ax.set_title("Eigenvalue Evolution", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_velocity_field(
    Phi: NDArray[np.floating],
    velocity: NDArray[np.floating],
    labels: Optional[NDArray] = None,
    palette: Optional[ColorPalette] = None,
    scale: float = 1.0,
) -> plt.Figure:
    """Plot the velocity field (tangent vectors) at current state."""
    if palette is None:
        palette = ColorPalette()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter current positions
    if labels is not None:
        colors = [palette.source if l == 0 else palette.target for l in labels]
    else:
        colors = palette.primary

    ax.scatter(Phi[:, 0], Phi[:, 1], c=colors, s=40, alpha=0.6, edgecolors='white', linewidths=0.5)

    # Draw velocity arrows
    ax.quiver(
        Phi[:, 0], Phi[:, 1],
        velocity[:, 0] * scale, velocity[:, 1] * scale,
        angles='xy', scale_units='xy', scale=1,
        color='red', alpha=0.5, width=0.005,
    )

    ax.set_xlabel(r"$\phi_1$", fontsize=11)
    ax.set_ylabel(r"$\phi_2$", fontsize=11)
    ax.set_title(r"Velocity Field $\dot{\Phi}$", fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_loss_curves(loss_history: list[FlowLosses]) -> plt.Figure:
    """Plot training loss curves."""
    if len(loss_history) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Train to see loss curves", ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    epochs = range(len(loss_history))

    # Total loss
    axes[0, 0].plot(epochs, [l.total_loss for l in loss_history], 'b-', linewidth=1.5)
    axes[0, 0].set_title("Total Loss", fontsize=10)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    # Grassmann loss
    axes[0, 1].plot(epochs, [l.grassmann_loss for l in loss_history], 'g-', linewidth=1.5)
    axes[0, 1].set_title("Grassmann Loss (Subspace)", fontsize=10)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    # Eigenvalue loss
    axes[1, 0].plot(epochs, [l.eigenvalue_loss for l in loss_history], 'r-', linewidth=1.5)
    axes[1, 0].set_title("Eigenvalue Loss", fontsize=10)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    # Energy regularization
    axes[1, 1].plot(epochs, [l.energy_regularization for l in loss_history], 'm-', linewidth=1.5)
    axes[1, 1].set_title("Energy Regularization", fontsize=10)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Training Losses", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: uv add streamlit")
        return

    st.set_page_config(
        page_title="Temporal Geodesic Flow Matching",
        page_icon="🌀",
        layout="wide",
    )

    init_session_state()

    # Header
    st.title("🌀 Temporal Geodesic Flow Matching")
    st.markdown(
        """
        **Training on consecutive spectral pairs** — No sampling from noise!

        The model learns to integrate one step: $(\\Phi_t, \\lambda_t) \\rightarrow (\\hat{\\Phi}_{t+1}, \\hat{\\lambda}_{t+1})$

        **Losses**: Grassmann (subspace) + Eigenvalue + Energy regularization
        """
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Data Generation")

        dataset = st.selectbox(
            "Dataset",
            options=["merging_clusters", "evolving_graph"],
            format_func=lambda x: {
                "merging_clusters": "Merging Clusters",
                "evolving_graph": "Evolving Graph",
            }.get(x, x),
        )

        n_timesteps = st.slider("Timesteps", 20, 100, 50, 10)
        n_eigenvectors = st.slider("Eigenvectors (k)", 3, 10, 6, 1)

        if dataset == "merging_clusters":
            n_points = st.slider("Points per cluster", 20, 100, 50, 10)
            separation_start = st.slider("Initial separation", 1.0, 6.0, 3.0, 0.5)
            data_kwargs = {
                "n_points": n_points,
                "separation_start": separation_start,
                "n_eigenvectors": n_eigenvectors,
            }
        else:
            n_nodes = st.slider("Nodes per ring", 10, 50, 20, 5)
            data_kwargs = {
                "n_nodes": n_nodes,
                "n_eigenvectors": n_eigenvectors,
            }

        st.divider()
        st.header("Model (Simulated)")

        noise_level = st.slider("Untrained noise", 0.0, 0.5, 0.15, 0.05)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train +10 epochs"):
                st.session_state.trained_epochs += 10
        with col2:
            if st.button("Reset training"):
                st.session_state.trained_epochs = 0
                st.session_state.loss_history = []

        st.metric("Trained epochs", st.session_state.trained_epochs)

        st.divider()
        st.header("Loss Weights")
        grassmann_weight = st.slider("Grassmann weight", 0.0, 2.0, 1.0, 0.1)
        eigenvalue_weight = st.slider("Eigenvalue weight", 0.0, 2.0, 1.0, 0.1)
        energy_weight = st.slider("Energy weight", 0.0, 0.1, 0.01, 0.005)

        st.divider()
        if st.button("🔄 Regenerate Data"):
            st.session_state.data = None
            st.session_state.time_idx = 0
            st.session_state.loss_history = []

    # Generate/load data
    params_hash = f"{dataset}_{n_timesteps}_{n_eigenvectors}_{hash(frozenset(data_kwargs.items()))}"

    if st.session_state.data is None or st.session_state.params_hash != params_hash:
        with st.spinner("Generating spectral sequence..."):
            frames, labels, alignment_data = generate_spectral_sequence(
                dataset, n_timesteps, **data_kwargs
            )
            st.session_state.data = {
                "frames": frames,
                "labels": labels,
                **alignment_data,
            }
            st.session_state.params_hash = params_hash
            st.session_state.time_idx = 0
            st.session_state.loss_history = []

    data = st.session_state.data
    Phi_aligned = data["Phi_aligned"]
    lambda_aligned = data["lambda_aligned"]
    labels = data["labels"]
    n_frames = len(Phi_aligned)

    # Time controls
    st.divider()

    col_slider, col_play, col_reset, col_speed = st.columns([6, 1, 1, 2])

    with col_slider:
        time_idx = st.slider(
            "Time step $t$",
            min_value=0,
            max_value=n_frames - 2,  # Need t and t+1
            value=st.session_state.time_idx,
            key="time_slider",
        )
        st.session_state.time_idx = time_idx

    with col_play:
        if st.button("▶️" if not st.session_state.playing else "⏸️"):
            st.session_state.playing = not st.session_state.playing

    with col_reset:
        if st.button("⏮️"):
            st.session_state.time_idx = 0
            st.session_state.playing = False

    with col_speed:
        speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)

    # Get current and next frames
    Phi_t = Phi_aligned[time_idx]
    lambda_t = lambda_aligned[time_idx]
    Phi_tp1_true = Phi_aligned[time_idx + 1]
    lambda_tp1_true = lambda_aligned[time_idx + 1]

    # Simulate flow prediction
    prediction = simulate_flow_prediction(
        Phi_t, lambda_t,
        Phi_tp1_true, lambda_tp1_true,
        noise_level=noise_level,
        trained_epochs=st.session_state.trained_epochs,
    )

    # Compute losses
    losses = compute_losses(
        prediction, Phi_tp1_true, lambda_tp1_true,
        grassmann_weight=grassmann_weight,
        eigenvalue_weight=eigenvalue_weight,
        energy_weight=energy_weight,
    )

    # Update loss history if training
    if st.session_state.trained_epochs > len(st.session_state.loss_history):
        # Simulate training progress
        for epoch in range(len(st.session_state.loss_history), st.session_state.trained_epochs):
            pred = simulate_flow_prediction(
                Phi_t, lambda_t, Phi_tp1_true, lambda_tp1_true,
                noise_level=noise_level, trained_epochs=epoch,
            )
            epoch_losses = compute_losses(
                pred, Phi_tp1_true, lambda_tp1_true,
                grassmann_weight, eigenvalue_weight, energy_weight,
            )
            st.session_state.loss_history.append(epoch_losses)

    # Main visualization
    st.divider()

    palette = ColorPalette()

    # Row 1: Flow step visualization
    st.subheader("One-Step Flow Integration")
    st.markdown(
        rf"Integrating from $t={time_idx}$ to $t={time_idx+1}$: "
        rf"$(\Phi_t, \lambda_t) \rightarrow (\hat{{\Phi}}_{{t+1}}, \hat{{\lambda}}_{{t+1}})$"
    )

    fig_flow = plot_flow_step(
        Phi_t, Phi_tp1_true, prediction.Phi_pred,
        labels=labels, palette=palette,
        title=f"Flow Step: t={time_idx} → t={time_idx+1}",
    )
    st.pyplot(fig_flow)
    plt.close(fig_flow)

    # Row 2: Three columns - eigenvalues, velocity, losses
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Eigenvalue Evolution")
        fig_eigen = plot_eigenvalue_evolution(
            lambda_aligned,
            lambda_pred=prediction.lambda_pred,
            current_idx=time_idx,
        )
        st.pyplot(fig_eigen)
        plt.close(fig_eigen)

    with col2:
        st.subheader("Velocity Field")
        fig_vel = plot_velocity_field(
            Phi_t, prediction.velocity_Phi,
            labels=labels, palette=palette,
            scale=3.0,
        )
        st.pyplot(fig_vel)
        plt.close(fig_vel)

    with col3:
        st.subheader("Training Losses")
        fig_loss = plot_loss_curves(st.session_state.loss_history)
        st.pyplot(fig_loss)
        plt.close(fig_loss)

    # Loss metrics
    st.divider()
    st.subheader("Current Step Losses")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            "Total Loss",
            f"{losses.total_loss:.4f}",
            help="Weighted sum of all losses",
        )

    with col_m2:
        st.metric(
            "Grassmann Loss",
            f"{losses.grassmann_loss:.4f}",
            help="Subspace distance: how far is predicted subspace from true?",
        )

    with col_m3:
        st.metric(
            "Eigenvalue Loss",
            f"{losses.eigenvalue_loss:.4f}",
            help="MSE between predicted and true eigenvalues",
        )

    with col_m4:
        st.metric(
            "Energy Reg.",
            f"{losses.energy_regularization:.4f}",
            help="Penalizes large velocities for stability",
        )

    # Additional info
    st.divider()

    with st.expander("Training Paradigm Details", expanded=False):
        st.markdown(
            r"""
            ### Temporal Geodesic Flow Matching

            **Key insight**: We don't sample from noise! Instead, we learn the actual
            temporal dynamics of spectral embeddings.

            **Training step**:
            ```
            train_step(model, Phi_t, lambda_t, Phi_{t+1}, lambda_{t+1}, optimizer)
            ```

            1. **Forward pass**: Integrate one step from $(\\Phi_t, \\lambda_t)$ to get
               $(\\hat{\\Phi}_{t+1}, \\hat{\\lambda}_{t+1})$

            2. **Grassmann loss**: Distance between predicted and true *subspaces*
               $$\\mathcal{L}_\\text{Gr} = d_\\text{Gr}(\\text{span}(\\hat{\\Phi}_{t+1}), \\text{span}(\\Phi_{t+1}))$$

            3. **Eigenvalue loss**: MSE on eigenvalues
               $$\\mathcal{L}_\\lambda = \\|\\hat{\\lambda}_{t+1} - \\lambda_{t+1}\\|^2$$

            4. **Energy regularization**: Penalize large velocities
               $$\\mathcal{L}_\\text{reg} = \\|\\dot{\\Phi}\\|_F^2 + \\|\\dot{\\lambda}\\|^2$$

            5. **Backprop** and update model parameters

            **No noise sampling. No reference geodesics.**
            """
        )

    # Animation loop
    if st.session_state.playing:
        new_idx = st.session_state.time_idx + 1
        if new_idx >= n_frames - 1:
            new_idx = 0
            st.session_state.playing = False
        st.session_state.time_idx = new_idx
        time.sleep(0.1 / speed)
        st.rerun()


if __name__ == "__main__":
    main()
