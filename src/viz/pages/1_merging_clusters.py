"""
Streamlit page: Merging Clusters with Temporal Geodesic Flow.

Demonstrates temporal geodesic flow matching on two Gaussian clusters
that gradually merge. Shows:
- One-step flow integration
- Grassmann loss between predicted and true subspaces
- Eigenvalue prediction
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from viz.generators import generate_merging_clusters, SpectralFrame
from viz.components import (
    plot_eigenvalue_trajectories,
    plot_eigenvector_heatmap,
    plot_spectral_embedding,
)
from viz.styles import VizStyle, ColorPalette

# Import alignment and flow machinery
from temporal_spectral_flow.alignment import SpectralAligner, SpectralMatcher, SignConvention
from temporal_spectral_flow.stiefel import StiefelManifold


@dataclass
class FlowPrediction:
    """Prediction from one-step flow integration."""
    Phi_pred: np.ndarray
    lambda_pred: np.ndarray
    velocity_Phi: np.ndarray
    velocity_lambda: np.ndarray


@dataclass
class FlowLosses:
    """Losses for temporal geodesic flow matching."""
    grassmann_loss: float
    eigenvalue_loss: float
    energy_regularization: float
    total_loss: float


def compute_grassmann_distance(Phi1: np.ndarray, Phi2: np.ndarray) -> float:
    """Compute Grassmann distance between subspaces."""
    _, s, _ = np.linalg.svd(Phi1.T @ Phi2, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return float(np.linalg.norm(np.sin(angles)))


def simulate_flow_prediction(
    Phi_t: np.ndarray,
    lambda_t: np.ndarray,
    Phi_tp1_true: np.ndarray,
    lambda_tp1_true: np.ndarray,
    noise_level: float = 0.1,
    trained_epochs: int = 0,
) -> FlowPrediction:
    """Simulate a flow model prediction."""
    velocity_Phi_true = Phi_tp1_true - Phi_t
    velocity_lambda_true = lambda_tp1_true - lambda_t

    quality = min(1.0, trained_epochs / 100.0)
    rng = np.random.default_rng(42 + trained_epochs)

    noise_Phi = rng.standard_normal(Phi_t.shape) * noise_level
    noise_lambda = rng.standard_normal(lambda_t.shape) * noise_level

    velocity_Phi = quality * velocity_Phi_true + (1 - quality) * noise_Phi
    velocity_lambda = quality * velocity_lambda_true + (1 - quality) * noise_lambda

    velocity_Phi = StiefelManifold.project_to_tangent(Phi_t, velocity_Phi)
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
    Phi_tp1_true: np.ndarray,
    lambda_tp1_true: np.ndarray,
) -> FlowLosses:
    """Compute all losses for temporal geodesic flow matching."""
    grassmann_loss = compute_grassmann_distance(prediction.Phi_pred, Phi_tp1_true)
    eigenvalue_loss = float(np.mean((prediction.lambda_pred - lambda_tp1_true) ** 2))
    energy_Phi = float(np.sum(prediction.velocity_Phi ** 2))
    energy_lambda = float(np.sum(prediction.velocity_lambda ** 2))
    energy_regularization = energy_Phi + energy_lambda

    total_loss = grassmann_loss + eigenvalue_loss + 0.01 * energy_regularization

    return FlowLosses(
        grassmann_loss=grassmann_loss,
        eigenvalue_loss=eigenvalue_loss,
        energy_regularization=energy_regularization,
        total_loss=total_loss,
    )


def compute_alignments(frames: list, labels: np.ndarray) -> dict:
    """Compute spectral alignments for the sequence."""
    aligner = SpectralAligner(
        matcher=SpectralMatcher(cost_type="absolute"),
        sign_convention=SignConvention(method="max_entry"),
    )

    Phi_sequence = [f.Phi for f in frames]
    lambda_sequence = [f.eigenvalues for f in frames]

    aligned_pairs = aligner.align_sequence(Phi_sequence, lambda_sequence)

    Phi_aligned = [Phi_sequence[0]]
    lambda_aligned = [lambda_sequence[0]]

    for pair in aligned_pairs:
        Phi_aligned.append(pair.Phi_target_aligned)
        lambda_aligned.append(pair.lambda_target_aligned)

    return {
        "Phi_sequence": Phi_sequence,
        "lambda_sequence": lambda_sequence,
        "Phi_aligned": Phi_aligned,
        "lambda_aligned": lambda_aligned,
        "aligned_pairs": aligned_pairs,
    }


def plot_flow_comparison(
    Phi_t: np.ndarray,
    Phi_tp1_true: np.ndarray,
    Phi_tp1_pred: np.ndarray,
    labels: np.ndarray,
    palette: ColorPalette,
) -> plt.Figure:
    """Plot current, true next, and predicted next embeddings."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def scatter(ax, Phi, title):
        colors = [palette.source if l == 0 else palette.target for l in labels]
        ax.scatter(Phi[:, 0], Phi[:, 1], c=colors, s=30, alpha=0.7,
                   edgecolors='white', linewidths=0.5)
        ax.set_xlabel(r"$\phi_1$", fontsize=10)
        ax.set_ylabel(r"$\phi_2$", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    scatter(axes[0], Phi_t, r"$\Phi_t$ (Current)")
    scatter(axes[1], Phi_tp1_true, r"$\Phi_{t+1}$ (True)")
    scatter(axes[2], Phi_tp1_pred, r"$\hat{\Phi}_{t+1}$ (Predicted)")

    grassmann_dist = compute_grassmann_distance(Phi_tp1_pred, Phi_tp1_true)
    axes[2].text(
        0.02, 0.98, f"Grassmann: {grassmann_dist:.4f}",
        transform=axes[2].transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    fig.tight_layout()
    return fig


def main():
    """Main page content."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available.")
        return

    st.set_page_config(
        page_title="Merging Clusters - Flow Matching",
        page_icon="🔵",
        layout="wide",
    )

    st.title("🔵 Merging Clusters")
    st.markdown(
        """
        Two Gaussian clusters gradually merge. The flow model learns to predict
        **one-step evolution**: $(\\Phi_t, \\lambda_t) \\rightarrow (\\hat{\\Phi}_{t+1}, \\hat{\\lambda}_{t+1})$
        """
    )

    # Initialize session state
    defaults = {
        "mc_time": 0.0,
        "mc_playing": False,
        "mc_data": None,
        "mc_params_hash": None,
        "mc_trained_epochs": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar controls
    with st.sidebar:
        st.header("Data Parameters")

        n_points = st.slider("Points per cluster", 20, 100, 50, 10)
        cluster_std = st.slider("Cluster std", 0.1, 1.0, 0.3, 0.1)
        separation_start = st.slider("Initial separation", 1.0, 6.0, 3.0, 0.5)
        k_neighbors = st.slider("k neighbors", 5, 20, 10, 1)
        n_eigenvectors = st.slider("Eigenvectors (k)", 3, 10, 6, 1)
        n_timesteps = st.slider("Timesteps", 20, 100, 50, 10)

        st.divider()
        st.header("Flow Model (Simulated)")

        noise_level = st.slider("Untrained noise", 0.0, 0.5, 0.15, 0.05)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train +10"):
                st.session_state.mc_trained_epochs += 10
        with col2:
            if st.button("Reset"):
                st.session_state.mc_trained_epochs = 0

        st.metric("Epochs", st.session_state.mc_trained_epochs)

        st.divider()
        st.header("Visualization")
        show_comparison = st.checkbox("Show raw vs aligned", value=False)
        trail_length = st.slider("Trail length", 0, 20, 5, 1)

        st.divider()
        if st.button("🔄 Regenerate"):
            st.session_state.mc_data = None
            st.session_state.mc_time = 0.0

    # Compute params hash
    params_hash = f"{n_points}_{cluster_std}_{separation_start}_{k_neighbors}_{n_eigenvectors}_{n_timesteps}"

    # Generate or load data
    if st.session_state.mc_data is None or st.session_state.mc_params_hash != params_hash:
        with st.spinner("Generating spectral sequence..."):
            frames, labels = generate_merging_clusters(
                n_points_per_cluster=n_points,
                cluster_std=cluster_std,
                separation_start=separation_start,
                separation_end=0.0,
                k_neighbors=k_neighbors,
                n_eigenvectors=n_eigenvectors,
                n_timesteps=n_timesteps,
            )

            alignments = compute_alignments(frames, labels)

            st.session_state.mc_data = {
                "frames": frames,
                "labels": labels,
                **alignments,
            }
            st.session_state.mc_params_hash = params_hash
            st.session_state.mc_time = 0.0

    data = st.session_state.mc_data
    frames = data["frames"]
    labels = data["labels"]
    n_timesteps_actual = len(frames)

    # Time controls
    st.divider()

    col_slider, col_play, col_reset, col_speed = st.columns([6, 1, 1, 2])

    with col_slider:
        t = st.slider(
            "Time $t$",
            min_value=0.0,
            max_value=1.0 - 1.0 / max(n_timesteps_actual - 1, 1),
            value=st.session_state.mc_time,
            step=1.0 / max(n_timesteps_actual - 1, 1),
            key="mc_time_slider",
        )
        st.session_state.mc_time = t

    with col_play:
        if st.button("▶️" if not st.session_state.mc_playing else "⏸️"):
            st.session_state.mc_playing = not st.session_state.mc_playing

    with col_reset:
        if st.button("⏮️"):
            st.session_state.mc_time = 0.0
            st.session_state.mc_playing = False

    with col_speed:
        speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)

    # Current index (need t and t+1)
    current_idx = int(t * (n_timesteps_actual - 1))
    current_idx = min(current_idx, n_timesteps_actual - 2)

    # Get aligned data
    Phi_aligned = data["Phi_aligned"]
    lambda_aligned = data["lambda_aligned"]

    Phi_t = Phi_aligned[current_idx]
    lambda_t = lambda_aligned[current_idx]
    Phi_tp1_true = Phi_aligned[current_idx + 1]
    lambda_tp1_true = lambda_aligned[current_idx + 1]

    # Simulate flow prediction
    prediction = simulate_flow_prediction(
        Phi_t, lambda_t, Phi_tp1_true, lambda_tp1_true,
        noise_level=noise_level,
        trained_epochs=st.session_state.mc_trained_epochs,
    )

    losses = compute_losses(prediction, Phi_tp1_true, lambda_tp1_true)

    palette = ColorPalette()
    style = VizStyle(mode="screen")

    # Flow step visualization
    st.divider()
    st.subheader("One-Step Flow Integration")
    st.markdown(
        rf"Integrating from $t={current_idx}$ to $t={current_idx+1}$"
    )

    fig_flow = plot_flow_comparison(
        Phi_t, Phi_tp1_true, prediction.Phi_pred,
        labels, palette,
    )
    st.pyplot(fig_flow)
    plt.close(fig_flow)

    # Three-panel display
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Eigenvalue Trajectories")
        fig1 = plot_eigenvalue_trajectories(
            lambda_sequence=data["lambda_sequence"],
            current_idx=current_idx,
            lambda_aligned=data["lambda_aligned"],
            show_raw=show_comparison,
            style=style,
        )
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Eigenvector Heatmap")
        current_Phi = data["Phi_sequence"][current_idx]
        current_Phi_aligned = data["Phi_aligned"][current_idx]

        sign_flips = None
        if current_idx > 0 and current_idx <= len(data["aligned_pairs"]):
            pair = data["aligned_pairs"][current_idx - 1]
            sign_flips = pair.sign_flips

        fig2 = plot_eigenvector_heatmap(
            Phi=current_Phi,
            Phi_aligned=current_Phi_aligned,
            show_comparison=show_comparison,
            sign_flips=sign_flips,
            style=style,
        )
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.subheader("Spectral Embedding")
        fig3 = plot_spectral_embedding(
            Phi_sequence=data["Phi_aligned"],
            current_idx=current_idx,
            labels=labels,
            trail_length=trail_length,
            style=style,
        )
        st.pyplot(fig3)
        plt.close(fig3)

    # Loss metrics
    st.divider()
    st.subheader("Flow Losses")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("Total Loss", f"{losses.total_loss:.4f}")

    with col_m2:
        st.metric(
            "Grassmann Loss",
            f"{losses.grassmann_loss:.4f}",
            help="Subspace distance between predicted and true",
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
            help="Velocity magnitude penalty",
        )

    # Additional info panel
    st.divider()

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        current_frame = frames[current_idx]
        st.metric("Time Step", f"{current_idx}")
        st.metric("λ₂ (Fiedler)", f"{current_frame.eigenvalues[0]:.4f}")

    with col_info2:
        if current_idx > 0 and current_idx <= len(data["aligned_pairs"]):
            pair = data["aligned_pairs"][current_idx - 1]
            st.metric("Alignment Cost", f"{pair.matching_cost:.4f}")
            perm_str = str(pair.eigenvalue_permutation.tolist())
            st.metric("Permutation", perm_str[:20] + "..." if len(perm_str) > 20 else perm_str)
        else:
            st.metric("Alignment Cost", "N/A (t=0)")
            st.metric("Permutation", "Identity")

    with col_info3:
        cluster1_mean = data["Phi_aligned"][current_idx][labels == 0].mean(axis=0)
        cluster2_mean = data["Phi_aligned"][current_idx][labels == 1].mean(axis=0)
        separation = np.linalg.norm(cluster1_mean - cluster2_mean)
        st.metric("Cluster Separation", f"{separation:.4f}")

    # Animation loop
    if st.session_state.mc_playing:
        new_t = st.session_state.mc_time + (0.02 * speed)
        max_t = 1.0 - 1.0 / max(n_timesteps_actual - 1, 1)
        if new_t >= max_t:
            new_t = 0.0
            st.session_state.mc_playing = False
        st.session_state.mc_time = new_t
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
