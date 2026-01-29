"""
Streamlit page: Evolving Graph with Temporal Geodesic Flow.

Demonstrates temporal geodesic flow matching on two ring graphs
that gradually couple. Shows:
- One-step flow integration with eigenvalue crossings
- Grassmann loss between predicted and true subspaces
- Mode tracking through crossings
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

from viz.generators import generate_evolving_graph, SpectralFrame
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


def plot_graph_structure(
    n1: int,
    n2: int,
    coupling: float,
    labels: np.ndarray,
    style: VizStyle,
) -> plt.Figure:
    """Plot the current graph structure (two rings with coupling)."""
    fig, ax = plt.subplots(figsize=(style.fig_width * 0.6, style.fig_height * 0.5))

    # Ring 1 positions
    theta1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    x1 = 0.8 * np.cos(theta1) - 1.2
    y1 = 0.8 * np.sin(theta1)

    # Ring 2 positions
    theta2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)
    x2 = 0.6 * np.cos(theta2) + 1.0
    y2 = 0.6 * np.sin(theta2)

    # Draw ring edges
    for i in range(n1):
        j = (i + 1) % n1
        ax.plot([x1[i], x1[j]], [y1[i], y1[j]], 'b-', alpha=0.5, linewidth=1)

    for i in range(n2):
        j = (i + 1) % n2
        ax.plot([x2[i], x2[j]], [y2[i], y2[j]], 'r-', alpha=0.5, linewidth=1)

    # Draw coupling edges
    if coupling > 0:
        ax.plot(
            [x1[0], x2[0]], [y1[0], y2[0]],
            'g-', linewidth=2 * coupling + 0.5, alpha=0.8,
            label=f'Coupling: {coupling:.2f}'
        )
        mid1 = n1 // 2
        mid2 = n2 // 2
        ax.plot(
            [x1[mid1], x2[mid2]], [y1[mid1], y2[mid2]],
            'g-', linewidth=coupling + 0.5, alpha=0.5,
        )

    ax.scatter(x1, y1, c='#3b82f6', s=40, zorder=10, label=f'Ring 1 (n={n1})')
    ax.scatter(x2, y2, c='#ef4444', s=40, zorder=10, label=f'Ring 2 (n={n2})')

    ax.set_xlim(-2.5, 2.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Graph Structure', fontsize=style.title_size)

    fig.tight_layout()
    return fig


def plot_eigenvalue_prediction(
    lambda_sequence: list[np.ndarray],
    lambda_pred: np.ndarray,
    current_idx: int,
) -> plt.Figure:
    """Plot eigenvalue trajectories with predicted values."""
    fig, ax = plt.subplots(figsize=(6, 4))

    n_timesteps = len(lambda_sequence)
    n_eigenvalues = lambda_sequence[0].shape[0]
    lambda_array = np.array(lambda_sequence)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_eigenvalues))
    for i in range(n_eigenvalues):
        ax.plot(
            range(n_timesteps), lambda_array[:, i],
            color=colors[i], linewidth=1.5, alpha=0.7,
            label=rf"$\lambda_{{{i+1}}}$",
        )

    ax.axvline(current_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Plot predicted eigenvalues
    if current_idx < n_timesteps - 1:
        for i in range(n_eigenvalues):
            ax.scatter(
                [current_idx + 1], [lambda_pred[i]],
                color=colors[i], s=100, marker='x', linewidths=2,
                zorder=10,
            )
        ax.scatter([], [], color='gray', s=100, marker='x', linewidths=2, label=r"$\hat{\lambda}$")

    ax.set_xlabel("Time step", fontsize=10)
    ax.set_ylabel(r"$\lambda$", fontsize=10)
    ax.set_title("Eigenvalue Evolution + Prediction", fontsize=11)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def main():
    """Main page content."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available.")
        return

    st.set_page_config(
        page_title="Evolving Graph - Flow Matching",
        page_icon="🔴",
        layout="wide",
    )

    st.title("🔴 Evolving Graph")
    st.markdown(
        """
        Two ring graphs gradually connect. The flow model tracks modes through
        **eigenvalue crossings**: $(\\Phi_t, \\lambda_t) \\rightarrow (\\hat{\\Phi}_{t+1}, \\hat{\\lambda}_{t+1})$
        """
    )

    # Initialize session state
    defaults = {
        "eg_time": 0.0,
        "eg_playing": False,
        "eg_data": None,
        "eg_params_hash": None,
        "eg_trained_epochs": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar controls
    with st.sidebar:
        st.header("Graph Parameters")

        n_nodes_ring1 = st.slider("Ring 1 nodes", 10, 50, 30, 5)
        n_nodes_ring2 = st.slider("Ring 2 nodes", 10, 50, 20, 5)

        coupling_schedule = st.selectbox(
            "Coupling schedule",
            options=["linear", "exponential", "step"],
            index=0,
        )

        n_eigenvectors = st.slider("Eigenvectors (k)", 3, 10, 6, 1)
        n_timesteps = st.slider("Timesteps", 20, 100, 50, 10)

        st.divider()
        st.header("Flow Model (Simulated)")

        noise_level = st.slider("Untrained noise", 0.0, 0.5, 0.15, 0.05)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train +10"):
                st.session_state.eg_trained_epochs += 10
        with col2:
            if st.button("Reset"):
                st.session_state.eg_trained_epochs = 0

        st.metric("Epochs", st.session_state.eg_trained_epochs)

        st.divider()
        st.header("Visualization")
        show_comparison = st.checkbox("Show raw vs aligned", value=False)
        trail_length = st.slider("Trail length", 0, 20, 5, 1)
        show_graph = st.checkbox("Show graph structure", value=True)

        st.divider()
        if st.button("🔄 Regenerate"):
            st.session_state.eg_data = None
            st.session_state.eg_time = 0.0

    # Compute params hash
    params_hash = f"{n_nodes_ring1}_{n_nodes_ring2}_{coupling_schedule}_{n_eigenvectors}_{n_timesteps}"

    # Generate or load data
    if st.session_state.eg_data is None or st.session_state.eg_params_hash != params_hash:
        with st.spinner("Generating spectral sequence..."):
            frames, labels = generate_evolving_graph(
                n_nodes_ring1=n_nodes_ring1,
                n_nodes_ring2=n_nodes_ring2,
                coupling_start=0.0,
                coupling_end=1.0,
                coupling_schedule=coupling_schedule,
                n_eigenvectors=n_eigenvectors,
                n_timesteps=n_timesteps,
            )

            alignments = compute_alignments(frames, labels)

            st.session_state.eg_data = {
                "frames": frames,
                "labels": labels,
                "n_nodes_ring1": n_nodes_ring1,
                "n_nodes_ring2": n_nodes_ring2,
                **alignments,
            }
            st.session_state.eg_params_hash = params_hash
            st.session_state.eg_time = 0.0

    data = st.session_state.eg_data
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
            value=st.session_state.eg_time,
            step=1.0 / max(n_timesteps_actual - 1, 1),
            key="eg_time_slider",
        )
        st.session_state.eg_time = t

    with col_play:
        if st.button("▶️" if not st.session_state.eg_playing else "⏸️"):
            st.session_state.eg_playing = not st.session_state.eg_playing

    with col_reset:
        if st.button("⏮️"):
            st.session_state.eg_time = 0.0
            st.session_state.eg_playing = False

    with col_speed:
        speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)

    # Current index (need t and t+1)
    current_idx = int(t * (n_timesteps_actual - 1))
    current_idx = min(current_idx, n_timesteps_actual - 2)

    # Compute current coupling
    if coupling_schedule == "linear":
        current_coupling = t
    elif coupling_schedule == "exponential":
        current_coupling = t ** 2
    else:
        current_coupling = 1.0 if t >= 0.5 else 0.0

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
        trained_epochs=st.session_state.eg_trained_epochs,
    )

    losses = compute_losses(prediction, Phi_tp1_true, lambda_tp1_true)

    style = VizStyle(mode="screen")
    palette = ColorPalette()

    # Display panels
    st.divider()

    if show_graph:
        col0, col1, col2, col3 = st.columns([1, 1, 1, 1])

        with col0:
            st.subheader("Graph")
            fig0 = plot_graph_structure(
                n1=data["n_nodes_ring1"],
                n2=data["n_nodes_ring2"],
                coupling=current_coupling,
                labels=labels,
                style=style,
            )
            st.pyplot(fig0)
            plt.close(fig0)
    else:
        col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Eigenvalues + Prediction")
        fig1 = plot_eigenvalue_prediction(
            lambda_sequence=lambda_aligned,
            lambda_pred=prediction.lambda_pred,
            current_idx=current_idx,
        )
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Eigenvectors")
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
        st.subheader("Embedding")
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

    # Info panel
    st.divider()

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)

    with col_info1:
        st.metric("Time Step", f"{current_idx}")
        st.metric("Coupling", f"{current_coupling:.3f}")

    with col_info2:
        current_frame = frames[current_idx]
        eig_str = ", ".join([f"{e:.3f}" for e in current_frame.eigenvalues[:3]])
        st.metric("λ₁, λ₂, λ₃", eig_str)

    with col_info3:
        if current_idx > 0 and current_idx <= len(data["aligned_pairs"]):
            pair = data["aligned_pairs"][current_idx - 1]
            st.metric("Alignment Cost", f"{pair.matching_cost:.4f}")

            perm = pair.eigenvalue_permutation
            is_crossing = not np.array_equal(perm, np.arange(len(perm)))
            st.metric("Crossing Detected", "Yes" if is_crossing else "No")
        else:
            st.metric("Alignment Cost", "N/A")
            st.metric("Crossing Detected", "N/A")

    with col_info4:
        if len(current_frame.eigenvalues) >= 2:
            spectral_gap = current_frame.eigenvalues[1] - current_frame.eigenvalues[0]
            st.metric("Spectral Gap", f"{spectral_gap:.4f}")

    # Animation loop
    if st.session_state.eg_playing:
        new_t = st.session_state.eg_time + (0.02 * speed)
        max_t = 1.0 - 1.0 / max(n_timesteps_actual - 1, 1)
        if new_t >= max_t:
            new_t = 0.0
            st.session_state.eg_playing = False
        st.session_state.eg_time = new_t
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
