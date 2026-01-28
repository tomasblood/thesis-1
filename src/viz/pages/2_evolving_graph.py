"""
Streamlit page: Evolving Graph toy example.

Demonstrates temporal spectral alignment on two ring graphs
that gradually couple via connecting edges.
"""

import sys
import time
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
from viz.styles import VizStyle

# Import alignment module
from temporal_spectral_flow.alignment import SpectralAligner, SpectralMatcher, SignConvention


def compute_alignments(frames: list, labels: np.ndarray) -> dict:
    """Compute spectral alignments for the sequence."""
    aligner = SpectralAligner(
        matcher=SpectralMatcher(cost_type="absolute"),
        sign_convention=SignConvention(method="max_entry"),
    )

    # Extract sequences
    Phi_sequence = [f.Phi for f in frames]
    lambda_sequence = [f.eigenvalues for f in frames]

    # Align sequence
    aligned_pairs = aligner.align_sequence(Phi_sequence, lambda_sequence)

    # Build aligned sequences (first frame is reference)
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
        # Main coupling
        ax.plot(
            [x1[0], x2[0]], [y1[0], y2[0]],
            'g-', linewidth=2 * coupling + 0.5, alpha=0.8,
            label=f'Coupling: {coupling:.2f}'
        )
        # Secondary coupling
        mid1 = n1 // 2
        mid2 = n2 // 2
        ax.plot(
            [x1[mid1], x2[mid2]], [y1[mid1], y2[mid2]],
            'g-', linewidth=coupling + 0.5, alpha=0.5,
        )

    # Draw nodes
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


def main():
    """Main page content."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available.")
        return

    st.set_page_config(
        page_title="Evolving Graph - Spectral Alignment",
        page_icon="🔴",
        layout="wide",
    )

    st.title("🔴 Evolving Graph")
    st.markdown(
        """
        Two ring graphs gradually connect. Watch eigenvalue crossings
        as modes "see" each other and new global modes emerge.
        """
    )

    # Initialize session state
    if "eg_time" not in st.session_state:
        st.session_state.eg_time = 0.0
    if "eg_playing" not in st.session_state:
        st.session_state.eg_playing = False
    if "eg_data" not in st.session_state:
        st.session_state.eg_data = None
    if "eg_params_hash" not in st.session_state:
        st.session_state.eg_params_hash = None

    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")

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

        st.header("Visualization")
        show_comparison = st.checkbox("Show raw vs aligned", value=False)
        trail_length = st.slider("Trail length", 0, 20, 5, 1)
        show_graph = st.checkbox("Show graph structure", value=True)

        st.divider()

        if st.button("🔄 Regenerate"):
            st.session_state.eg_data = None
            st.session_state.eg_time = 0.0

    # Compute params hash for cache invalidation
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
            max_value=1.0,
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

    # Compute current index and coupling
    current_idx = int(t * (n_timesteps_actual - 1))
    current_idx = min(current_idx, n_timesteps_actual - 1)

    # Compute current coupling based on schedule
    if coupling_schedule == "linear":
        current_coupling = t
    elif coupling_schedule == "exponential":
        current_coupling = t ** 2
    else:  # step
        current_coupling = 1.0 if t >= 0.5 else 0.0

    # Display panels
    st.divider()

    style = VizStyle(mode="screen")

    if show_graph:
        # Four panels: graph structure + three spectral panels
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
        st.subheader("Eigenvalues")
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

    # Info panel
    st.divider()

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)

    with col_info1:
        st.metric("Current Time", f"{t:.3f}")
        st.metric("Coupling Strength", f"{current_coupling:.3f}")

    with col_info2:
        current_frame = frames[current_idx]
        # Show first few eigenvalues
        eig_str = ", ".join([f"{e:.3f}" for e in current_frame.eigenvalues[:3]])
        st.metric("λ₁, λ₂, λ₃", eig_str)

    with col_info3:
        if current_idx > 0:
            pair = data["aligned_pairs"][current_idx - 1]
            st.metric("Alignment Cost", f"{pair.matching_cost:.4f}")

            # Check for non-identity permutation (crossing)
            perm = pair.eigenvalue_permutation
            is_crossing = not np.array_equal(perm, np.arange(len(perm)))
            st.metric("Crossing Detected", "Yes" if is_crossing else "No")
        else:
            st.metric("Alignment Cost", "N/A")
            st.metric("Crossing Detected", "N/A")

    with col_info4:
        # Spectral gap (difference between λ₁ and λ₂)
        if len(current_frame.eigenvalues) >= 2:
            spectral_gap = current_frame.eigenvalues[1] - current_frame.eigenvalues[0]
            st.metric("Spectral Gap (λ₂-λ₁)", f"{spectral_gap:.4f}")

    # Animation loop
    if st.session_state.eg_playing:
        new_t = st.session_state.eg_time + (0.02 * speed)
        if new_t >= 1.0:
            new_t = 0.0
            st.session_state.eg_playing = False
        st.session_state.eg_time = new_t
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
