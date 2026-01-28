"""
Streamlit page: Merging Clusters toy example.

Demonstrates temporal spectral alignment on two Gaussian clusters
that gradually merge into one.
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

from viz.generators import generate_merging_clusters, SpectralFrame
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


def main():
    """Main page content."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available.")
        return

    st.set_page_config(
        page_title="Merging Clusters - Spectral Alignment",
        page_icon="🔵",
        layout="wide",
    )

    st.title("🔵 Merging Clusters")
    st.markdown(
        """
        Two Gaussian clusters gradually merge. Watch how the spectral
        embedding evolves and how eigenvalue matching keeps modes tracked.
        """
    )

    # Initialize session state
    if "mc_time" not in st.session_state:
        st.session_state.mc_time = 0.0
    if "mc_playing" not in st.session_state:
        st.session_state.mc_playing = False
    if "mc_data" not in st.session_state:
        st.session_state.mc_data = None
    if "mc_params_hash" not in st.session_state:
        st.session_state.mc_params_hash = None

    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")

        n_points = st.slider("Points per cluster", 20, 100, 50, 10)
        cluster_std = st.slider("Cluster std", 0.1, 1.0, 0.3, 0.1)
        separation_start = st.slider("Initial separation", 1.0, 6.0, 3.0, 0.5)
        k_neighbors = st.slider("k neighbors", 5, 20, 10, 1)
        n_eigenvectors = st.slider("Eigenvectors (k)", 3, 10, 6, 1)
        n_timesteps = st.slider("Timesteps", 20, 100, 50, 10)

        st.divider()

        st.header("Visualization")
        show_comparison = st.checkbox("Show raw vs aligned", value=False)
        trail_length = st.slider("Trail length", 0, 20, 5, 1)

        st.divider()

        if st.button("🔄 Regenerate"):
            st.session_state.mc_data = None
            st.session_state.mc_time = 0.0

    # Compute params hash for cache invalidation
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
            max_value=1.0,
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

    # Compute current index
    current_idx = int(t * (n_timesteps_actual - 1))
    current_idx = min(current_idx, n_timesteps_actual - 1)

    # Three-panel display
    st.divider()

    col1, col2, col3 = st.columns(3)

    style = VizStyle(mode="screen")

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

        # Get alignment info if available
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

    # Info panel
    st.divider()

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        current_frame = frames[current_idx]
        st.metric("Current Time", f"{t:.3f}")
        st.metric("λ₂ (Fiedler)", f"{current_frame.eigenvalues[0]:.4f}")

    with col_info2:
        if current_idx > 0:
            pair = data["aligned_pairs"][current_idx - 1]
            st.metric("Alignment Cost", f"{pair.matching_cost:.4f}")
            perm_str = str(pair.eigenvalue_permutation.tolist())
            st.metric("Permutation", perm_str[:20] + "..." if len(perm_str) > 20 else perm_str)
        else:
            st.metric("Alignment Cost", "N/A (t=0)")
            st.metric("Permutation", "Identity")

    with col_info3:
        # Cluster separation metric
        cluster1_mean = data["Phi_aligned"][current_idx][labels == 0].mean(axis=0)
        cluster2_mean = data["Phi_aligned"][current_idx][labels == 1].mean(axis=0)
        separation = np.linalg.norm(cluster1_mean - cluster2_mean)
        st.metric("Cluster Separation", f"{separation:.4f}")

    # Animation loop
    if st.session_state.mc_playing:
        new_t = st.session_state.mc_time + (0.02 * speed)
        if new_t >= 1.0:
            new_t = 0.0
            st.session_state.mc_playing = False
        st.session_state.mc_time = new_t
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
