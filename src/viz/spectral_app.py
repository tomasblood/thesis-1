"""
Streamlit app for Temporal Spectral Alignment visualization.

Interactive demonstration of how spectral embeddings (Phi, lambda)
evolve over time and how eigenvalue matching + sign conventions
keep them aligned.

Usage:
    streamlit run src/viz/spectral_app.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


def main():
    """Main entry point for Temporal Spectral Alignment app."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: uv add streamlit")
        return

    st.set_page_config(
        page_title="Temporal Spectral Alignment",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("Temporal Spectral Alignment")
    st.markdown(
        """
        Interactive visualization of how spectral embeddings evolve over time
        and how eigenvalue matching + sign conventions maintain consistency.
        """
    )

    st.divider()

    # Overview section
    st.header("Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("The Problem")
        st.markdown(
            """
            Spectral embeddings have **gauge ambiguity**:

            1. **Sign flips**: $\\phi$ and $-\\phi$ are both valid eigenvectors
            2. **Permutations**: Eigenvalue ordering can change between timesteps
            3. **Rotations**: Degenerate eigenspaces allow arbitrary rotations

            Naive comparison of eigenvectors across time fails because
            the "same" geometric mode can appear with different signs
            or at different indices.
            """
        )

    with col2:
        st.subheader("The Solution")
        st.markdown(
            """
            **Transport-Consistent Spectral Alignment**:

            1. **Eigenvalue matching**: Use proximity to track modes through crossings
            2. **Sign conventions**: Canonical rules that don't require node correspondence
            3. **Temporal stability**: Smooth evolution reveals signal vs noise

            This allows principled comparison of spectral structure across time.
            """
        )

    st.divider()

    # Example pages
    st.header("Interactive Examples")

    st.markdown(
        """
        Explore two toy examples that demonstrate temporal spectral alignment:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Merging Clusters")
        st.markdown(
            """
            Two Gaussian clusters gradually merge into one.

            **What to watch:**
            - $\\lambda_2$ (Fiedler value) starts near 0 when clusters are separated
            - As clusters merge, $\\lambda_2$ increases
            - Eigenvectors transition from localized to global modes
            """
        )
        st.page_link(
            "pages/1_merging_clusters.py",
            label="Open Merging Clusters Demo",
            icon="🔵",
        )

    with col2:
        st.subheader("2. Evolving Graph")
        st.markdown(
            """
            Two ring graphs gradually connect via coupling edges.

            **What to watch:**
            - Disconnected rings have degenerate eigenvalues
            - Coupling creates eigenvalue crossings
            - New global modes emerge spanning both rings
            """
        )
        st.page_link(
            "pages/2_evolving_graph.py",
            label="Open Evolving Graph Demo",
            icon="🔴",
        )

    st.divider()

    # Key concepts
    st.header("Key Concepts")

    with st.expander("Eigenvalue Matching", expanded=False):
        st.markdown(
            """
            **Problem**: Eigenvalues can reorder between timesteps (crossings).

            **Solution**: Use Hungarian algorithm to find optimal matching
            based on eigenvalue proximity:

            $$\\min_P \\sum_{i,j} P_{ij} |\\lambda_i^{(t)} - \\lambda_j^{(t+1)}|$$

            The permutation $P$ tells us which mode at time $t$ corresponds
            to which mode at time $t+1$.
            """
        )

    with st.expander("Sign Conventions", expanded=False):
        st.markdown(
            """
            **Problem**: Eigenvectors are only defined up to sign.

            **Solution**: Canonical sign rule based on spectral statistics
            (not node coordinates):

            - **Max entry positive**: Sign of maximum absolute entry is positive
            - **Sum positive**: Total sum of entries is non-negative
            - **Moment-based**: Use third moment (skewness) to break symmetry

            These rules give consistent signs without requiring node-to-node
            correspondence between graphs.
            """
        )

    with st.expander("Temporal Stability", expanded=False):
        st.markdown(
            """
            **Key insight**: Stable modes across time represent signal;
            volatile modes represent noise.

            By tracking eigenvalue and eigenvector stability, we can:

            1. Identify the **Temporal Intrinsic Dimension (TID)**
            2. Separate geometric structure from sampling noise
            3. Learn smooth flows on the Stiefel manifold
            """
        )

    st.divider()

    # Footer
    st.caption(
        """
        Part of the Temporal Spectral Flow framework.
        Use the sidebar to navigate between examples.
        """
    )


if __name__ == "__main__":
    main()
