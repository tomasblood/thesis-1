"""
Streamlit app for Temporal Geodesic Flow Matching visualization.

Interactive demonstration of how spectral embeddings (Phi, lambda)
evolve over time using the new temporal geodesic flow matching paradigm:
- Training on consecutive pairs (Phi_t, lambda_t) -> (Phi_{t+1}, lambda_{t+1})
- Grassmann loss for subspace comparison
- No sampling from noise - we learn actual temporal dynamics

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
    """Main entry point for Temporal Geodesic Flow Matching app."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: uv add streamlit")
        return

    st.set_page_config(
        page_title="Temporal Geodesic Flow Matching",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("Temporal Geodesic Flow Matching")
    st.markdown(
        """
        Interactive visualization of spectral dynamics on the Stiefel manifold.
        **No sampling from noise** — we learn the actual temporal evolution.
        """
    )

    st.divider()

    # Overview section
    st.header("The New Paradigm")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What Changed")
        st.markdown(
            """
            **Old approach** (standard flow matching):
            - Sample from noise distribution
            - Learn to denoise toward target
            - Reference geodesics between random pairs

            **New approach** (temporal geodesic):
            - Train on consecutive spectral pairs
            - Learn actual temporal dynamics
            - No noise, no reference geodesics
            """
        )

    with col2:
        st.subheader("Training Step")
        st.code(
            """
train_step(model, Phi_t, lambda_t,
           Phi_{t+1}, lambda_{t+1}, optimizer)

1. Integrate: (Phi_t, lambda_t) -> (Phi_hat, lambda_hat)
2. Grassmann loss: d(span(Phi_hat), span(Phi_{t+1}))
3. Eigenvalue loss: ||lambda_hat - lambda_{t+1}||^2
4. Energy regularization: ||velocity||^2
5. Backprop
            """,
            language="python",
        )

    st.divider()

    # Key concepts
    st.header("Key Concepts")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Stiefel Manifold")
        st.markdown(
            r"""
            **St(n, k)** = orthonormal k-frames in $\mathbb{R}^n$

            $$\text{St}(n, k) = \{\Phi \in \mathbb{R}^{n \times k} : \Phi^T \Phi = I_k\}$$

            Spectral embeddings $\Phi_t$ live on this manifold.
            Velocities must be **tangent vectors**.
            """
        )

    with col2:
        st.subheader("Grassmann Distance")
        st.markdown(
            r"""
            **Gr(n, k)** = k-dimensional subspaces of $\mathbb{R}^n$

            Distance based on **principal angles**:
            $$d_{\text{Gr}}(\mathcal{U}, \mathcal{V}) = \|\sin(\theta_1, ..., \theta_k)\|$$

            Invariant to basis choice within subspace.
            """
        )

    with col3:
        st.subheader("Why This Works")
        st.markdown(
            """
            - **Geometry-aware**: Respects manifold structure
            - **Gauge-invariant**: Grassmann loss ignores sign/rotation
            - **Temporal**: Learns actual dynamics, not denoising
            - **Extrapolation**: Can predict future states
            """
        )

    st.divider()

    # Interactive examples
    st.header("Interactive Examples")

    st.markdown(
        """
        Explore two toy examples that demonstrate temporal geodesic flow matching:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Merging Clusters")
        st.markdown(
            """
            Two Gaussian clusters gradually merge into one.

            **What to watch:**
            - $\\lambda_2$ (Fiedler value) increases as clusters merge
            - Model learns to predict subspace evolution
            - Grassmann loss measures prediction quality
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
            - Eigenvalue crossings as structure changes
            - Model tracks modes through crossings
            - One-step integration accuracy
            """
        )
        st.page_link(
            "pages/2_evolving_graph.py",
            label="Open Evolving Graph Demo",
            icon="🔴",
        )

    st.divider()

    # Mathematical details
    st.header("Mathematical Details")

    with st.expander("Loss Functions", expanded=False):
        st.markdown(
            r"""
            ### Grassmann Loss (Subspace Distance)

            Given predicted $\hat{\Phi}_{t+1}$ and true $\Phi_{t+1}$, compute
            principal angles via SVD:

            $$\hat{\Phi}_{t+1}^T \Phi_{t+1} = U \Sigma V^T$$

            The diagonal of $\Sigma$ contains $\cos(\theta_i)$ for principal angles.
            Grassmann distance:

            $$\mathcal{L}_{\text{Gr}} = \|\sin(\theta_1, ..., \theta_k)\|_2$$

            ### Eigenvalue Loss

            Simple MSE on eigenvalues:

            $$\mathcal{L}_\lambda = \frac{1}{k} \sum_{i=1}^k (\hat{\lambda}_i - \lambda_i)^2$$

            ### Energy Regularization

            Penalize large velocities for stability:

            $$\mathcal{L}_{\text{reg}} = \|\dot{\Phi}\|_F^2 + \|\dot{\lambda}\|^2$$
            """
        )

    with st.expander("Stiefel Geometry", expanded=False):
        st.markdown(
            r"""
            ### Tangent Space

            At $\Phi \in \text{St}(n, k)$, the tangent space is:

            $$T_\Phi \text{St}(n, k) = \{V : \Phi^T V + V^T \Phi = 0\}$$

            To project arbitrary $V$ to tangent space:

            $$V_{\text{tan}} = V - \Phi \cdot \text{sym}(\Phi^T V)$$

            where $\text{sym}(A) = (A + A^T)/2$.

            ### Retraction

            QR retraction maps tangent vector back to manifold:

            $$R_\Phi(V) = \text{qf}(\Phi + V)$$

            where $\text{qf}$ extracts the Q factor from QR decomposition.

            ### Integration

            One step of flow integration:

            $$\Phi_{t+1} = R_{\Phi_t}(v_\theta(\Phi_t, t) \cdot \Delta t)$$
            """
        )

    with st.expander("Why Not Standard Flow Matching?", expanded=False):
        st.markdown(
            r"""
            ### Standard Flow Matching

            In standard flow matching (e.g., for images):
            1. Sample source $x_0 \sim p_0$ (e.g., noise)
            2. Sample target $x_1 \sim p_1$ (e.g., data)
            3. Construct reference path $x_t = (1-t)x_0 + t x_1$
            4. Train velocity field to match $\dot{x}_t$

            ### Why This Doesn't Work Here

            For temporal spectral data:
            - We have **ordered sequences** $(\\Phi_0, \\Phi_1, ..., \\Phi_T)$
            - Consecutive pairs are **already coupled** by temporal dynamics
            - We want to learn the **actual evolution**, not interpolation
            - Sampling random pairs destroys temporal structure

            ### Our Approach

            Instead of noise → target, we learn:
            - **Consecutive pair dynamics**: $(\\Phi_t, \\lambda_t) \\to (\\Phi_{t+1}, \\lambda_{t+1})$
            - Model predicts one-step evolution
            - Loss compares predicted vs true next state
            - No artificial interpolation needed
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
