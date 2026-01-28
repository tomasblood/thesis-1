"""
Reusable panel components for spectral alignment visualization.

Each panel is a self-contained function that takes spectral data
and returns a matplotlib figure for display in Streamlit.
"""

from viz.components.eigenvalue_panel import plot_eigenvalue_trajectories
from viz.components.eigenvector_panel import plot_eigenvector_heatmap
from viz.components.embedding_panel import plot_spectral_embedding

__all__ = [
    "plot_eigenvalue_trajectories",
    "plot_eigenvector_heatmap",
    "plot_spectral_embedding",
]
