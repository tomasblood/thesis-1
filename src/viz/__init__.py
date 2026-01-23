"""
Visualization module for Temporal Spectral Flow.

This module provides D3.js-inspired visualizations for flow matching and
optimal transport research. Designed to be paper-ready while maintaining
the aesthetic quality of interactive web visualizations.

Usage:
    from viz import plot_flow_trajectories, plot_vector_field
    from viz.styles import apply_style, VizStyle

    # Apply consistent styling
    apply_style(VizStyle(mode="paper"))

    # Create visualizations
    fig = plot_flow_trajectories(model, x0, steps=50)
    fig.savefig("figures/trajectories.pdf")

Modules:
    styles: Color palettes, typography, and matplotlib configuration
    static: Static plotting functions for trajectories, vector fields, KDE
    app: Streamlit interactive dashboard
"""

from viz.styles import (
    ColorPalette,
    VizStyle,
    apply_style,
    create_figure,
    save_figure,
    get_cmap_flow,
    get_time_colors,
)

from viz.static import (
    plot_flow_trajectories,
    plot_vector_field,
    plot_density_evolution,
    plot_comparison_panel,
    plot_transport_plan,
)

__all__ = [
    # Styles
    "ColorPalette",
    "VizStyle",
    "apply_style",
    "create_figure",
    "save_figure",
    "get_cmap_flow",
    "get_time_colors",
    # Static plots
    "plot_flow_trajectories",
    "plot_vector_field",
    "plot_density_evolution",
    "plot_comparison_panel",
    "plot_transport_plan",
]

__version__ = "0.1.0"
