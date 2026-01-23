"""
D3.js-inspired styling for Flow Matching / Optimal Transport visualizations.

This module provides consistent color palettes and styling that emulates
the aesthetic quality of Diffusion Explorer while being paper-ready.

Key Design Principles:
- Minimalist: Avoid clutter, let the data speak
- Colorblind-friendly: Tested with Sim Daltonism
- Paper-ready: Works in both color and grayscale
- LaTeX-compatible: Serif fonts for static, sans-serif for interactive
"""

from dataclasses import dataclass, field
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for flow visualizations.

    Inspired by Diffusion Explorer with modifications for paper readability.
    """

    # Primary colors (for source/target distributions)
    source: str = "#3b82f6"      # Cool Blue (x_0, source distribution)
    target: str = "#ef4444"      # Warm Red (x_1, target distribution)

    # Secondary colors (for trajectories and vector fields)
    trajectory: str = "#f17720"  # Diffusion Explorer orange
    trajectory_alt: str = "#6b7280"  # Gray alternative for multiple trajectories
    vector_field: str = "#8b5cf6"    # Distinct purple for vectors

    # Neutral colors
    grid: str = "#e5e7eb"        # Light gray for grid
    text: str = "#1f2937"        # Dark gray for text
    background_light: str = "#ffffff"  # White (paper)
    background_dark: str = "#1e293b"   # Dark slate (screen)

    # Semantic colors
    error: str = "#dc2626"       # Red for errors
    success: str = "#16a34a"     # Green for ground truth
    highlight: str = "#fbbf24"   # Yellow for highlights

    # Alphas
    scatter_alpha: float = 0.6
    trajectory_alpha: float = 0.3
    contour_alpha: float = 0.5
    vector_alpha: float = 0.8


@dataclass
class VizStyle:
    """Visualization style configuration.

    Separates paper (static) vs screen (interactive) styling.
    """

    mode: Literal["paper", "screen"] = "paper"
    palette: ColorPalette = field(default_factory=ColorPalette)

    # Figure dimensions
    fig_width: float = 8.0
    fig_height: float = 6.0
    dpi: int = 150

    # Line widths
    trajectory_lw: float = 1.5
    vector_lw: float = 1.0
    axis_lw: float = 0.8

    # Marker sizes
    scatter_size: float = 30
    arrow_scale: float = 30

    # Font sizes
    title_size: float = 14
    label_size: float = 12
    tick_size: float = 10
    legend_size: float = 10

    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.3

    def __post_init__(self) -> None:
        """Adjust settings based on mode."""
        if self.mode == "screen":
            self.fig_width = 10.0
            self.fig_height = 8.0
            self.title_size = 16
            self.label_size = 14

    @property
    def background(self) -> str:
        """Get background color based on mode."""
        if self.mode == "paper":
            return self.palette.background_light
        return self.palette.background_dark

    @property
    def text_color(self) -> str:
        """Get text color based on mode."""
        if self.mode == "paper":
            return self.palette.text
        return "#f8fafc"  # Light text for dark mode


def apply_style(style: VizStyle | None = None) -> None:
    """Apply visualization style to matplotlib.

    Configures matplotlib rcParams for consistent styling.
    Call this at the start of your visualization script.

    Args:
        style: VizStyle configuration. Uses paper mode by default.
    """
    if style is None:
        style = VizStyle()

    # Use seaborn-whitegrid as base
    plt.style.use("seaborn-v0_8-whitegrid" if hasattr(plt.style, "use") else "default")

    # Font configuration
    font_family = "serif" if style.mode == "paper" else "sans-serif"

    rc_params = {
        # Figure
        "figure.figsize": (style.fig_width, style.fig_height),
        "figure.dpi": style.dpi,
        "figure.facecolor": style.background,
        "figure.edgecolor": style.background,

        # Axes
        "axes.facecolor": style.background,
        "axes.edgecolor": style.palette.grid,
        "axes.linewidth": style.axis_lw,
        "axes.labelsize": style.label_size,
        "axes.titlesize": style.title_size,
        "axes.labelcolor": style.text_color,
        "axes.grid": style.show_grid,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "grid.color": style.palette.grid,
        "grid.alpha": style.grid_alpha,
        "grid.linewidth": 0.5,

        # Ticks
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "xtick.color": style.text_color,
        "ytick.color": style.text_color,

        # Lines
        "lines.linewidth": style.trajectory_lw,

        # Legend
        "legend.fontsize": style.legend_size,
        "legend.framealpha": 0.8,
        "legend.facecolor": style.background,
        "legend.edgecolor": style.palette.grid,

        # Font
        "font.family": font_family,
        "font.size": style.label_size,

        # Text
        "text.color": style.text_color,

        # LaTeX
        "text.usetex": style.mode == "paper" and _latex_available(),
        "mathtext.fontset": "cm",
    }

    mpl.rcParams.update(rc_params)


def _latex_available() -> bool:
    """Check if LaTeX is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_cmap_flow() -> mpl.colors.LinearSegmentedColormap:
    """Get colormap for flow magnitude visualization.

    Returns a colormap from cool (slow) to warm (fast).
    """
    palette = ColorPalette()
    colors = [palette.source, "#a855f7", palette.target]  # Blue -> Purple -> Red
    return mpl.colors.LinearSegmentedColormap.from_list("flow", colors)


def get_time_colors(n_steps: int) -> NDArray[np.floating]:
    """Get colors for a time sequence from t=0 to t=1.

    Args:
        n_steps: Number of time steps.

    Returns:
        Array of RGB colors shape (n_steps, 3).
    """
    cmap = get_cmap_flow()
    return cmap(np.linspace(0, 1, n_steps))[:, :3]


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    style: VizStyle | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes | NDArray]:
    """Create a figure with consistent styling.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        style: VizStyle configuration.
        **kwargs: Additional arguments to plt.subplots.

    Returns:
        Tuple of (figure, axes).
    """
    if style is None:
        style = VizStyle()

    apply_style(style)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(style.fig_width * ncols * 0.8, style.fig_height * nrows * 0.8),
        **kwargs
    )

    fig.patch.set_facecolor(style.background)

    return fig, axes


def add_time_annotation(
    ax: plt.Axes,
    t: float,
    style: VizStyle | None = None,
) -> None:
    """Add time annotation to plot.

    Args:
        ax: Matplotlib axes.
        t: Time value in [0, 1].
        style: VizStyle configuration.
    """
    if style is None:
        style = VizStyle()

    ax.text(
        0.95, 0.95,
        f"$t = {t:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=style.label_size,
        color=style.text_color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=style.background,
            edgecolor=style.palette.grid,
            alpha=0.8,
        ),
    )


def format_axis_labels(
    ax: plt.Axes,
    xlabel: str = "$x_1$",
    ylabel: str = "$x_2$",
    title: str | None = None,
    style: VizStyle | None = None,
) -> None:
    """Format axis labels and title consistently.

    Args:
        ax: Matplotlib axes.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional title.
        style: VizStyle configuration.
    """
    if style is None:
        style = VizStyle()

    ax.set_xlabel(xlabel, fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)

    if title:
        ax.set_title(title, fontsize=style.title_size, pad=10)


def save_figure(
    fig: plt.Figure,
    path: str,
    style: VizStyle | None = None,
    tight: bool = True,
) -> None:
    """Save figure with consistent settings.

    Args:
        fig: Matplotlib figure.
        path: Output path (should end in .pdf or .png).
        style: VizStyle configuration.
        tight: Use tight_layout.
    """
    if style is None:
        style = VizStyle()

    if tight:
        fig.tight_layout()

    fig.savefig(
        path,
        dpi=style.dpi * 2 if path.endswith(".png") else style.dpi,
        facecolor=style.background,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
    )
