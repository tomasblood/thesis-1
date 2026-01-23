"""
Tests for the visualization module.

Verifies that all plotting functions execute without error and
produce valid matplotlib figures.

Run: python tests/test_viz.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from viz.styles import (
    ColorPalette,
    VizStyle,
    apply_style,
    create_figure,
    get_cmap_flow,
    get_time_colors,
)
from viz.static import (
    plot_flow_trajectories,
    plot_vector_field,
    plot_density_evolution,
    plot_comparison_panel,
    plot_transport_plan,
    plot_euler_comparison,
    integrate_ode,
    sample_from_gaussian_mixture,
    create_toy_velocity_field,
)


# =============================================================================
# Test Utilities
# =============================================================================


def linear_velocity(x: np.ndarray, t: float) -> np.ndarray:
    """Simple linear velocity field for testing."""
    target = np.array([1.5, 0.0])
    source = np.array([-1.5, 0.0])
    return np.tile(target - source, (len(x), 1))


def rotation_velocity(x: np.ndarray, t: float) -> np.ndarray:
    """Rotating velocity field for testing."""
    omega = np.pi
    vx = -omega * x[:, 1]
    vy = omega * x[:, 0]
    return np.stack([vx, vy], axis=1)


# =============================================================================
# Style Tests
# =============================================================================


def test_color_palette():
    """Test ColorPalette creation."""
    palette = ColorPalette()
    assert palette.source == "#3b82f6"
    assert palette.target == "#ef4444"
    assert 0 <= palette.scatter_alpha <= 1
    print("✓ ColorPalette creation")


def test_viz_style():
    """Test VizStyle creation and mode switching."""
    style_paper = VizStyle(mode="paper")
    assert style_paper.background == "#ffffff"

    style_screen = VizStyle(mode="screen")
    assert style_screen.background == "#1e293b"
    assert style_screen.fig_width > style_paper.fig_width
    print("✓ VizStyle modes")


def test_apply_style():
    """Test matplotlib style application."""
    apply_style(VizStyle(mode="paper"))
    # Just verify no errors
    print("✓ apply_style")


def test_create_figure():
    """Test figure creation."""
    fig, ax = create_figure(1, 1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig, axes = create_figure(2, 2)
    assert axes.shape == (2, 2)
    plt.close(fig)
    print("✓ create_figure")


def test_get_cmap_flow():
    """Test flow colormap."""
    cmap = get_cmap_flow()
    colors = cmap(np.linspace(0, 1, 10))
    assert colors.shape == (10, 4)  # RGBA
    print("✓ get_cmap_flow")


def test_get_time_colors():
    """Test time-based colors."""
    colors = get_time_colors(5)
    assert colors.shape == (5, 3)  # RGB
    print("✓ get_time_colors")


# =============================================================================
# Integration Tests
# =============================================================================


def test_integrate_ode():
    """Test ODE integration."""
    x0 = np.array([[0.0, 0.0], [1.0, 0.0]])

    # Euler
    traj = integrate_ode(linear_velocity, x0, n_steps=10, method="euler")
    assert len(traj) == 11  # n_steps + 1
    assert traj[0].shape == x0.shape

    # Midpoint
    traj = integrate_ode(linear_velocity, x0, n_steps=10, method="midpoint")
    assert len(traj) == 11

    # RK4
    traj = integrate_ode(linear_velocity, x0, n_steps=10, method="rk4")
    assert len(traj) == 11
    print("✓ integrate_ode (all methods)")


def test_sample_from_gaussian_mixture():
    """Test Gaussian mixture sampling."""
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    samples = sample_from_gaussian_mixture(100, centers, stds=0.3, seed=42)
    assert samples.shape == (100, 2)
    print("✓ sample_from_gaussian_mixture")


# =============================================================================
# Plotting Tests
# =============================================================================


def test_plot_flow_trajectories():
    """Test flow trajectory plotting."""
    x0 = sample_from_gaussian_mixture(50, np.array([[-1.5, 0.0]]), stds=0.3, seed=42)

    fig = plot_flow_trajectories(
        linear_velocity, x0,
        steps=20,
        title="Test Trajectories",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # With color_by_time=False
    fig = plot_flow_trajectories(
        linear_velocity, x0,
        steps=20,
        color_by_time=False,
    )
    plt.close(fig)
    print("✓ plot_flow_trajectories")


def test_plot_vector_field():
    """Test vector field plotting."""
    # Quiver
    fig = plot_vector_field(
        linear_velocity, t=0.5,
        grid_size=10,
        method="quiver",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Streamplot
    fig = plot_vector_field(
        linear_velocity, t=0.5,
        grid_size=10,
        method="streamplot",
    )
    plt.close(fig)

    # Without color by magnitude
    fig = plot_vector_field(
        linear_velocity, t=0.5,
        color_by_magnitude=False,
    )
    plt.close(fig)
    print("✓ plot_vector_field")


def test_plot_density_evolution():
    """Test density evolution plotting."""
    x0 = sample_from_gaussian_mixture(100, np.array([[-1.5, 0.0]]), stds=0.3, seed=42)

    fig = plot_density_evolution(
        linear_velocity, x0,
        t_steps=[0.0, 0.5, 1.0],
        n_integration_steps=50,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✓ plot_density_evolution")


def test_plot_comparison_panel():
    """Test comparison panel plotting."""
    x0 = sample_from_gaussian_mixture(30, np.array([[-1.5, 0.0]]), stds=0.3, seed=42)

    fig = plot_comparison_panel(
        linear_velocity,
        rotation_velocity,
        x0,
        steps=20,
        title_left="Linear",
        title_right="Rotation",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✓ plot_comparison_panel")


def test_plot_transport_plan():
    """Test transport plan plotting."""
    x_source = sample_from_gaussian_mixture(20, np.array([[-1.5, 0.0]]), stds=0.3, seed=42)
    x_target = sample_from_gaussian_mixture(20, np.array([[1.5, 0.0]]), stds=0.3, seed=43)

    # Simple transport plan (identity-like)
    T = np.eye(20) / 20

    fig = plot_transport_plan(x_source, x_target, T)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✓ plot_transport_plan")


def test_plot_euler_comparison():
    """Test Euler step comparison plotting."""
    x0 = sample_from_gaussian_mixture(20, np.array([[-1.5, 0.0]]), stds=0.3, seed=42)

    fig = plot_euler_comparison(
        linear_velocity, x0,
        step_counts=[1, 4, 16],
        reference_steps=64,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✓ plot_euler_comparison")


def test_create_toy_velocity_field():
    """Test toy velocity field creation."""
    for mode in ["linear", "rotation", "swiss_roll", "two_moons"]:
        vf = create_toy_velocity_field(mode)
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        v = vf(x, 0.5)
        assert v.shape == x.shape
    print("✓ create_toy_velocity_field (all modes)")


# =============================================================================
# Output Generation Test
# =============================================================================


def generate_sample_figures():
    """Generate sample figures to figures/test_runs/."""
    output_dir = Path(__file__).parent.parent / "figures" / "test_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating sample figures...")

    # Setup
    x0 = sample_from_gaussian_mixture(100, np.array([[-1.5, 0.0]]), stds=0.4, seed=42)
    style = VizStyle(mode="paper")

    # 1. Flow trajectories
    fig = plot_flow_trajectories(
        linear_velocity, x0,
        steps=50,
        style=style,
        title="Gaussian → Gaussian Flow",
    )
    fig.savefig(output_dir / "flow_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'flow_trajectories.png'}")

    # 2. Vector field
    fig = plot_vector_field(
        rotation_velocity, t=0.5,
        grid_size=20,
        method="streamplot",
        style=style,
        title="Rotation Vector Field",
    )
    fig.savefig(output_dir / "vector_field_t_0.5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'vector_field_t_0.5.png'}")

    # 3. Density evolution
    fig = plot_density_evolution(
        linear_velocity, x0,
        t_steps=[0.0, 0.25, 0.5, 0.75, 1.0],
        style=style,
    )
    fig.savefig(output_dir / "density_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'density_evolution.png'}")

    # 4. Method comparison
    fig = plot_comparison_panel(
        linear_velocity,
        rotation_velocity,
        x0[:50],
        steps=30,
        style=style,
        title_left="Linear Flow (OT)",
        title_right="Rotation Flow",
    )
    fig.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'method_comparison.png'}")

    # 5. Euler comparison
    fig = plot_euler_comparison(
        linear_velocity, x0[:30],
        step_counts=[1, 4, 16, 64],
        style=style,
    )
    fig.savefig(output_dir / "euler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'euler_comparison.png'}")

    print(f"\nAll figures saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running visualization tests...")
    print("=" * 60)

    # Style tests
    test_color_palette()
    test_viz_style()
    test_apply_style()
    test_create_figure()
    test_get_cmap_flow()
    test_get_time_colors()

    # Integration tests
    test_integrate_ode()
    test_sample_from_gaussian_mixture()

    # Plotting tests
    test_plot_flow_trajectories()
    test_plot_vector_field()
    test_plot_density_evolution()
    test_plot_comparison_panel()
    test_plot_transport_plan()
    test_plot_euler_comparison()
    test_create_toy_velocity_field()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Generate sample outputs
    generate_sample_figures()


if __name__ == "__main__":
    run_all_tests()
