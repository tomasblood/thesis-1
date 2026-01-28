"""
TS-GFM Demo: Temporal Spectral Geodesic Flow Matching

Demonstrates the complete flow matching pipeline:
1. Generate toy spectral trajectories (merging clusters)
2. Train a GeodesicFlowModel to learn the dynamics
3. Interpolate between frames
4. Visualize results

Usage:
    python examples/ts_gfm_demo.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from temporal_spectral_flow.riemannian_flow import GeodesicFlowModel, uniform_stiefel
from temporal_spectral_flow.riemannian_training import (
    RiemannianTrainingConfig,
    RiemannianFlowTrainer,
)
from temporal_spectral_flow.inference import interpolate, get_trajectory
from temporal_spectral_flow.spectral import SpectralSnapshot
from temporal_spectral_flow.grassmann import grassmann_distance, projection_frobenius_efficient


def generate_toy_trajectory(
    n_points: int = 50,
    k: int = 4,
    n_frames: int = 20,
    seed: int = 42,
) -> list:
    """
    Generate a simple toy trajectory of spectral frames.

    Creates frames that smoothly transition eigenvectors on Stiefel manifold.
    """
    rng = np.random.default_rng(seed)

    # Start and end eigenvector bases
    Phi_start, _ = np.linalg.qr(rng.standard_normal((n_points, k)))
    Phi_end, _ = np.linalg.qr(rng.standard_normal((n_points, k)))

    # Eigenvalues: smooth transition
    lambda_start = np.sort(rng.uniform(0.5, 2.0, k))[::-1]
    lambda_end = np.sort(rng.uniform(0.5, 2.0, k))[::-1]

    snapshots = []

    for i in range(n_frames):
        t = i / (n_frames - 1)

        # Linear interpolation of eigenvalues
        eigenvalues = (1 - t) * lambda_start + t * lambda_end

        # Geodesic-ish interpolation of eigenvectors (using naive SVD approach)
        # Note: For actual geodesic, we'd use stiefel_geodesic_qr
        U, s, Vt = np.linalg.svd(Phi_start.T @ Phi_end, full_matrices=False)
        R_optimal = U @ Vt

        # Interpolated rotation (approximate)
        # For simplicity, blend and re-orthonormalize
        Phi_interp = (1 - t) * Phi_start + t * (Phi_start @ R_optimal)
        Phi, _ = np.linalg.qr(Phi_interp)

        snap = SpectralSnapshot(
            Phi=Phi,
            eigenvalues=eigenvalues,
            n_samples=n_points,
            k=k,
        )
        snapshots.append(snap)

    return snapshots


def train_model(
    snapshots: list,
    n_epochs: int = 50,
    device: str = "cpu",
) -> tuple:
    """Train a GeodesicFlowModel on the snapshots."""
    n, k = snapshots[0].Phi.shape

    # Create model
    model = GeodesicFlowModel(
        n=n,
        k=k,
        hidden_dim_phi=256,
        hidden_dim_lambda=128,
        n_blocks_phi=2,
        n_blocks_lambda=2,
    )

    # Create trainer
    config = RiemannianTrainingConfig(
        n_epochs=n_epochs,
        batch_size=8,
        learning_rate=1e-3,
        device=device,
        log_interval=10,
    )
    trainer = RiemannianFlowTrainer(model, config)

    # Train
    print(f"Training GeodesicFlowModel for {n_epochs} epochs...")
    state = trainer.train(snapshots)
    print(f"Training complete. Final loss: {state.losses[-1]:.4f}")

    return model, trainer, state


def evaluate_interpolation(
    model: GeodesicFlowModel,
    snapshots: list,
    device: torch.device,
) -> dict:
    """Evaluate interpolation quality."""
    grassmann_errors = []
    frobenius_errors = []
    lambda_errors = []

    # Test on triplets (A, B, C) - interpolate A→C, check at B
    for i in range(len(snapshots) - 2):
        Phi_A = snapshots[i].Phi
        lambda_A = snapshots[i].eigenvalues
        Phi_B = snapshots[i + 1].Phi
        lambda_B = snapshots[i + 1].eigenvalues
        Phi_C = snapshots[i + 2].Phi
        lambda_C = snapshots[i + 2].eigenvalues

        # Interpolate from A to C at s=0.5
        Phi_pred, lambda_pred = interpolate(
            model,
            Phi_A, lambda_A,
            Phi_C, lambda_C,
            s_target=0.5,
            n_steps=30,
            device=device,
        )

        grassmann_errors.append(grassmann_distance(Phi_pred, Phi_B))
        frobenius_errors.append(projection_frobenius_efficient(Phi_pred, Phi_B))
        lambda_errors.append(np.linalg.norm(lambda_pred - lambda_B))

    return {
        "grassmann_mean": np.mean(grassmann_errors),
        "grassmann_std": np.std(grassmann_errors),
        "frobenius_mean": np.mean(frobenius_errors),
        "frobenius_std": np.std(frobenius_errors),
        "lambda_mean": np.mean(lambda_errors),
        "lambda_std": np.std(lambda_errors),
    }


def visualize_results(
    model: GeodesicFlowModel,
    snapshots: list,
    training_state,
    device: torch.device,
    save_path: Path = None,
):
    """Create visualization of training and interpolation results."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Training loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(training_state.losses, 'b-', label='Total', linewidth=2)
    ax1.plot(training_state.phi_losses, 'r--', label='Φ', alpha=0.7)
    ax1.plot(training_state.lambda_losses, 'g--', label='λ', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Eigenvalue trajectories (ground truth)
    ax2 = fig.add_subplot(gs[0, 1])
    n_frames = len(snapshots)
    k = snapshots[0].eigenvalues.shape[0]
    t_values = np.linspace(0, 1, n_frames)

    for j in range(k):
        eigs_j = [snap.eigenvalues[j] for snap in snapshots]
        ax2.plot(t_values, eigs_j, '-', label=f'λ_{j+1}', linewidth=2)

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Ground Truth Eigenvalues')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Interpolation: generate trajectory and compare
    ax3 = fig.add_subplot(gs[0, 2])

    # Get trajectory from model
    Phi_A = snapshots[0].Phi
    lambda_A = snapshots[0].eigenvalues
    Phi_B = snapshots[-1].Phi
    lambda_B = snapshots[-1].eigenvalues

    Phi_traj, lambda_traj = get_trajectory(
        model,
        Phi_A, lambda_A,
        Phi_B, lambda_B,
        n_points=20,
        n_steps_per_segment=5,
        device=device,
    )

    # Plot interpolated eigenvalues
    t_interp = np.linspace(0, 1, len(lambda_traj))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for j in range(k):
        interp_eigs_j = [lam[j] for lam in lambda_traj]
        ax3.plot(t_interp, interp_eigs_j, 'o-', color=colors[j],
                 label=f'λ_{j+1} (model)', markersize=4, alpha=0.8)
        # Ground truth for comparison
        gt_eigs_j = [snap.eigenvalues[j] for snap in snapshots]
        ax3.plot(t_values, gt_eigs_j, '--', color=colors[j], alpha=0.4)

    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Model Interpolation vs Ground Truth')
    ax3.grid(True, alpha=0.3)

    # 4. First two eigenvector components at start
    ax4 = fig.add_subplot(gs[1, 0])
    Phi_0 = snapshots[0].Phi
    ax4.scatter(Phi_0[:, 0], Phi_0[:, 1], c='blue', alpha=0.6, s=20)
    ax4.set_xlabel('φ_1')
    ax4.set_ylabel('φ_2')
    ax4.set_title('Eigenvectors at t=0')
    ax4.set_aspect('equal')

    # 5. Eigenvector components at midpoint (ground truth vs model)
    ax5 = fig.add_subplot(gs[1, 1])
    mid_idx = len(snapshots) // 2
    Phi_mid_gt = snapshots[mid_idx].Phi

    # Get model prediction at midpoint
    Phi_mid_pred, _ = interpolate(
        model,
        Phi_A, lambda_A,
        Phi_B, lambda_B,
        s_target=0.5,
        n_steps=30,
        device=device,
    )

    ax5.scatter(Phi_mid_gt[:, 0], Phi_mid_gt[:, 1], c='blue', alpha=0.6, s=20, label='GT')
    ax5.scatter(Phi_mid_pred[:, 0], Phi_mid_pred[:, 1], c='red', alpha=0.4, s=20, label='Model')
    ax5.set_xlabel('φ_1')
    ax5.set_ylabel('φ_2')
    ax5.set_title('Eigenvectors at t=0.5')
    ax5.legend()
    ax5.set_aspect('equal')

    # 6. Eigenvector components at end
    ax6 = fig.add_subplot(gs[1, 2])
    Phi_1 = snapshots[-1].Phi
    ax6.scatter(Phi_1[:, 0], Phi_1[:, 1], c='blue', alpha=0.6, s=20)
    ax6.set_xlabel('φ_1')
    ax6.set_ylabel('φ_2')
    ax6.set_title('Eigenvectors at t=1')
    ax6.set_aspect('equal')

    plt.suptitle('TS-GFM: Temporal Spectral Geodesic Flow Matching', fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Run the full TS-GFM demo."""
    print("=" * 60)
    print("TS-GFM Demo: Temporal Spectral Geodesic Flow Matching")
    print("=" * 60)

    # Parameters
    n_points = 50
    k = 4
    n_frames = 30
    n_epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Points: {n_points}, Spectral dims: {k}, Frames: {n_frames}")

    # 1. Generate data
    print("\n[1] Generating toy trajectory...")
    snapshots = generate_toy_trajectory(n_points, k, n_frames)
    print(f"    Generated {len(snapshots)} spectral frames")

    # 2. Train model
    print("\n[2] Training GeodesicFlowModel...")
    model, trainer, state = train_model(snapshots, n_epochs, device)

    # 3. Evaluate
    print("\n[3] Evaluating interpolation quality...")
    metrics = evaluate_interpolation(model, snapshots, torch.device(device))
    print(f"    Grassmann distance: {metrics['grassmann_mean']:.4f} ± {metrics['grassmann_std']:.4f}")
    print(f"    Projection Frobenius: {metrics['frobenius_mean']:.4f} ± {metrics['frobenius_std']:.4f}")
    print(f"    Eigenvalue error: {metrics['lambda_mean']:.4f} ± {metrics['lambda_std']:.4f}")

    # 4. Visualize
    print("\n[4] Creating visualization...")
    output_path = Path(__file__).parent / "ts_gfm_demo_results.png"
    visualize_results(model, snapshots, state, torch.device(device), output_path)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
