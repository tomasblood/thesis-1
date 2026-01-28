"""
Riemannian Flow Matching training for Temporal Spectral Geodesic Flow.

Implements the training loop:
1. Sample noise start: Φ_0 ~ Uniform(St), λ_0 ~ N(0, σ²I)
2. Sample time: s ~ Uniform(0, 1)
3. Compute geodesic bridge: Φ_s = geodesic(Φ_0, Φ_1, s)
4. Compute target velocity: v* = Log_{Φ_s}(Φ_1) / (1 - s + ε)
5. Match predicted velocity to target

Core principle: Let manifold geometry define the reference flow.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from einops import rearrange

from temporal_spectral_flow.riemannian_flow import (
    GeodesicFlowModel,
    uniform_stiefel,
    stiefel_log_qr,
    stiefel_geodesic_qr,
)
from temporal_spectral_flow.grassmann import (
    grassmann_loss,
    projection_frobenius_torch,
)
from temporal_spectral_flow.spectral import SpectralSnapshot


@dataclass
class RiemannianTrainingConfig:
    """Configuration for Riemannian Flow Matching training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    n_epochs: int = 200
    grad_clip: float = 1.0

    # Loss weights
    phi_loss_weight: float = 1.0
    lambda_loss_weight: float = 1.0
    grassmann_loss_weight: float = 0.1  # Optional endpoint consistency

    # Flow matching
    noise_sigma_lambda: float = 1.0  # Std for eigenvalue noise
    eps: float = 1e-4  # Numerical stability for 1/(1-s)

    # Evaluation
    eval_n_steps: int = 50  # Integration steps for endpoint loss

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 50

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RiemannianTrainingState:
    """Tracks training progress."""

    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    losses: List[float] = field(default_factory=list)
    phi_losses: List[float] = field(default_factory=list)
    lambda_losses: List[float] = field(default_factory=list)


class SpectralFrameDataset(Dataset):
    """
    Dataset of spectral frames for flow matching.

    Each item is (Φ, λ) representing a target frame.
    Training samples noise and creates the bridge on-the-fly.
    """

    def __init__(
        self,
        snapshots: List[SpectralSnapshot],
    ):
        """
        Initialize dataset.

        Args:
            snapshots: List of spectral snapshots (targets)
        """
        self.frames = []
        for snap in snapshots:
            self.frames.append((
                snap.Phi.astype(np.float32),
                snap.eigenvalues.astype(np.float32),
            ))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi, lam = self.frames[idx]
        return torch.from_numpy(Phi), torch.from_numpy(lam)


class RiemannianFlowTrainer:
    """
    Trainer for Riemannian Flow Matching.

    Implements the TS-GFM training algorithm:
    1. For each target frame (Φ_1, λ_1):
       - Sample noise: Φ_0 ~ Uniform(St), λ_0 ~ N(0, σ²I)
       - Sample time: s ~ Uniform(0, 1)
       - Compute bridge: Φ_s = geodesic(Φ_0, Φ_1, s), λ_s = lerp
       - Compute target velocity: intrinsic geodesic direction
       - Train to match velocities
    """

    def __init__(
        self,
        model: GeodesicFlowModel,
        config: Optional[RiemannianTrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: GeodesicFlowModel to train
            config: Training configuration
        """
        self.model = model
        self.config = config or RiemannianTrainingConfig()
        self.state = RiemannianTrainingState()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_epochs,
        )

    def sample_noise_start(
        self,
        batch_size: int,
        n: int,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noise starting point for flow matching.

        Φ_0 ~ Uniform(St(n, k))
        λ_0 ~ N(0, σ²I)

        Args:
            batch_size: Number of samples
            n: Ambient dimension
            k: Spectral dimension

        Returns:
            (Phi_0, lambda_0)
        """
        Phi_0 = uniform_stiefel(
            n, k, batch_size,
            device=self.device,
            dtype=torch.float32,
        )

        lambda_0 = self.config.noise_sigma_lambda * torch.randn(
            batch_size, k,
            device=self.device,
            dtype=torch.float32,
        )

        return Phi_0, lambda_0

    def compute_target_velocity(
        self,
        Phi_s: torch.Tensor,
        Phi_1: torch.Tensor,
        lambda_0: torch.Tensor,
        lambda_1: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target velocities for flow matching.

        Φ velocity: v_Φ* = Log_{Φ_s}(Φ_1) / (1 - s + ε)
        λ velocity: v_λ* = λ_1 - λ_0

        Args:
            Phi_s: Interpolated eigenvectors at time s
            Phi_1: Target eigenvectors
            lambda_0: Noise eigenvalues
            lambda_1: Target eigenvalues
            s: Current time (batch,)

        Returns:
            (v_Phi_target, v_lambda_target)
        """
        # Eigenvector target velocity: intrinsic direction toward target
        # v* = Log_{Φ_s}(Φ_1) / (1 - s + ε)
        V = stiefel_log_qr(Phi_s, Phi_1)

        # Scale by 1 / (1 - s + ε)
        # s has shape (batch,), need to broadcast
        scale = 1.0 / (1.0 - s + self.config.eps)
        scale = rearrange(scale, 'b -> b 1 1')
        v_Phi_target = V * scale

        # Eigenvalue target velocity: constant direction (Euclidean)
        v_lambda_target = lambda_1 - lambda_0

        return v_Phi_target, v_lambda_target

    def train_step(
        self,
        Phi_1: torch.Tensor,
        lambda_1: torch.Tensor,
    ) -> dict:
        """
        Single training step (flow matching on one batch of targets).

        Args:
            Phi_1: Target eigenvectors (batch, n, k)
            lambda_1: Target eigenvalues (batch, k)

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_size, n, k = Phi_1.shape

        # 1. Sample noise start
        Phi_0, lambda_0 = self.sample_noise_start(batch_size, n, k)

        # 2. Sample time s ~ Uniform(0, 1)
        s = torch.rand(batch_size, device=self.device)

        # 3. Compute geodesic bridge
        # Φ_s = geodesic(Φ_0, Φ_1, s)
        Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s.view(-1, 1, 1))

        # λ_s = (1 - s) λ_0 + s λ_1
        s_expand = s.unsqueeze(-1)
        lambda_s = (1 - s_expand) * lambda_0 + s_expand * lambda_1

        # 4. Compute target velocities
        v_Phi_target, v_lambda_target = self.compute_target_velocity(
            Phi_s, Phi_1, lambda_0, lambda_1, s
        )

        # 5. Predict velocities
        v_Phi_pred, v_lambda_pred = self.model.velocity(
            Phi_s, lambda_s, Phi_1, lambda_1, s
        )

        # 6. Compute losses
        # Eigenvector velocity loss (Frobenius MSE)
        L_Phi = torch.mean((v_Phi_pred - v_Phi_target) ** 2)

        # Eigenvalue velocity loss (MSE)
        L_lambda = torch.mean((v_lambda_pred - v_lambda_target) ** 2)

        # Total loss
        total_loss = (
            self.config.phi_loss_weight * L_Phi +
            self.config.lambda_loss_weight * L_lambda
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "phi_loss": L_Phi.item(),
            "lambda_loss": L_lambda.item(),
        }

    def train_step_with_endpoint(
        self,
        Phi_1: torch.Tensor,
        lambda_1: torch.Tensor,
    ) -> dict:
        """
        Training step with optional Grassmann endpoint consistency loss.

        Adds: L_Grass = ||Φ̂_1 Φ̂_1^T - Φ_1 Φ_1^T||_F^2
        after integrating from noise to t=1.
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_size, n, k = Phi_1.shape

        # Sample noise start
        Phi_0, lambda_0 = self.sample_noise_start(batch_size, n, k)

        # Sample time
        s = torch.rand(batch_size, device=self.device)

        # Geodesic bridge
        Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s.view(-1, 1, 1))
        s_expand = s.unsqueeze(-1)
        lambda_s = (1 - s_expand) * lambda_0 + s_expand * lambda_1

        # Target velocities
        v_Phi_target, v_lambda_target = self.compute_target_velocity(
            Phi_s, Phi_1, lambda_0, lambda_1, s
        )

        # Predicted velocities
        v_Phi_pred, v_lambda_pred = self.model.velocity(
            Phi_s, lambda_s, Phi_1, lambda_1, s
        )

        # Velocity losses
        L_Phi = torch.mean((v_Phi_pred - v_Phi_target) ** 2)
        L_lambda = torch.mean((v_lambda_pred - v_lambda_target) ** 2)

        # Optional: Grassmann endpoint loss
        # Integrate from noise to target and check subspace consistency
        if self.config.grassmann_loss_weight > 0:
            with torch.no_grad():
                # Detach for endpoint evaluation
                Phi_pred, lambda_pred = self.model.integrate(
                    Phi_0.detach(), lambda_0.detach(),
                    Phi_1, lambda_1,
                    t_end=1.0,
                    n_steps=self.config.eval_n_steps,
                )

            # Compute Grassmann loss (on detached prediction for stability)
            L_Grass = grassmann_loss(Phi_pred, Phi_1)
        else:
            L_Grass = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            self.config.phi_loss_weight * L_Phi +
            self.config.lambda_loss_weight * L_lambda +
            self.config.grassmann_loss_weight * L_Grass
        )

        total_loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "phi_loss": L_Phi.item(),
            "lambda_loss": L_lambda.item(),
            "grassmann_loss": L_Grass.item() if isinstance(L_Grass, torch.Tensor) else L_Grass,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        use_endpoint_loss: bool = False,
    ) -> dict:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader of target frames
            epoch: Current epoch number
            use_endpoint_loss: Include Grassmann endpoint loss

        Returns:
            Average losses for the epoch
        """
        total_losses = {"total_loss": 0, "phi_loss": 0, "lambda_loss": 0}
        if use_endpoint_loss:
            total_losses["grassmann_loss"] = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for Phi_1, lambda_1 in pbar:
            Phi_1 = Phi_1.to(self.device)
            lambda_1 = lambda_1.to(self.device)

            if use_endpoint_loss:
                losses = self.train_step_with_endpoint(Phi_1, lambda_1)
            else:
                losses = self.train_step(Phi_1, lambda_1)

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total_loss']:.4f}",
                "Φ": f"{losses['phi_loss']:.4f}",
                "λ": f"{losses['lambda_loss']:.4f}",
            })

            self.state.step += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def train(
        self,
        snapshots: List[SpectralSnapshot],
        use_endpoint_loss: bool = False,
        callbacks: Optional[List[Callable]] = None,
    ) -> RiemannianTrainingState:
        """
        Full training loop.

        Args:
            snapshots: Training spectral frames (targets)
            use_endpoint_loss: Include Grassmann endpoint consistency
            callbacks: Optional callbacks after each epoch

        Returns:
            Final training state
        """
        dataset = SpectralFrameDataset(snapshots)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        for epoch in range(self.config.n_epochs):
            self.state.epoch = epoch

            train_losses = self.train_epoch(dataloader, epoch, use_endpoint_loss)
            self.state.losses.append(train_losses["total_loss"])
            self.state.phi_losses.append(train_losses["phi_loss"])
            self.state.lambda_losses.append(train_losses["lambda_loss"])

            if epoch % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: loss={train_losses['total_loss']:.4f}, "
                    f"Φ={train_losses['phi_loss']:.4f}, "
                    f"λ={train_losses['lambda_loss']:.4f}"
                )

            if train_losses["total_loss"] < self.state.best_loss:
                self.state.best_loss = train_losses["total_loss"]

            self.scheduler.step()

            if callbacks:
                for callback in callbacks:
                    callback(self.model, self.state, train_losses)

        return self.state

    def evaluate_grassmann(
        self,
        snapshots: List[SpectralSnapshot],
        n_steps: int = 50,
    ) -> float:
        """
        Evaluate model using Grassmann distance.

        For each frame, integrate from noise and measure subspace error.

        Args:
            snapshots: Evaluation frames
            n_steps: Integration steps

        Returns:
            Average Grassmann distance
        """
        self.model.eval()
        total_distance = 0
        n_frames = len(snapshots)

        with torch.no_grad():
            for snap in snapshots:
                Phi_1 = torch.from_numpy(snap.Phi.astype(np.float32)).to(self.device)
                lambda_1 = torch.from_numpy(snap.eigenvalues.astype(np.float32)).to(self.device)

                n, k = Phi_1.shape

                # Sample noise
                Phi_0 = uniform_stiefel(n, k, 1, self.device).squeeze(0)
                lambda_0 = self.config.noise_sigma_lambda * torch.randn(k, device=self.device)

                # Integrate
                Phi_pred, _ = self.model.integrate(
                    Phi_0, lambda_0, Phi_1, lambda_1,
                    t_end=1.0, n_steps=n_steps,
                )

                # Grassmann distance
                d = projection_frobenius_torch(
                    Phi_pred.unsqueeze(0), Phi_1.unsqueeze(0)
                ).sqrt().item()
                total_distance += d

        return total_distance / n_frames

    def save_checkpoint(self, path: Path) -> None:
        """Save model and training state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": self.state,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = checkpoint["training_state"]


# =============================================================================
# Convenience function
# =============================================================================


def train_geodesic_flow(
    snapshots: List[SpectralSnapshot],
    n_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    hidden_dim_phi: int = 1024,
    hidden_dim_lambda: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[GeodesicFlowModel, RiemannianFlowTrainer]:
    """
    Train a geodesic flow model on spectral frames.

    Args:
        snapshots: List of SpectralSnapshot targets
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim_phi: Hidden dim for eigenvector network
        hidden_dim_lambda: Hidden dim for eigenvalue network
        device: Compute device

    Returns:
        (trained_model, trainer)
    """
    # Get dimensions from first snapshot
    n, k = snapshots[0].Phi.shape

    # Create model
    model = GeodesicFlowModel(
        n=n,
        k=k,
        hidden_dim_phi=hidden_dim_phi,
        hidden_dim_lambda=hidden_dim_lambda,
    )

    # Create config
    config = RiemannianTrainingConfig(
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # Create trainer
    trainer = RiemannianFlowTrainer(model, config)

    # Train
    trainer.train(snapshots)

    return model, trainer
