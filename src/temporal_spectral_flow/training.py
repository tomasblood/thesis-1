"""
Training pipeline for the joint spectral flow model.

This module implements the training loop that:
1. Uses raw consecutive spectral pairs WITHOUT alignment
2. Trains via endpoint prediction with Grassmann-invariant loss
3. Integrates the learned vector field from t to t+1

Key principle: NO alignment, canonicalization, or interpolation during training.
The model learns to predict dynamics on raw spectral data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from einops import rearrange

from temporal_spectral_flow.joint_flow import JointSpectralFlow
from temporal_spectral_flow.spectral import SpectralSnapshot, TemporalSpectralEmbedding
from temporal_spectral_flow.losses import (
    principal_angle_loss,
    eigenvalue_mse_loss,
    velocity_energy_regularization,
)


@dataclass
class TrainingConfig:
    """Configuration for joint flow training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    n_epochs: int = 100
    grad_clip: float = 1.0

    # Loss weights
    alpha: float = 1.0  # Weight for eigenvalue MSE loss
    eta: float = 0.0  # Weight for energy regularization (default off)
    gamma: float = 1.0  # Weight for eigenvector velocity in energy regularization

    # Integration
    n_integration_steps: int = 10

    # Dataset options
    allow_size_mismatch: bool = False  # If True, raise error on size mismatch; if False, filter out

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingState:
    """Tracks training progress."""

    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    losses: List[float] = field(default_factory=list)
    grassmann_losses: List[float] = field(default_factory=list)
    eigenvalue_losses: List[float] = field(default_factory=list)
    energy_losses: List[float] = field(default_factory=list)


class TemporalPairDataset(Dataset):
    """
    Dataset of consecutive spectral snapshot pairs.

    Returns RAW pairs (Phi_t, lambda_t, Phi_next, lambda_next, t_index)
    WITHOUT any alignment, canonicalization, or sign fixing.

    Size mismatches are filtered out by default.
    """

    def __init__(
        self,
        snapshots: List[SpectralSnapshot],
        allow_size_mismatch: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            snapshots: List of spectral snapshots (each has Phi and eigenvalues)
            allow_size_mismatch: If True, raise error on size mismatch; if False, filter out
        """
        self.snapshots = snapshots
        self.pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for t in range(len(snapshots) - 1):
            Phi_t = snapshots[t].Phi
            Phi_next = snapshots[t + 1].Phi
            lambda_t = snapshots[t].eigenvalues
            lambda_next = snapshots[t + 1].eigenvalues

            # Check size compatibility
            if Phi_t.shape[0] != Phi_next.shape[0]:
                if allow_size_mismatch:
                    raise ValueError(
                        f"Size mismatch at t={t}: N_t={Phi_t.shape[0]}, N_{{t+1}}={Phi_next.shape[0]}. "
                        "Set allow_size_mismatch=False to filter out mismatched pairs."
                    )
                else:
                    logger.warning(
                        f"Skipping pair at t={t} due to size mismatch: "
                        f"N_t={Phi_t.shape[0]}, N_{{t+1}}={Phi_next.shape[0]}"
                    )
                    continue

            # Check spectral dimension compatibility
            if Phi_t.shape[1] != Phi_next.shape[1]:
                logger.warning(
                    f"Skipping pair at t={t} due to k mismatch: "
                    f"k_t={Phi_t.shape[1]}, k_{{t+1}}={Phi_next.shape[1]}"
                )
                continue

            # Store raw pairs without alignment
            self.pairs.append((
                Phi_t.astype(np.float32),
                lambda_t.astype(np.float32),
                Phi_next.astype(np.float32),
                lambda_next.astype(np.float32),
                np.array([float(t)], dtype=np.float32),
            ))

        logger.info(f"Created dataset with {len(self.pairs)} valid pairs from {len(snapshots)} snapshots")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Phi_t, lambda_t, Phi_next, lambda_next, t_idx = self.pairs[idx]
        return (
            torch.from_numpy(Phi_t),
            torch.from_numpy(lambda_t),
            torch.from_numpy(Phi_next),
            torch.from_numpy(lambda_next),
            torch.from_numpy(t_idx),
        )


class FlowTrainer:
    """
    Trainer for the joint spectral flow model.

    Uses endpoint prediction with Grassmann-invariant loss for eigenvectors
    and Euclidean MSE for eigenvalues. No alignment during training.
    """

    def __init__(
        self,
        model: JointSpectralFlow,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: JointSpectralFlow model to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.state = TrainingState()

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

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        """
        Single training step using endpoint prediction.

        Args:
            batch: (Phi_t, lambda_t, Phi_next, lambda_next, t_idx)

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        Phi_t, lambda_t, Phi_next, lambda_next, t_idx = batch
        Phi_t = Phi_t.to(self.device)
        lambda_t = lambda_t.to(self.device)
        Phi_next = Phi_next.to(self.device)
        lambda_next = lambda_next.to(self.device)
        t_idx = rearrange(t_idx, 'b 1 -> b').to(self.device)

        batch_size = Phi_t.shape[0]

        # Integrate from t to t+1
        lambda_pred, Phi_pred = self.model.integrate(
            lambda_t,
            Phi_t,
            t_start=0.0,  # Normalized time: 0 -> 1 for one step
            t_end=1.0,
            n_steps=self.config.n_integration_steps,
        )

        # Grassmann-invariant loss for eigenvectors (principal angles)
        grassmann_loss = principal_angle_loss(Phi_pred, Phi_next)

        # Euclidean MSE for eigenvalues
        eigenvalue_loss = eigenvalue_mse_loss(lambda_pred, lambda_next)

        # Optional energy regularization
        if self.config.eta > 0:
            # Compute velocity at start time for regularization
            t_start = torch.zeros(batch_size, device=self.device)
            v_lambda, v_Phi = self.model.velocity(lambda_t, Phi_t, t_start)
            energy_loss = velocity_energy_regularization(
                v_lambda, v_Phi, gamma=self.config.gamma
            )
        else:
            energy_loss = torch.tensor(0.0, device=self.device)

        # Total loss: L = L_G + alpha * L_lambda + eta * L_energy
        total_loss = (
            grassmann_loss +
            self.config.alpha * eigenvalue_loss +
            self.config.eta * energy_loss
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
            "grassmann_loss": grassmann_loss.item(),
            "eigenvalue_loss": eigenvalue_loss.item(),
            "energy_loss": energy_loss.item() if isinstance(energy_loss, torch.Tensor) else energy_loss,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average losses for the epoch
        """
        total_losses = {
            "total_loss": 0.0,
            "grassmann_loss": 0.0,
            "eigenvalue_loss": 0.0,
            "energy_loss": 0.0,
        }
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses = self.train_step(batch)

            for key in total_losses:
                total_losses[key] += losses[key]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total_loss']:.4f}",
                "G": f"{losses['grassmann_loss']:.4f}",
                "λ": f"{losses['eigenvalue_loss']:.4f}",
            })

            self.state.step += 1

        # Average losses
        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        return avg_losses

    def train(
        self,
        snapshots: List[SpectralSnapshot],
        val_snapshots: Optional[List[SpectralSnapshot]] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingState:
        """
        Full training loop.

        Args:
            snapshots: Training spectral snapshots
            val_snapshots: Optional validation snapshots
            callbacks: Optional callbacks called after each epoch

        Returns:
            Final training state
        """
        # Create dataset and dataloader (NO alignment)
        dataset = TemporalPairDataset(
            snapshots,
            allow_size_mismatch=self.config.allow_size_mismatch,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check that snapshots have consistent sizes.")

        # Validation data
        if val_snapshots is not None:
            val_dataset = TemporalPairDataset(
                val_snapshots,
                allow_size_mismatch=self.config.allow_size_mismatch,
            )
            val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        else:
            val_dataloader = None

        # Training loop
        for epoch in range(self.config.n_epochs):
            self.state.epoch = epoch

            # Train
            train_losses = self.train_epoch(dataloader, epoch)
            self.state.losses.append(train_losses["total_loss"])
            self.state.grassmann_losses.append(train_losses["grassmann_loss"])
            self.state.eigenvalue_losses.append(train_losses["eigenvalue_loss"])
            self.state.energy_losses.append(train_losses["energy_loss"])

            # Validation
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(
                    f"Epoch {epoch}: train_loss={train_losses['total_loss']:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: loss={train_losses['total_loss']:.4f}, "
                    f"G={train_losses['grassmann_loss']:.4f}, λ={train_losses['eigenvalue_loss']:.4f}"
                )

            # Update best loss
            if train_losses["total_loss"] < self.state.best_loss:
                self.state.best_loss = train_losses["total_loss"]

            # Learning rate scheduling
            self.scheduler.step()

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self.model, self.state, train_losses)

        return self.state

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> float:
        """
        Evaluate model on dataloader.

        Args:
            dataloader: Evaluation data

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            Phi_t, lambda_t, Phi_next, lambda_next, t_idx = batch
            Phi_t = Phi_t.to(self.device)
            lambda_t = lambda_t.to(self.device)
            Phi_next = Phi_next.to(self.device)
            lambda_next = lambda_next.to(self.device)

            # Integrate
            lambda_pred, Phi_pred = self.model.integrate(
                lambda_t,
                Phi_t,
                t_start=0.0,
                t_end=1.0,
                n_steps=self.config.n_integration_steps,
            )

            # Losses
            grassmann_loss = principal_angle_loss(Phi_pred, Phi_next)
            eigenvalue_loss = eigenvalue_mse_loss(lambda_pred, lambda_next)
            loss = grassmann_loss + self.config.alpha * eigenvalue_loss

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = checkpoint["training_state"]


def train_from_data(
    data_snapshots: List[NDArray[np.floating]],
    k: int = 10,
    config: Optional[TrainingConfig] = None,
    **embedding_kwargs,
) -> Tuple[JointSpectralFlow, FlowTrainer, List[SpectralSnapshot]]:
    """
    Convenience function to train a joint flow model from raw data.

    Args:
        data_snapshots: List of data matrices (N_t, d) for each time t
        k: Spectral dimension
        config: Training configuration
        **embedding_kwargs: Arguments for spectral embedding

    Returns:
        Trained model, trainer, and spectral snapshots
    """
    # Compute spectral embeddings
    embedder = TemporalSpectralEmbedding(k=k, **embedding_kwargs)
    snapshots = embedder.process_sequence(data_snapshots)

    # Create JointSpectralFlow model (NOT SpectralFlowModel)
    model = JointSpectralFlow(k=k)

    # Create trainer
    config = config or TrainingConfig()
    trainer = FlowTrainer(model, config)

    # Train
    trainer.train(snapshots)

    return model, trainer, snapshots
