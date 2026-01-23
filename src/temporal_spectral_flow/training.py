"""
Training pipeline for the spectral flow model.

This module implements the training loop that:
1. Uses the inner alignment oracle to generate transport-consistent targets
2. Trains the outer flow model to match these targets
3. Ensures the outer flow learns true dynamics, not just alignment

Key principle: The inner alignment provides supervision, but the outer
flow must learn to predict without alignment at inference time.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from einops import rearrange, einsum
from beartype import beartype
from jaxtyping import Float, jaxtyped

from temporal_spectral_flow.flow import SpectralFlowModel, numpy_to_torch, torch_to_numpy
from temporal_spectral_flow.spectral import SpectralSnapshot, TemporalSpectralEmbedding
from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.transport import TransportAlignment, AlignmentResult, BasisAligner


@dataclass
class TrainingConfig:
    """Configuration for flow training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    n_epochs: int = 100
    grad_clip: float = 1.0

    # Loss weights
    velocity_loss_weight: float = 1.0
    endpoint_loss_weight: float = 0.1
    regularization_weight: float = 0.01

    # Integration
    n_integration_steps: int = 10

    # Alignment
    alignment_method: str = "unbalanced"
    alignment_reg: float = 0.1
    alignment_reg_m: float = 1.0

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
    losses: list[float] = field(default_factory=list)
    velocity_losses: list[float] = field(default_factory=list)
    endpoint_losses: list[float] = field(default_factory=list)


class TemporalPairDataset(Dataset):
    """
    Dataset of consecutive spectral snapshot pairs.

    Each item is (Phi_t, Phi_{t+1}_aligned, t) where the target
    has been pre-aligned using the transport oracle.
    """

    def __init__(
        self,
        snapshots: list[SpectralSnapshot],
        aligner: TransportAlignment,
        include_eigenvalues: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            snapshots: List of spectral snapshots
            aligner: Transport alignment module
            include_eigenvalues: Use eigenvalues in alignment cost
        """
        self.snapshots = snapshots
        self.aligner = aligner
        self.include_eigenvalues = include_eigenvalues

        # Pre-compute alignments for all consecutive pairs
        self.pairs = []
        self.alignment_results = []

        basis_aligner = BasisAligner()

        for t in range(len(snapshots) - 1):
            Phi_t = snapshots[t].Phi
            Phi_next = snapshots[t + 1].Phi

            # Handle size mismatch with transport
            if Phi_t.shape[0] != Phi_next.shape[0]:
                # Full transport alignment
                eig_t = snapshots[t].eigenvalues if include_eigenvalues else None
                eig_next = snapshots[t + 1].eigenvalues if include_eigenvalues else None

                result = aligner.align(
                    Phi_t, Phi_next,
                    eigenvalues_source=eig_t,
                    eigenvalues_target=eig_next,
                )
                Phi_next_aligned = result.aligned_target
                self.alignment_results.append(result)
            else:
                # Same size: just align bases
                _, Phi_next_aligned = basis_aligner.align_bases(Phi_t, Phi_next)
                self.alignment_results.append(None)

            self.pairs.append((
                Phi_t.astype(np.float32),
                Phi_next_aligned.astype(np.float32),
                np.array([t], dtype=np.float32),
            ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Phi_t, Phi_next, t = self.pairs[idx]
        return (
            torch.from_numpy(Phi_t),
            torch.from_numpy(Phi_next),
            torch.from_numpy(t),
        )


class FlowTrainer:
    """
    Trainer for the spectral flow model.

    Manages the training loop, loss computation, and checkpointing.
    """

    def __init__(
        self,
        model: SpectralFlowModel,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: SpectralFlowModel to train
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

        self.aligner = TransportAlignment(
            method=self.config.alignment_method,
            reg=self.config.alignment_reg,
            reg_m=self.config.alignment_reg_m,
        )

    def compute_velocity_loss(
        self,
        Phi_t: torch.Tensor,
        Phi_next_aligned: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target velocities.

        Target velocity is the geodesic/log-map displacement from
        Phi_t to Phi_{t+1}^{aligned}.

        Args:
            Phi_t: Current spectral embedding (batch, N, k)
            Phi_next_aligned: Aligned next embedding (batch, N, k)
            t: Time values (batch,)

        Returns:
            Velocity matching loss
        """
        # Predict velocity
        v_pred = self.model.predict_velocity(Phi_t, t)

        # Target velocity: simple difference (projected to tangent space)
        # For small timesteps, this approximates the log map
        v_target = Phi_next_aligned - Phi_t

        # Project target to tangent space at Phi_t
        v_target = self._project_to_tangent_batch(Phi_t, v_target)

        # MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)

        return loss

    def compute_endpoint_loss(
        self,
        Phi_t: torch.Tensor,
        Phi_next_aligned: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss on integrated endpoint.

        Args:
            Phi_t: Current embedding
            Phi_next_aligned: Target aligned embedding
            t: Time

        Returns:
            Endpoint matching loss
        """
        # Integrate from t to t+1
        t_end = t + 1.0

        Phi_pred = self.model(
            Phi_t, t, t_end,
            n_steps=self.config.n_integration_steps,
        )

        # Geodesic distance (approximated by Frobenius norm)
        loss = torch.mean((Phi_pred - Phi_next_aligned) ** 2)

        return loss

    def compute_regularization_loss(
        self,
        Phi_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Regularization on velocity field smoothness.

        Penalizes high velocity magnitudes for stability.
        """
        v = self.model.predict_velocity(Phi_t, t)
        return torch.mean(v ** 2)

    @jaxtyped(typechecker=beartype)
    def _project_to_tangent_batch(
        self,
        Phi: Float[torch.Tensor, "batch n k"],
        V: Float[torch.Tensor, "batch n k"],
    ) -> Float[torch.Tensor, "batch n k"]:
        """
        Project V to tangent space at Phi (batched).

        Args:
            Phi: Points on Stiefel (batch, N, k)
            V: Vectors to project (batch, N, k)

        Returns:
            Projected tangent vectors
        """
        # V_tan = V - Phi @ sym(Phi^T @ V)
        PhiTV = einsum(Phi, V, 'b n k, b n l -> b k l')  # (batch, k, k)
        sym_PhiTV = (PhiTV + rearrange(PhiTV, 'b i j -> b j i')) / 2
        return V - einsum(Phi, sym_PhiTV, 'b n k, b k l -> b n l')

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        """
        Single training step.

        Args:
            batch: (Phi_t, Phi_next_aligned, t)

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        Phi_t, Phi_next_aligned, t = batch
        Phi_t = Phi_t.to(self.device)
        Phi_next_aligned = Phi_next_aligned.to(self.device)
        t = rearrange(t, 'b 1 -> b').to(self.device)

        # Compute losses
        velocity_loss = self.compute_velocity_loss(Phi_t, Phi_next_aligned, t)
        endpoint_loss = self.compute_endpoint_loss(Phi_t, Phi_next_aligned, t)
        reg_loss = self.compute_regularization_loss(Phi_t, t)

        # Total loss
        total_loss = (
            self.config.velocity_loss_weight * velocity_loss +
            self.config.endpoint_loss_weight * endpoint_loss +
            self.config.regularization_weight * reg_loss
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
            "velocity_loss": velocity_loss.item(),
            "endpoint_loss": endpoint_loss.item(),
            "reg_loss": reg_loss.item(),
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
        total_losses = {"total_loss": 0, "velocity_loss": 0, "endpoint_loss": 0, "reg_loss": 0}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses = self.train_step(batch)

            for key in total_losses:
                total_losses[key] += losses[key]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total_loss']:.4f}",
                "v_loss": f"{losses['velocity_loss']:.4f}",
            })

            self.state.step += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def train(
        self,
        snapshots: list[SpectralSnapshot],
        val_snapshots: Optional[list[SpectralSnapshot]] = None,
        callbacks: Optional[list[Callable]] = None,
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
        # Create dataset and dataloader
        dataset = TemporalPairDataset(snapshots, self.aligner)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Validation data
        if val_snapshots is not None:
            val_dataset = TemporalPairDataset(val_snapshots, self.aligner)
            val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        else:
            val_dataloader = None

        # Training loop
        for epoch in range(self.config.n_epochs):
            self.state.epoch = epoch

            # Train
            train_losses = self.train_epoch(dataloader, epoch)
            self.state.losses.append(train_losses["total_loss"])
            self.state.velocity_losses.append(train_losses["velocity_loss"])
            self.state.endpoint_losses.append(train_losses["endpoint_loss"])

            # Validation
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(
                    f"Epoch {epoch}: train_loss={train_losses['total_loss']:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_losses['total_loss']:.4f}")

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
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                Phi_t, Phi_next_aligned, t = batch
                Phi_t = Phi_t.to(self.device)
                Phi_next_aligned = Phi_next_aligned.to(self.device)
                t = rearrange(t, 'b 1 -> b').to(self.device)

                velocity_loss = self.compute_velocity_loss(Phi_t, Phi_next_aligned, t)
                total_loss += velocity_loss.item()
                n_batches += 1

        return total_loss / n_batches

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
    data_snapshots: list[NDArray[np.floating]],
    k: int = 10,
    config: Optional[TrainingConfig] = None,
    **embedding_kwargs,
) -> Tuple[SpectralFlowModel, FlowTrainer, list[SpectralSnapshot]]:
    """
    Convenience function to train a flow model from raw data.

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

    # Create model
    model = SpectralFlowModel(k=k)

    # Create trainer
    trainer = FlowTrainer(model, config)

    # Train
    trainer.train(snapshots)

    return model, trainer, snapshots
