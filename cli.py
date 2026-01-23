#!/usr/bin/env python3
"""
CLI for Temporal Spectral Flow.

Usage:
    python cli.py train --config config.yaml
    python cli.py visualize --data path/to/data
    python cli.py compute_tid --checkpoint path/to/checkpoint
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
import yaml
from loguru import logger

from temporal_spectral_flow.flow import SpectralFlowModel
from temporal_spectral_flow.spectral import TemporalSpectralEmbedding
from temporal_spectral_flow.training import FlowTrainer, TrainingConfig, train_from_data
from temporal_spectral_flow.tid import TIDComputer, TIDConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class CLI:
    """Temporal Spectral Flow CLI."""

    def train(
        self,
        data_path: str,
        config: str = "config.yaml",
        checkpoint_dir: str = "checkpoints",
        seed: int = 42,
    ) -> None:
        """
        Train a spectral flow model.

        Args:
            data_path: Path to training data (numpy .npy file with list of snapshots)
            config: Path to configuration YAML file
            checkpoint_dir: Directory to save checkpoints
            seed: Random seed for reproducibility
        """
        logger.info(f"Loading config from {config}")
        cfg = load_config(config)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data
        logger.info(f"Loading data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data_snapshots = list(data)
        else:
            data_snapshots = [data[i] for i in range(len(data))]

        # Create training config
        train_cfg = TrainingConfig(
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            batch_size=cfg["training"]["batch_size"],
            n_epochs=cfg["training"]["n_steps"] // 100,  # Approximate
            grad_clip=cfg["training"]["grad_clip"],
            velocity_loss_weight=cfg["training"]["velocity_loss_weight"],
            endpoint_loss_weight=cfg["training"]["endpoint_loss_weight"],
            regularization_weight=cfg["training"]["regularization_weight"],
            n_integration_steps=cfg["training"]["n_integration_steps"],
            alignment_method=cfg["alignment"]["method"],
            alignment_reg=cfg["alignment"]["reg"],
            alignment_reg_m=cfg["alignment"]["reg_m"],
            log_interval=cfg["training"]["log_interval"],
            checkpoint_interval=cfg["training"]["checkpoint_interval"],
        )

        # Train
        logger.info("Starting training")
        model, trainer, snapshots = train_from_data(
            data_snapshots,
            k=cfg["model"]["k"],
            config=train_cfg,
        )

        # Save checkpoint
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def compute_tid(
        self,
        data_path: str,
        checkpoint: str,
        config: str = "config.yaml",
        output: Optional[str] = None,
    ) -> None:
        """
        Compute Temporal Intrinsic Dimension.

        Args:
            data_path: Path to data
            checkpoint: Path to model checkpoint
            config: Path to configuration YAML file
            output: Optional path to save TID results
        """
        logger.info(f"Loading config from {config}")
        cfg = load_config(config)

        # Load data
        logger.info(f"Loading data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data_snapshots = list(data)
        else:
            data_snapshots = [data[i] for i in range(len(data))]

        # Create spectral embeddings
        embedder = TemporalSpectralEmbedding(k=cfg["model"]["k"])
        snapshots = embedder.process_sequence(data_snapshots)

        # Load model
        logger.info(f"Loading model from {checkpoint}")
        model = SpectralFlowModel(k=cfg["model"]["k"])
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        # Compute TID
        tid_cfg = TIDConfig(
            method=cfg["tid"]["method"],
            energy_threshold=cfg["tid"]["energy_threshold"],
            min_stability=cfg["tid"]["min_stability"],
        )
        tid_computer = TIDComputer(model=model, config=tid_cfg)
        result = tid_computer.compute(snapshots)

        logger.info(f"TID: {result.tid}")
        logger.info(f"Effective TID: {result.effective_tid:.2f}")
        logger.info(f"Dimension scores: {result.dimension_scores}")

        if output:
            output_path = Path(output)
            np.savez(
                output_path,
                tid=result.tid,
                effective_tid=result.effective_tid,
                dimension_scores=result.dimension_scores,
                stability_scores=result.stability_scores,
            )
            logger.info(f"Saved results to {output_path}")

    def info(self) -> None:
        """Print information about the package."""
        logger.info("Temporal Spectral Flow")
        logger.info("=" * 40)
        logger.info("A framework for learning temporal dynamics on spectral embeddings")
        logger.info("")
        logger.info("Commands:")
        logger.info("  train       - Train a spectral flow model")
        logger.info("  compute_tid - Compute Temporal Intrinsic Dimension")
        logger.info("  info        - Print this information")


def main():
    """Main entry point."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
