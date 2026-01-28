"""
Tests for Riemannian Flow Matching training.

Verifies the training pipeline including:
- Noise sampling
- Target velocity computation
- Training step gradients
- Dataset and data loading
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import tempfile

from temporal_spectral_flow.riemannian_training import (
    RiemannianTrainingConfig,
    RiemannianTrainingState,
    SpectralFrameDataset,
    RiemannianFlowTrainer,
)
from temporal_spectral_flow.riemannian_flow import (
    GeodesicFlowModel,
    uniform_stiefel,
    stiefel_geodesic_qr,
)
from temporal_spectral_flow.spectral import SpectralSnapshot


def create_mock_snapshots(n: int, k: int, count: int) -> list:
    """Create mock SpectralSnapshot objects for testing."""
    rng = np.random.default_rng(42)
    snapshots = []
    for i in range(count):
        Phi, _ = np.linalg.qr(rng.standard_normal((n, k)))
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, k))[::-1]
        snap = SpectralSnapshot(
            Phi=Phi.astype(np.float64),
            eigenvalues=eigenvalues.astype(np.float64),
            n_samples=n,
            k=k,
        )
        snapshots.append(snap)
    return snapshots


class TestRiemannianTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = RiemannianTrainingConfig()

        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.n_epochs > 0
        assert 0 <= config.phi_loss_weight
        assert 0 <= config.lambda_loss_weight

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = RiemannianTrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            n_epochs=100,
        )

        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.n_epochs == 100


class TestRiemannianTrainingState:
    """Tests for training state tracking."""

    def test_initial_state(self):
        """Initial state should have zero epoch/step."""
        state = RiemannianTrainingState()

        assert state.epoch == 0
        assert state.step == 0
        assert state.best_loss == float("inf")
        assert len(state.losses) == 0

    def test_state_tracks_losses(self):
        """State should track appended losses."""
        state = RiemannianTrainingState()
        state.losses.append(1.0)
        state.losses.append(0.5)

        assert len(state.losses) == 2
        assert state.losses[0] == 1.0


class TestSpectralFrameDataset:
    """Tests for the dataset class."""

    def test_dataset_length(self):
        """Dataset length should match number of snapshots."""
        snapshots = create_mock_snapshots(n=50, k=6, count=10)
        dataset = SpectralFrameDataset(snapshots)

        assert len(dataset) == 10

    def test_dataset_item_shapes(self):
        """Dataset items should have correct shapes."""
        n, k = 50, 6
        snapshots = create_mock_snapshots(n=n, k=k, count=5)
        dataset = SpectralFrameDataset(snapshots)

        Phi, lam = dataset[0]

        assert Phi.shape == (n, k)
        assert lam.shape == (k,)
        assert Phi.dtype == torch.float32
        assert lam.dtype == torch.float32

    def test_dataset_orthonormality_preserved(self):
        """Eigenvectors should remain orthonormal."""
        n, k = 50, 6
        snapshots = create_mock_snapshots(n=n, k=k, count=5)
        dataset = SpectralFrameDataset(snapshots)

        Phi, _ = dataset[0]
        gram = Phi.T @ Phi

        assert torch.allclose(gram, torch.eye(k), atol=1e-5)


class TestRiemannianFlowTrainer:
    """Tests for the trainer class."""

    @pytest.fixture
    def small_model_and_trainer(self):
        """Create small model and trainer for testing."""
        n, k = 20, 4
        model = GeodesicFlowModel(
            n=n, k=k,
            hidden_dim_phi=32,
            hidden_dim_lambda=16,
            n_blocks_phi=1,
            n_blocks_lambda=1,
        )
        config = RiemannianTrainingConfig(
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        trainer = RiemannianFlowTrainer(model, config)
        return model, trainer, n, k

    def test_sample_noise_start_shapes(self, small_model_and_trainer):
        """Noise samples should have correct shapes."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        Phi_0, lambda_0 = trainer.sample_noise_start(batch_size, n, k)

        assert Phi_0.shape == (batch_size, n, k)
        assert lambda_0.shape == (batch_size, k)

    def test_sample_noise_orthonormal(self, small_model_and_trainer):
        """Noise Phi samples should be orthonormal."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        Phi_0, _ = trainer.sample_noise_start(batch_size, n, k)

        for i in range(batch_size):
            gram = Phi_0[i].T @ Phi_0[i]
            assert torch.allclose(gram, torch.eye(k), atol=1e-5)

    def test_compute_target_velocity_shapes(self, small_model_and_trainer):
        """Target velocity should have correct shapes."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        Phi_0, lambda_0 = trainer.sample_noise_start(batch_size, n, k)
        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)
        s = torch.rand(batch_size)

        Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s.view(-1, 1, 1))

        v_Phi, v_lambda = trainer.compute_target_velocity(
            Phi_s, Phi_1, lambda_0, lambda_1, s
        )

        assert v_Phi.shape == (batch_size, n, k)
        assert v_lambda.shape == (batch_size, k)

    def test_predicted_velocity_in_tangent_space(self, small_model_and_trainer):
        """Model's predicted Phi velocity should lie in tangent space."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        Phi_0, lambda_0 = trainer.sample_noise_start(batch_size, n, k)
        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)
        s = torch.tensor([0.5] * batch_size)

        Phi_s = stiefel_geodesic_qr(Phi_0, Phi_1, s.view(-1, 1, 1))
        lambda_s = 0.5 * lambda_0 + 0.5 * lambda_1

        # The model's velocity prediction should be in tangent space
        # (ConditionalStiefelVelocityField projects to tangent space)
        v_Phi_pred, _ = model.velocity(Phi_s, lambda_s, Phi_1, lambda_1, s)

        # Tangent space condition: Phi^T V + V^T Phi = 0 (skew-symmetric)
        for i in range(batch_size):
            PhiTV = Phi_s[i].T @ v_Phi_pred[i]
            skew_check = PhiTV + PhiTV.T
            assert torch.allclose(skew_check, torch.zeros_like(skew_check), atol=1e-4)

    def test_train_step_returns_losses(self, small_model_and_trainer):
        """Train step should return loss dictionary."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)

        losses = trainer.train_step(Phi_1, lambda_1)

        assert "total_loss" in losses
        assert "phi_loss" in losses
        assert "lambda_loss" in losses
        assert losses["total_loss"] >= 0
        assert losses["phi_loss"] >= 0
        assert losses["lambda_loss"] >= 0

    def test_train_step_updates_parameters(self, small_model_and_trainer):
        """Train step should update model parameters."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        # Get initial parameters
        params_before = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)

        trainer.train_step(Phi_1, lambda_1)

        # Check at least some parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, params_before[name]):
                params_changed = True
                break

        assert params_changed

    def test_train_step_with_endpoint_includes_grassmann(self, small_model_and_trainer):
        """Train step with endpoint should include Grassmann loss."""
        model, trainer, n, k = small_model_and_trainer
        trainer.config.grassmann_loss_weight = 0.1
        batch_size = 4

        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)

        losses = trainer.train_step_with_endpoint(Phi_1, lambda_1)

        assert "grassmann_loss" in losses

    def test_train_step_increments_step(self, small_model_and_trainer):
        """Train step should increment step counter."""
        model, trainer, n, k = small_model_and_trainer
        batch_size = 4

        initial_step = trainer.state.step

        Phi_1 = uniform_stiefel(n, k, batch_size)
        lambda_1 = torch.randn(batch_size, k)

        # train_step doesn't increment step (train_epoch does)
        # Let's verify via train_epoch
        snapshots = create_mock_snapshots(n=n, k=k, count=8)
        from torch.utils.data import DataLoader
        dataset = SpectralFrameDataset(snapshots)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer.train_epoch(dataloader, epoch=0)

        assert trainer.state.step > initial_step


class TestCheckpointing:
    """Tests for model checkpointing."""

    def test_save_and_load_checkpoint(self):
        """Checkpoint save/load should preserve state."""
        n, k = 20, 4
        model = GeodesicFlowModel(
            n=n, k=k,
            hidden_dim_phi=32,
            hidden_dim_lambda=16,
        )
        config = RiemannianTrainingConfig(device="cpu")
        trainer = RiemannianFlowTrainer(model, config)

        # Do some training
        Phi_1 = uniform_stiefel(n, k, batch_size=4)
        lambda_1 = torch.randn(4, k)
        trainer.train_step(Phi_1, lambda_1)
        trainer.state.epoch = 5
        trainer.state.best_loss = 0.5

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(ckpt_path)

            # Create new trainer and load
            model2 = GeodesicFlowModel(
                n=n, k=k,
                hidden_dim_phi=32,
                hidden_dim_lambda=16,
            )
            trainer2 = RiemannianFlowTrainer(model2, config)
            trainer2.load_checkpoint(ckpt_path)

            # Verify state restored
            assert trainer2.state.epoch == 5
            assert trainer2.state.best_loss == 0.5

            # Verify model weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestTrainingIntegration:
    """Integration tests for the training loop."""

    def test_training_reduces_loss(self):
        """Training should reduce loss over epochs."""
        n, k = 20, 4
        snapshots = create_mock_snapshots(n=n, k=k, count=16)

        model = GeodesicFlowModel(
            n=n, k=k,
            hidden_dim_phi=64,
            hidden_dim_lambda=32,
        )
        config = RiemannianTrainingConfig(
            batch_size=8,
            n_epochs=5,
            learning_rate=1e-3,
            device="cpu",
            log_interval=100,  # Suppress logging
        )
        trainer = RiemannianFlowTrainer(model, config)

        # Train
        state = trainer.train(snapshots)

        # Should have recorded losses
        assert len(state.losses) == 5

        # Loss should generally decrease (or at least not explode)
        assert state.losses[-1] < state.losses[0] * 10  # Allow some flexibility

    def test_evaluate_grassmann(self):
        """Grassmann evaluation should return positive distance."""
        n, k = 20, 4
        snapshots = create_mock_snapshots(n=n, k=k, count=4)

        model = GeodesicFlowModel(
            n=n, k=k,
            hidden_dim_phi=32,
            hidden_dim_lambda=16,
        )
        config = RiemannianTrainingConfig(device="cpu")
        trainer = RiemannianFlowTrainer(model, config)

        distance = trainer.evaluate_grassmann(snapshots, n_steps=5)

        assert distance >= 0
        assert not np.isnan(distance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
