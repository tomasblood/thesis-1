"""Tests for spectral flow model."""

import numpy as np
import pytest
import torch

from temporal_spectral_flow.flow import (
    StiefelVelocityField,
    SpectralFlowModel,
    SinusoidalTimeEmbedding,
    numpy_to_torch,
    torch_to_numpy,
)
from temporal_spectral_flow.stiefel import StiefelManifold


class TestSinusoidalTimeEmbedding:
    """Tests for time embedding."""

    def test_embedding_shape(self):
        """Test output shape."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embed(t)

        assert out.shape == (3, 64)

    def test_different_times_different_embeddings(self):
        """Test that different times give different embeddings."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])

        out1 = embed(t1)
        out2 = embed(t2)

        assert not torch.allclose(out1, out2)


class TestStiefelVelocityField:
    """Tests for velocity field network."""

    @pytest.fixture
    def velocity_field(self):
        """Create velocity field."""
        return StiefelVelocityField(k=5, hidden_dims=(64, 64))

    @pytest.fixture
    def sample_input(self):
        """Generate sample input on Stiefel manifold."""
        np.random.seed(42)
        manifold = StiefelManifold(n=50, k=5)
        Phi = manifold.random_point()
        return numpy_to_torch(Phi)

    def test_output_shape_unbatched(self, velocity_field, sample_input):
        """Test output shape for single input."""
        t = torch.tensor([0.5])
        v = velocity_field(sample_input, t)

        assert v.shape == sample_input.shape

    def test_output_shape_batched(self, velocity_field):
        """Test output shape for batched input."""
        np.random.seed(42)
        manifold = StiefelManifold(n=50, k=5)

        batch = []
        for _ in range(4):
            batch.append(manifold.random_point())
        Phi = numpy_to_torch(np.stack(batch, axis=0))

        t = torch.tensor([0.0, 0.25, 0.5, 0.75])
        v = velocity_field(Phi, t)

        assert v.shape == (4, 50, 5)

    def test_output_in_tangent_space(self, velocity_field, sample_input):
        """Test that output lies in tangent space."""
        t = torch.tensor([0.5])
        v = velocity_field(sample_input, t)

        # Tangent condition: Phi^T V + V^T Phi = 0
        Phi = sample_input
        PhiTV = Phi.T @ v
        sym = PhiTV + PhiTV.T

        assert torch.allclose(sym, torch.zeros_like(sym), atol=1e-5)

    def test_gradient_flow(self, velocity_field, sample_input):
        """Test that gradients flow properly."""
        sample_input.requires_grad_(True)
        t = torch.tensor([0.5])

        v = velocity_field(sample_input, t)
        loss = v.sum()
        loss.backward()

        assert sample_input.grad is not None


class TestSpectralFlowModel:
    """Tests for full flow model."""

    @pytest.fixture
    def model(self):
        """Create flow model."""
        return SpectralFlowModel(k=5, hidden_dims=(64, 64))

    @pytest.fixture
    def sample_input(self):
        """Generate sample input."""
        np.random.seed(42)
        manifold = StiefelManifold(n=50, k=5)
        return numpy_to_torch(manifold.random_point())

    def test_integration_euler(self, model, sample_input):
        """Test Euler integration."""
        model.integration_method = "euler"

        t_start = torch.tensor([0.0])
        t_end = torch.tensor([1.0])

        result = model(sample_input, t_start, t_end, n_steps=5)

        assert result.shape == sample_input.shape
        # Should stay on manifold
        gram = result.T @ result
        assert torch.allclose(gram, torch.eye(5), atol=1e-4)

    def test_integration_midpoint(self, model, sample_input):
        """Test midpoint integration."""
        model.integration_method = "midpoint"

        t_start = torch.tensor([0.0])
        t_end = torch.tensor([1.0])

        result = model(sample_input, t_start, t_end, n_steps=5)

        assert result.shape == sample_input.shape

    def test_integration_rk4(self, model, sample_input):
        """Test RK4 integration."""
        model.integration_method = "rk4"

        t_start = torch.tensor([0.0])
        t_end = torch.tensor([1.0])

        result = model(sample_input, t_start, t_end, n_steps=5)

        assert result.shape == sample_input.shape

    def test_trajectory_generation(self, model, sample_input):
        """Test trajectory generation."""
        t_start = torch.tensor([0.0])
        t_end = torch.tensor([1.0])

        trajectory = model.trajectory(sample_input, t_start, t_end, n_steps=5)

        assert len(trajectory) == 6  # Initial + 5 steps
        for point in trajectory:
            assert point.shape == sample_input.shape

    def test_batched_integration(self, model):
        """Test batched integration."""
        np.random.seed(42)
        manifold = StiefelManifold(n=50, k=5)

        batch = np.stack([manifold.random_point() for _ in range(3)], axis=0)
        Phi = numpy_to_torch(batch)

        t_start = torch.tensor([0.0, 0.0, 0.0])
        t_end = torch.tensor([1.0, 1.0, 1.0])

        result = model(Phi, t_start, t_end, n_steps=3)

        assert result.shape == (3, 50, 5)


class TestConversionFunctions:
    """Tests for numpy/torch conversion."""

    def test_numpy_to_torch(self):
        """Test numpy to torch conversion."""
        arr = np.random.randn(10, 5).astype(np.float32)
        tensor = numpy_to_torch(arr)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 5)
        assert tensor.dtype == torch.float32

    def test_torch_to_numpy(self):
        """Test torch to numpy conversion."""
        tensor = torch.randn(10, 5)
        arr = torch_to_numpy(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 5)

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = np.random.randn(10, 5).astype(np.float32)
        tensor = numpy_to_torch(original)
        recovered = torch_to_numpy(tensor)

        assert np.allclose(original, recovered)
