"""
Inference utilities for Geodesic Flow Model.

Provides interpolation and extrapolation using the trained flow:
- Interpolate between observed frames
- Extrapolate beyond observations
- Generate smooth trajectories

All operations use the learned dynamics conditioned on target frames.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch

from temporal_spectral_flow.riemannian_flow import GeodesicFlowModel, uniform_stiefel
from temporal_spectral_flow.grassmann import grassmann_distance, projection_frobenius_efficient
from temporal_spectral_flow.spectral import SpectralSnapshot


def interpolate(
    model: GeodesicFlowModel,
    Phi_A: Union[torch.Tensor, NDArray],
    lambda_A: Union[torch.Tensor, NDArray],
    Phi_B: Union[torch.Tensor, NDArray],
    lambda_B: Union[torch.Tensor, NDArray],
    s_target: float,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Interpolate between two frames using the learned flow.

    Integrates from (Φ_A, λ_A) toward (Φ_B, λ_B) stopping at s_target.

    Args:
        model: Trained GeodesicFlowModel
        Phi_A: Start eigenvectors (n, k)
        lambda_A: Start eigenvalues (k,)
        Phi_B: End eigenvectors (n, k) - conditioning target
        lambda_B: End eigenvalues (k,) - conditioning target
        s_target: Interpolation parameter in [0, 1]
        n_steps: Integration steps
        device: Torch device

    Returns:
        (Phi_s, lambda_s): Interpolated frame
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Convert to tensors
    Phi_A = _to_tensor(Phi_A, device)
    lambda_A = _to_tensor(lambda_A, device)
    Phi_B = _to_tensor(Phi_B, device)
    lambda_B = _to_tensor(lambda_B, device)

    with torch.no_grad():
        Phi_s, lambda_s = model.integrate(
            Phi_A, lambda_A,
            Phi_B, lambda_B,
            t_end=s_target,
            n_steps=n_steps,
        )

    return Phi_s.cpu().numpy(), lambda_s.cpu().numpy()


def get_trajectory(
    model: GeodesicFlowModel,
    Phi_A: Union[torch.Tensor, NDArray],
    lambda_A: Union[torch.Tensor, NDArray],
    Phi_B: Union[torch.Tensor, NDArray],
    lambda_B: Union[torch.Tensor, NDArray],
    n_points: int = 10,
    n_steps_per_segment: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Generate trajectory from frame A to frame B.

    Returns frames at evenly spaced times along the flow.

    Args:
        model: Trained GeodesicFlowModel
        Phi_A: Start eigenvectors
        lambda_A: Start eigenvalues
        Phi_B: End eigenvectors (conditioning)
        lambda_B: End eigenvalues (conditioning)
        n_points: Number of trajectory points (including endpoints)
        n_steps_per_segment: Integration steps between points
        device: Torch device

    Returns:
        (Phi_trajectory, lambda_trajectory): Lists of frames
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    Phi_A = _to_tensor(Phi_A, device)
    lambda_A = _to_tensor(lambda_A, device)
    Phi_B = _to_tensor(Phi_B, device)
    lambda_B = _to_tensor(lambda_B, device)

    s_values = np.linspace(0, 1, n_points)

    Phi_trajectory = []
    lambda_trajectory = []

    with torch.no_grad():
        for s in s_values:
            if s == 0:
                Phi_s, lambda_s = Phi_A, lambda_A
            elif s == 1:
                # Integrate all the way
                Phi_s, lambda_s = model.integrate(
                    Phi_A, lambda_A,
                    Phi_B, lambda_B,
                    t_end=1.0,
                    n_steps=n_steps_per_segment * n_points,
                )
            else:
                Phi_s, lambda_s = model.integrate(
                    Phi_A, lambda_A,
                    Phi_B, lambda_B,
                    t_end=s,
                    n_steps=max(1, int(n_steps_per_segment * n_points * s)),
                )

            Phi_trajectory.append(Phi_s.cpu().numpy())
            lambda_trajectory.append(lambda_s.cpu().numpy())

    return Phi_trajectory, lambda_trajectory


def extrapolate(
    model: GeodesicFlowModel,
    frames: List[SpectralSnapshot],
    n_steps_forward: int,
    dt: float = 0.1,
    n_integration_steps: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Extrapolate beyond the last observed frame.

    Uses the last two frames to establish direction, then continues
    the learned dynamics.

    Note: For extrapolation, we condition on the last frame and integrate
    past t=1, which requires the model to generalize beyond training.

    Args:
        model: Trained GeodesicFlowModel
        frames: Observed spectral frames (at least 2)
        n_steps_forward: Number of extrapolation steps
        dt: Time step size for extrapolation
        n_integration_steps: Integration steps per extrapolation step
        device: Torch device

    Returns:
        (Phi_future, lambda_future): Lists of extrapolated frames
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for extrapolation")

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Start from last frame, condition on it (extrapolate in same direction)
    Phi_current = _to_tensor(frames[-1].Phi, device)
    lambda_current = _to_tensor(frames[-1].eigenvalues, device)

    # For conditioning, use the last frame as target
    # This means we're asking "continue from here"
    Phi_target = Phi_current.clone()
    lambda_target = lambda_current.clone()

    Phi_future = [Phi_current.cpu().numpy()]
    lambda_future = [lambda_current.cpu().numpy()]

    with torch.no_grad():
        for _ in range(n_steps_forward):
            # Integrate one step forward
            # We use dt as the integration time, conditioning on current state
            Phi_next, lambda_next = model.integrate(
                Phi_current, lambda_current,
                Phi_target, lambda_target,
                t_end=dt,
                n_steps=n_integration_steps,
            )

            Phi_future.append(Phi_next.cpu().numpy())
            lambda_future.append(lambda_next.cpu().numpy())

            # Update for next step
            Phi_current = Phi_next
            lambda_current = lambda_next

    return Phi_future, lambda_future


def interpolate_sequence(
    model: GeodesicFlowModel,
    frames: List[SpectralSnapshot],
    n_interp: int = 5,
    n_steps: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[List[NDArray], List[NDArray], List[float]]:
    """
    Interpolate between all consecutive frame pairs in a sequence.

    Args:
        model: Trained GeodesicFlowModel
        frames: Observed spectral frames
        n_interp: Number of interpolation points between each pair
        n_steps: Integration steps for each interpolation
        device: Torch device

    Returns:
        (Phi_sequence, lambda_sequence, times): Dense trajectory with times
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    Phi_sequence = []
    lambda_sequence = []
    times = []

    n_frames = len(frames)

    for i in range(n_frames - 1):
        Phi_A = _to_tensor(frames[i].Phi, device)
        lambda_A = _to_tensor(frames[i].eigenvalues, device)
        Phi_B = _to_tensor(frames[i + 1].Phi, device)
        lambda_B = _to_tensor(frames[i + 1].eigenvalues, device)

        # Interpolation times (exclude endpoint except for last pair)
        if i == n_frames - 2:
            s_values = np.linspace(0, 1, n_interp + 1)
        else:
            s_values = np.linspace(0, 1, n_interp + 1)[:-1]

        with torch.no_grad():
            for s in s_values:
                if s == 0:
                    Phi_s, lambda_s = Phi_A, lambda_A
                else:
                    Phi_s, lambda_s = model.integrate(
                        Phi_A, lambda_A,
                        Phi_B, lambda_B,
                        t_end=s,
                        n_steps=n_steps,
                    )

                Phi_sequence.append(Phi_s.cpu().numpy())
                lambda_sequence.append(lambda_s.cpu().numpy())
                times.append(i + s)

    return Phi_sequence, lambda_sequence, times


def evaluate_interpolation(
    model: GeodesicFlowModel,
    frames: List[SpectralSnapshot],
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate interpolation quality on held-out middle frames.

    For each triplet (A, B, C), interpolate from A to C and measure
    error at B.

    Args:
        model: Trained GeodesicFlowModel
        frames: Observed spectral frames (at least 3)
        n_steps: Integration steps
        device: Torch device

    Returns:
        Dictionary with evaluation metrics
    """
    if len(frames) < 3:
        raise ValueError("Need at least 3 frames for evaluation")

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    grassmann_errors = []
    frobenius_errors = []
    lambda_errors = []

    # Evaluate on triplets
    for i in range(len(frames) - 2):
        Phi_A = frames[i].Phi
        lambda_A = frames[i].eigenvalues
        Phi_B = frames[i + 1].Phi  # Ground truth middle
        lambda_B = frames[i + 1].eigenvalues
        Phi_C = frames[i + 2].Phi
        lambda_C = frames[i + 2].eigenvalues

        # Interpolate from A to C at s=0.5
        Phi_pred, lambda_pred = interpolate(
            model,
            Phi_A, lambda_A,
            Phi_C, lambda_C,
            s_target=0.5,
            n_steps=n_steps,
            device=device,
        )

        # Compute errors
        grassmann_errors.append(grassmann_distance(Phi_pred, Phi_B))
        frobenius_errors.append(projection_frobenius_efficient(Phi_pred, Phi_B))
        lambda_errors.append(np.linalg.norm(lambda_pred - lambda_B))

    return {
        "grassmann_error_mean": np.mean(grassmann_errors),
        "grassmann_error_std": np.std(grassmann_errors),
        "frobenius_error_mean": np.mean(frobenius_errors),
        "frobenius_error_std": np.std(frobenius_errors),
        "lambda_error_mean": np.mean(lambda_errors),
        "lambda_error_std": np.std(lambda_errors),
        "n_triplets": len(grassmann_errors),
    }


def generate_from_noise(
    model: GeodesicFlowModel,
    Phi_target: Union[torch.Tensor, NDArray],
    lambda_target: Union[torch.Tensor, NDArray],
    n_samples: int = 1,
    n_steps: int = 50,
    sigma_lambda: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Generate samples by integrating from noise to target.

    This is the generative direction of flow matching.

    Args:
        model: Trained GeodesicFlowModel
        Phi_target: Target eigenvectors (conditioning)
        lambda_target: Target eigenvalues (conditioning)
        n_samples: Number of samples to generate
        n_steps: Integration steps
        sigma_lambda: Noise std for eigenvalues
        device: Torch device

    Returns:
        (Phi_samples, lambda_samples): Lists of generated frames
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    Phi_target = _to_tensor(Phi_target, device)
    lambda_target = _to_tensor(lambda_target, device)

    n, k = Phi_target.shape

    Phi_samples = []
    lambda_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Sample noise
            Phi_0 = uniform_stiefel(n, k, 1, device).squeeze(0)
            lambda_0 = sigma_lambda * torch.randn(k, device=device)

            # Integrate to target
            Phi_1, lambda_1 = model.integrate(
                Phi_0, lambda_0,
                Phi_target, lambda_target,
                t_end=1.0,
                n_steps=n_steps,
            )

            Phi_samples.append(Phi_1.cpu().numpy())
            lambda_samples.append(lambda_1.cpu().numpy())

    return Phi_samples, lambda_samples


# =============================================================================
# Utility functions
# =============================================================================


def _to_tensor(
    x: Union[torch.Tensor, NDArray],
    device: torch.device,
) -> torch.Tensor:
    """Convert to tensor on device."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    return x.to(device)
