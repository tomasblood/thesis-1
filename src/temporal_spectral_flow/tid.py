"""
Temporal Intrinsic Dimension (TID) computation.

TID is the primary analytical output of the system. It identifies the number
of spectral modes that represent stable, long-term structure rather than
transient noise.

Definition: TID is the count (or effective count) of spectral dimensions that:
1. Evolve smoothly under the learned flow
2. Require low transport energy to persist
3. Do not rely on artificial mass injection

This provides a statistical notion of effective dimensionality defined by
temporal consistency rather than variance.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch

from temporal_spectral_flow.flow import SpectralFlowModel, numpy_to_torch, torch_to_numpy
from temporal_spectral_flow.spectral import SpectralSnapshot
from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.transport import TransportAlignment, AlignmentResult


@dataclass
class DimensionStability:
    """
    Stability analysis for a single spectral dimension.

    Attributes:
        dimension_index: Index of the spectral dimension
        stability_score: Overall stability score in [0, 1]
        flow_energy: Energy required to evolve this dimension
        curvature: Average curvature of trajectory
        mass_consistency: How well mass is preserved
        eigenvalue_stability: Stability of the corresponding eigenvalue
    """

    dimension_index: int
    stability_score: float
    flow_energy: float
    curvature: float
    mass_consistency: float
    eigenvalue_stability: float


@dataclass
class TIDResult:
    """
    Result of Temporal Intrinsic Dimension analysis.

    Attributes:
        tid_count: Integer count of stable dimensions
        tid_effective: Effective dimension (can be fractional)
        dimension_stabilities: Stability analysis for each dimension
        stability_threshold: Threshold used for counting
        overall_stability: Aggregate stability measure
    """

    tid_count: int
    tid_effective: float
    dimension_stabilities: list[DimensionStability]
    stability_threshold: float
    overall_stability: float


class TemporalIntrinsicDimension:
    """
    Computes Temporal Intrinsic Dimension from spectral flow analysis.

    TID identifies which degrees of freedom represent stable signal
    versus transient noise by analyzing temporal evolution patterns.
    """

    def __init__(
        self,
        model: Optional[SpectralFlowModel] = None,
        aligner: Optional[TransportAlignment] = None,
        stability_threshold: float = 0.5,
        aggregation_method: Literal["count", "soft", "entropy"] = "soft",
    ):
        """
        Initialize TID analyzer.

        Args:
            model: Trained spectral flow model (optional for some analyses)
            aligner: Transport alignment module (optional)
            stability_threshold: Threshold for counting stable dimensions
            aggregation_method: How to aggregate stability scores
                - 'count': Hard count above threshold
                - 'soft': Sum of stability scores
                - 'entropy': Entropy-based effective dimension
        """
        self.model = model
        self.aligner = aligner or TransportAlignment()
        self.stability_threshold = stability_threshold
        self.aggregation_method = aggregation_method

    def compute_tid(
        self,
        snapshots: list[SpectralSnapshot],
        use_flow: bool = True,
    ) -> TIDResult:
        """
        Compute Temporal Intrinsic Dimension from spectral snapshots.

        Args:
            snapshots: List of spectral snapshots over time
            use_flow: Whether to use learned flow (requires model)

        Returns:
            TIDResult with dimension analysis
        """
        if len(snapshots) < 2:
            raise ValueError("Need at least 2 snapshots for TID")

        k = snapshots[0].k

        # Compute per-dimension stability
        stabilities = []
        for j in range(k):
            stability = self._analyze_dimension(snapshots, j, use_flow)
            stabilities.append(stability)

        # Sort by stability score
        stabilities.sort(key=lambda s: s.stability_score, reverse=True)

        # Compute TID
        scores = np.array([s.stability_score for s in stabilities])

        if self.aggregation_method == "count":
            tid_count = int(np.sum(scores >= self.stability_threshold))
            tid_effective = float(tid_count)

        elif self.aggregation_method == "soft":
            tid_count = int(np.sum(scores >= self.stability_threshold))
            tid_effective = float(np.sum(scores))

        elif self.aggregation_method == "entropy":
            # Normalize to probability distribution
            p = scores / (np.sum(scores) + 1e-10)
            # Entropy-based effective dimension
            entropy = -np.sum(p * np.log(p + 1e-10))
            tid_effective = np.exp(entropy)
            tid_count = int(np.round(tid_effective))

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        overall_stability = float(np.mean(scores))

        return TIDResult(
            tid_count=tid_count,
            tid_effective=tid_effective,
            dimension_stabilities=stabilities,
            stability_threshold=self.stability_threshold,
            overall_stability=overall_stability,
        )

    def _analyze_dimension(
        self,
        snapshots: list[SpectralSnapshot],
        dim_idx: int,
        use_flow: bool,
    ) -> DimensionStability:
        """
        Analyze stability of a single spectral dimension.

        Args:
            snapshots: Spectral snapshots
            dim_idx: Index of dimension to analyze
            use_flow: Whether to use learned flow

        Returns:
            DimensionStability for this dimension
        """
        T = len(snapshots)

        # Extract dimension trajectory (column of Phi matrices)
        trajectories = []
        for snap in snapshots:
            trajectories.append(snap.Phi[:, dim_idx])

        # Compute flow energy
        flow_energy = self._compute_flow_energy(trajectories, snapshots, dim_idx, use_flow)

        # Compute trajectory curvature
        curvature = self._compute_curvature(trajectories)

        # Compute mass consistency
        mass_consistency = self._compute_mass_consistency(snapshots, dim_idx)

        # Compute eigenvalue stability
        eigenvalues = np.array([snap.eigenvalues[dim_idx] for snap in snapshots])
        eigenvalue_stability = self._compute_eigenvalue_stability(eigenvalues)

        # Aggregate into stability score
        # Higher score = more stable = should be counted in TID
        stability_score = self._aggregate_stability(
            flow_energy, curvature, mass_consistency, eigenvalue_stability
        )

        return DimensionStability(
            dimension_index=dim_idx,
            stability_score=stability_score,
            flow_energy=flow_energy,
            curvature=curvature,
            mass_consistency=mass_consistency,
            eigenvalue_stability=eigenvalue_stability,
        )

    def _compute_flow_energy(
        self,
        trajectories: list[NDArray],
        snapshots: list[SpectralSnapshot],
        dim_idx: int,
        use_flow: bool,
    ) -> float:
        """
        Compute energy required to evolve this dimension.

        Low energy = smooth evolution = stable dimension.
        """
        if use_flow and self.model is not None:
            # Use learned velocity field
            energies = []
            device = next(self.model.parameters()).device

            for t in range(len(snapshots) - 1):
                Phi = numpy_to_torch(snapshots[t].Phi, device)
                t_tensor = torch.tensor([float(t)], device=device)

                with torch.no_grad():
                    v = self.model.predict_velocity(Phi.unsqueeze(0), t_tensor)
                    v = v.squeeze(0)

                # Energy in this dimension
                v_dim = v[:, dim_idx]
                energy = torch.mean(v_dim ** 2).item()
                energies.append(energy)

            total_energy = np.mean(energies)

        else:
            # Use simple finite difference
            energies = []
            for t in range(len(trajectories) - 1):
                diff = trajectories[t + 1] - trajectories[t]
                energy = np.mean(diff ** 2)
                energies.append(energy)

            total_energy = np.mean(energies)

        # Normalize to [0, 1] where 1 = low energy (stable)
        # Use exponential mapping
        return float(np.exp(-total_energy))

    def _compute_curvature(
        self,
        trajectories: list[NDArray],
    ) -> float:
        """
        Compute average curvature of dimension trajectory.

        Low curvature = smooth path = stable dimension.
        """
        if len(trajectories) < 3:
            return 1.0  # Can't compute curvature

        curvatures = []
        for t in range(1, len(trajectories) - 1):
            # Second derivative approximation
            d2 = trajectories[t + 1] - 2 * trajectories[t] + trajectories[t - 1]
            curvature = np.mean(np.abs(d2))
            curvatures.append(curvature)

        avg_curvature = np.mean(curvatures)

        # Normalize to [0, 1] where 1 = low curvature
        return float(np.exp(-avg_curvature))

    def _compute_mass_consistency(
        self,
        snapshots: list[SpectralSnapshot],
        dim_idx: int,
    ) -> float:
        """
        Compute how consistently mass is preserved in this dimension.

        Uses transport analysis to measure mass creation/destruction.
        """
        mass_deviations = []

        for t in range(len(snapshots) - 1):
            Phi_t = snapshots[t].Phi
            Phi_next = snapshots[t + 1].Phi

            # If sizes match, mass is trivially consistent
            if Phi_t.shape[0] == Phi_next.shape[0]:
                mass_deviations.append(0.0)
                continue

            # Compute transport and measure mass change
            result = self.aligner.align(
                Phi_t, Phi_next,
                eigenvalues_source=snapshots[t].eigenvalues,
                eigenvalues_target=snapshots[t + 1].eigenvalues,
            )

            # Mass deviation for this dimension
            expected_mass = 1.0 / Phi_t.shape[0]
            actual_mass = result.mass_source

            # Weight by contribution to this dimension
            weights = np.abs(Phi_t[:, dim_idx])
            weights = weights / (np.sum(weights) + 1e-10)

            weighted_deviation = np.sum(weights * np.abs(actual_mass - expected_mass))
            mass_deviations.append(weighted_deviation)

        avg_deviation = np.mean(mass_deviations)

        # Normalize to [0, 1] where 1 = consistent mass
        return float(np.exp(-10 * avg_deviation))

    def _compute_eigenvalue_stability(
        self,
        eigenvalues: NDArray,
    ) -> float:
        """
        Compute stability of the corresponding eigenvalue over time.

        Stable eigenvalue = stable geometric scale = stable dimension.
        """
        if len(eigenvalues) < 2:
            return 1.0

        # Coefficient of variation
        mean_eig = np.mean(eigenvalues)
        std_eig = np.std(eigenvalues)

        if mean_eig < 1e-10:
            return 1.0  # Zero eigenvalue is stable

        cv = std_eig / (mean_eig + 1e-10)

        # Normalize to [0, 1] where 1 = stable
        return float(np.exp(-cv))

    def _aggregate_stability(
        self,
        flow_energy: float,
        curvature: float,
        mass_consistency: float,
        eigenvalue_stability: float,
    ) -> float:
        """
        Aggregate component scores into overall stability.

        All inputs are in [0, 1] where 1 = stable.
        """
        # Weighted geometric mean
        weights = np.array([0.3, 0.2, 0.25, 0.25])
        scores = np.array([flow_energy, curvature, mass_consistency, eigenvalue_stability])

        # Geometric mean with weights
        log_scores = np.log(scores + 1e-10)
        weighted_log_mean = np.sum(weights * log_scores)
        stability = np.exp(weighted_log_mean)

        return float(np.clip(stability, 0, 1))

    def compute_tid_over_time(
        self,
        snapshots: list[SpectralSnapshot],
        window_size: int = 5,
    ) -> NDArray[np.floating]:
        """
        Compute TID in sliding windows over time.

        Useful for detecting distribution shifts and structural changes.

        Args:
            snapshots: Full sequence of snapshots
            window_size: Number of snapshots per window

        Returns:
            Array of TID values, one per window
        """
        n_windows = len(snapshots) - window_size + 1
        tid_values = []

        for i in range(n_windows):
            window = snapshots[i : i + window_size]
            result = self.compute_tid(window, use_flow=False)
            tid_values.append(result.tid_effective)

        return np.array(tid_values)

    def detect_dimension_emergence(
        self,
        snapshots: list[SpectralSnapshot],
        threshold: float = 0.3,
    ) -> list[Tuple[int, int]]:
        """
        Detect when new stable dimensions emerge over time.

        Returns (time_index, dimension_index) pairs for emergence events.
        """
        emergences = []

        for t in range(1, len(snapshots)):
            prev_result = self.compute_tid(snapshots[:t], use_flow=False)
            curr_result = self.compute_tid(snapshots[: t + 1], use_flow=False)

            prev_stable = {s.dimension_index for s in prev_result.dimension_stabilities
                          if s.stability_score >= self.stability_threshold}
            curr_stable = {s.dimension_index for s in curr_result.dimension_stabilities
                          if s.stability_score >= self.stability_threshold}

            new_stable = curr_stable - prev_stable
            for dim in new_stable:
                emergences.append((t, dim))

        return emergences

    def detect_dimension_decay(
        self,
        snapshots: list[SpectralSnapshot],
    ) -> list[Tuple[int, int]]:
        """
        Detect when stable dimensions decay (become unstable).

        Returns (time_index, dimension_index) pairs for decay events.
        """
        decays = []

        for t in range(1, len(snapshots)):
            prev_result = self.compute_tid(snapshots[:t], use_flow=False)
            curr_result = self.compute_tid(snapshots[: t + 1], use_flow=False)

            prev_stable = {s.dimension_index for s in prev_result.dimension_stabilities
                          if s.stability_score >= self.stability_threshold}
            curr_stable = {s.dimension_index for s in curr_result.dimension_stabilities
                          if s.stability_score >= self.stability_threshold}

            lost_stable = prev_stable - curr_stable
            for dim in lost_stable:
                decays.append((t, dim))

        return decays


def compute_tid_from_data(
    data_snapshots: list[NDArray[np.floating]],
    k: int = 10,
    model: Optional[SpectralFlowModel] = None,
    stability_threshold: float = 0.5,
) -> TIDResult:
    """
    Convenience function to compute TID directly from data.

    Args:
        data_snapshots: List of data matrices
        k: Spectral dimension
        model: Optional trained flow model
        stability_threshold: Threshold for stable dimensions

    Returns:
        TIDResult with analysis
    """
    from temporal_spectral_flow.spectral import TemporalSpectralEmbedding

    embedder = TemporalSpectralEmbedding(k=k)
    snapshots = embedder.process_sequence(data_snapshots)

    tid_analyzer = TemporalIntrinsicDimension(
        model=model,
        stability_threshold=stability_threshold,
    )

    return tid_analyzer.compute_tid(snapshots)
