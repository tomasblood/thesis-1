"""
Optimal transport alignment for spectral representations.

This module implements the "inner flow" / alignment oracle that generates
transport-consistent supervision targets during training. It resolves:
- Basis ambiguity (rotations/sign flips in eigenvectors)
- Sample correspondence ambiguity between snapshots
- Mass creation/destruction (open systems)

Key principle: Alignment is performed between spectral representations,
not raw data points. We align geometric structure, not individual points.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import ot
    HAS_OT = True
except ImportError:
    ot = None
    HAS_OT = False


def _require_ot():
    """Raise ImportError if POT is not available."""
    if not HAS_OT:
        raise ImportError(
            "Optimal transport features require the 'pot' package. "
            "Install with: pip install pot"
        )



@dataclass
class AlignmentResult:
    """
    Result of transport-based alignment between spectral representations.

    Attributes:
        transport_plan: Soft alignment operator P of shape (N_s, N_t)
        aligned_target: Transport-aligned target Phi_{t+1}^{aligned}
        mass_source: Source marginal (sum over target dimension)
        mass_target: Target marginal (sum over source dimension)
        transport_cost: Total transport cost
        convergence_info: Optional convergence diagnostics
    """

    transport_plan: NDArray[np.floating]
    aligned_target: NDArray[np.floating]
    mass_source: NDArray[np.floating]
    mass_target: NDArray[np.floating]
    transport_cost: float
    convergence_info: Optional[dict] = None


class TransportAlignment:
    """
    Optimal transport-based alignment for spectral representations.

    Provides transport-consistent targets for training the outer flow model.
    Supports both balanced and unbalanced transport to handle mass
    creation/destruction in open systems.

    The alignment process:
    1. Compute cost matrix between spectral representations
    2. Solve optimal transport problem
    3. Apply transport plan to align target to source frame

    Attributes:
        method: Transport method ('balanced', 'unbalanced', 'partial')
        reg: Entropic regularization strength
        reg_m: Marginal relaxation for unbalanced transport
        cost_type: Type of cost function to use
    """

    def __init__(
        self,
        method: Literal["balanced", "unbalanced", "partial"] = "unbalanced",
        reg: float = 0.1,
        reg_m: float = 1.0,
        cost_type: Literal["euclidean", "spectral", "geometric"] = "spectral",
        max_iter: int = 1000,
        tol: float = 1e-9,
    ):
        """
        Initialize transport alignment.

        Args:
            method: Transport formulation
                - 'balanced': Standard OT with mass conservation
                - 'unbalanced': Relaxed marginals (KL divergence penalty)
                - 'partial': Partial transport (mass can be discarded)
            reg: Entropic regularization (higher = smoother transport)
            reg_m: Marginal relaxation strength for unbalanced transport
                   (lower = more mass flexibility)
            cost_type: Cost function for transport
                - 'euclidean': L2 distance between spectral rows
                - 'spectral': Weighted by eigenvalue importance
                - 'geometric': Distance in local geometry fingerprints
            max_iter: Maximum Sinkhorn iterations
            tol: Convergence tolerance
        """
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.cost_type = cost_type
        self.max_iter = max_iter
        self.tol = tol

    def compute_cost_matrix(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
        eigenvalues_source: Optional[NDArray[np.floating]] = None,
        eigenvalues_target: Optional[NDArray[np.floating]] = None,
        adjacency_source: Optional[NDArray[np.floating]] = None,
        adjacency_target: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """
        Compute cost matrix between source and target spectral representations.

        Args:
            Phi_source: Source spectral embedding (N_s, k)
            Phi_target: Target spectral embedding (N_t, k)
            eigenvalues_source: Source eigenvalues for weighting
            eigenvalues_target: Target eigenvalues for weighting
            adjacency_source: Source adjacency for geometric cost
            adjacency_target: Target adjacency for geometric cost

        Returns:
            Cost matrix C of shape (N_s, N_t)
        """
        if self.cost_type == "euclidean":
            return self._euclidean_cost(Phi_source, Phi_target)

        elif self.cost_type == "spectral":
            return self._spectral_cost(
                Phi_source, Phi_target,
                eigenvalues_source, eigenvalues_target
            )

        elif self.cost_type == "geometric":
            return self._geometric_cost(
                Phi_source, Phi_target,
                adjacency_source, adjacency_target
            )

        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

    def _euclidean_cost(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Squared Euclidean distance between spectral rows."""
        # ||phi_i - phi_j||^2 = ||phi_i||^2 + ||phi_j||^2 - 2*phi_i^T*phi_j
        norm_s = np.sum(Phi_source ** 2, axis=1, keepdims=True)
        norm_t = np.sum(Phi_target ** 2, axis=1, keepdims=True)
        cross = Phi_source @ Phi_target.T

        C = norm_s + norm_t.T - 2 * cross
        return np.maximum(C, 0)  # Numerical stability

    def _spectral_cost(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
        eigenvalues_source: Optional[NDArray[np.floating]] = None,
        eigenvalues_target: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """
        Spectral-weighted cost emphasizing low-frequency modes.

        Low eigenvalue modes (smooth, global) are weighted more heavily
        than high eigenvalue modes (local, potentially noisy).
        """
        k = Phi_source.shape[1]

        # Default to uniform if eigenvalues not provided
        if eigenvalues_source is None:
            weights = np.ones(k)
        else:
            # Weight by inverse eigenvalue (smooth modes matter more)
            # Use softmax-like normalization
            eig = np.abs(eigenvalues_source) + 1e-10
            weights = 1.0 / eig
            weights = weights / np.sum(weights) * k

        # Weighted difference
        Phi_s_weighted = Phi_source * np.sqrt(weights)
        Phi_t_weighted = Phi_target * np.sqrt(weights)

        return self._euclidean_cost(Phi_s_weighted, Phi_t_weighted)

    def _geometric_cost(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
        adjacency_source: Optional[NDArray[np.floating]] = None,
        adjacency_target: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """
        Cost based on local geometry fingerprints.

        Uses adjacency/diffusion rows as geometric descriptors.
        Falls back to spectral cost if adjacency not provided.
        """
        if adjacency_source is None or adjacency_target is None:
            return self._euclidean_cost(Phi_source, Phi_target)

        # Normalize adjacency rows (diffusion-like)
        A_s = adjacency_source / (adjacency_source.sum(axis=1, keepdims=True) + 1e-10)
        A_t = adjacency_target / (adjacency_target.sum(axis=1, keepdims=True) + 1e-10)

        # Combine spectral and geometric features
        # Project adjacency to spectral dimension for comparability
        geom_s = A_s @ Phi_source
        geom_t = A_t @ Phi_target

        # Combined feature
        feat_s = np.hstack([Phi_source, geom_s])
        feat_t = np.hstack([Phi_target, geom_t])

        return self._euclidean_cost(feat_s, feat_t)

    def align(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
        eigenvalues_source: Optional[NDArray[np.floating]] = None,
        eigenvalues_target: Optional[NDArray[np.floating]] = None,
        mass_source: Optional[NDArray[np.floating]] = None,
        mass_target: Optional[NDArray[np.floating]] = None,
        **cost_kwargs,
    ) -> AlignmentResult:
        """
        Compute optimal transport alignment between spectral representations.

        Args:
            Phi_source: Source spectral embedding (N_s, k)
            Phi_target: Target spectral embedding (N_t, k)
            eigenvalues_source: Source eigenvalues for cost weighting
            eigenvalues_target: Target eigenvalues for cost weighting
            mass_source: Source mass distribution (uniform if None)
            mass_target: Target mass distribution (uniform if None)
            **cost_kwargs: Additional arguments for cost computation

        Returns:
            AlignmentResult with transport plan and aligned target
        """
        _require_ot()
        N_s = Phi_source.shape[0]
        N_t = Phi_target.shape[0]

        # Default uniform masses
        if mass_source is None:
            mass_source = np.ones(N_s) / N_s
        if mass_target is None:
            mass_target = np.ones(N_t) / N_t

        # Compute cost matrix
        C = self.compute_cost_matrix(
            Phi_source, Phi_target,
            eigenvalues_source, eigenvalues_target,
            **cost_kwargs
        )

        # Solve transport problem
        if self.method == "balanced":
            P, info = self._solve_balanced(C, mass_source, mass_target)

        elif self.method == "unbalanced":
            P, info = self._solve_unbalanced(C, mass_source, mass_target)

        elif self.method == "partial":
            P, info = self._solve_partial(C, mass_source, mass_target)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute aligned target
        aligned_target = self._apply_transport(P, Phi_target)

        # Compute transport cost
        transport_cost = np.sum(P * C)

        # Compute effective marginals
        effective_mass_source = P.sum(axis=1)
        effective_mass_target = P.sum(axis=0)

        return AlignmentResult(
            transport_plan=P,
            aligned_target=aligned_target,
            mass_source=effective_mass_source,
            mass_target=effective_mass_target,
            transport_cost=transport_cost,
            convergence_info=info,
        )

    def _solve_balanced(
        self,
        C: NDArray[np.floating],
        a: NDArray[np.floating],
        b: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], dict]:
        """Solve balanced OT with Sinkhorn algorithm."""
        P = ot.sinkhorn(
            a, b, C, self.reg,
            numItermax=self.max_iter,
            stopThr=self.tol,
            log=False,
        )
        return P, {"method": "sinkhorn"}

    def _solve_unbalanced(
        self,
        C: NDArray[np.floating],
        a: NDArray[np.floating],
        b: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], dict]:
        """Solve unbalanced OT with Sinkhorn-Knopp algorithm."""
        P = ot.unbalanced.sinkhorn_unbalanced(
            a, b, C, self.reg, self.reg_m,
            numItermax=self.max_iter,
            stopThr=self.tol,
        )
        return P, {"method": "sinkhorn_unbalanced", "reg_m": self.reg_m}

    def _solve_partial(
        self,
        C: NDArray[np.floating],
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        m: Optional[float] = None,
    ) -> Tuple[NDArray[np.floating], dict]:
        """Solve partial OT (transport only fraction of mass)."""
        if m is None:
            m = 0.8 * min(a.sum(), b.sum())

        P = ot.partial.partial_wasserstein(
            a, b, C, m=m,
            numItermax=self.max_iter,
        )
        return P, {"method": "partial", "transported_mass": m}

    def _apply_transport(
        self,
        P: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply transport plan to align target to source frame.

        The aligned target lives in the same representation gauge as source.

        Args:
            P: Transport plan (N_s, N_t)
            Phi_target: Target spectral embedding (N_t, k)

        Returns:
            Aligned target of shape (N_s, k)
        """
        # Normalize transport plan row-wise
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        P_normalized = P / row_sums

        # Apply barycentric projection
        aligned = P_normalized @ Phi_target

        return aligned

    def compute_mass_change(
        self,
        alignment_result: AlignmentResult,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute mass creation and destruction from alignment.

        Args:
            alignment_result: Result from align()

        Returns:
            mass_created: Mass appearing at each target position
            mass_destroyed: Mass disappearing at each source position
        """
        N_s = alignment_result.mass_source.shape[0]
        N_t = alignment_result.mass_target.shape[0]

        # Expected uniform mass
        expected_source = 1.0 / N_s
        expected_target = 1.0 / N_t

        # Deviation from expected
        mass_destroyed = expected_source - alignment_result.mass_source
        mass_destroyed = np.maximum(mass_destroyed, 0)

        mass_created = alignment_result.mass_target - expected_target
        mass_created = np.maximum(mass_created, 0)

        return mass_created, mass_destroyed


class BasisAligner:
    """
    Resolve basis ambiguity in spectral embeddings.

    Eigenvectors are defined only up to sign flips and, for repeated eigenvalues,
    rotations within the eigenspace. This class finds the optimal orthogonal
    transformation to align bases.
    """

    def __init__(self, allow_reflection: bool = True):
        """
        Initialize basis aligner.

        Args:
            allow_reflection: Whether to allow reflections (det = -1)
        """
        self.allow_reflection = allow_reflection

    def align_bases(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Find optimal orthogonal transformation to align bases.

        Solves the Procrustes problem: min_Q ||Phi_source - Phi_target @ Q||_F
        subject to Q^T Q = I.

        Args:
            Phi_source: Source embedding (N, k)
            Phi_target: Target embedding (N, k) - must have same shape

        Returns:
            Q: Optimal orthogonal matrix (k, k)
            Phi_aligned: Aligned target embedding
        """
        if Phi_source.shape != Phi_target.shape:
            raise ValueError(
                f"Shape mismatch: source {Phi_source.shape} vs target {Phi_target.shape}"
            )

        # SVD of cross-covariance
        M = Phi_source.T @ Phi_target
        U, _, Vt = np.linalg.svd(M)

        # Optimal rotation/reflection
        Q = Vt.T @ U.T

        if not self.allow_reflection:
            # Ensure det(Q) = +1
            if np.linalg.det(Q) < 0:
                Vt[-1, :] *= -1
                Q = Vt.T @ U.T

        Phi_aligned = Phi_target @ Q

        return Q, Phi_aligned

    def align_signs(
        self,
        Phi_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Simple sign alignment for eigenvectors.

        Flips signs of target columns to best match source.

        Args:
            Phi_source: Source embedding
            Phi_target: Target embedding

        Returns:
            Sign-aligned target
        """
        k = Phi_source.shape[1]
        signs = np.ones(k)

        for j in range(k):
            if np.dot(Phi_source[:, j], Phi_target[:, j]) < 0:
                signs[j] = -1

        return Phi_target * signs
