"""
Spectral alignment without pointwise node comparison.

This module aligns spectral frames (Phi, lambda) across timesteps using only
spectral properties - eigenvalue matching and sign conventions that
don't require node-to-node correspondence.

Key principle: Eigenvalues are geometric invariants. Match modes by
eigenvalue proximity, then resolve eigenvector ambiguity using
spectral statistics, not node coordinates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


@dataclass
class AlignedSpectralPair:
    """
    Result of aligning two spectral frames.

    Attributes:
        Phi_source: Source eigenvector matrix (N_s, k)
        Phi_target_aligned: Target eigenvectors after alignment (N_t, k)
        lambda_source: Source eigenvalues (k,)
        lambda_target_aligned: Target eigenvalues after permutation (k,)
        eigenvalue_permutation: How target eigenvalues were reordered
        sign_flips: Signs applied to target eigenvectors
        matching_cost: Total cost of eigenvalue matching
        eigenvalue_gaps: Gaps between matched eigenvalue pairs
    """
    Phi_source: NDArray[np.floating]
    Phi_target_aligned: NDArray[np.floating]
    lambda_source: NDArray[np.floating]
    lambda_target_aligned: NDArray[np.floating]
    eigenvalue_permutation: NDArray[np.integer]
    sign_flips: NDArray[np.floating]
    matching_cost: float
    eigenvalue_gaps: NDArray[np.floating]


class SpectralMatcher:
    """
    Match eigenvalues across timesteps to establish mode correspondence.

    Eigenvalues are geometric invariants - they encode the scales of
    geometric features (bottlenecks, clusters, etc.). Matching eigenvalues
    by proximity identifies which geometric feature at time t corresponds
    to which at time t+1.

    Handles:
    - Eigenvalue reordering (crossings)
    - Missing/new modes (partial matching)
    - Degenerate eigenvalues (near-equal values)
    """

    def __init__(
        self,
        cost_type: str = "absolute",
        crossing_penalty: float = 0.0,
        degeneracy_threshold: float = 1e-6,
    ):
        """
        Initialize spectral matcher.

        Args:
            cost_type: How to measure eigenvalue distance
                - "absolute": |lambda_i - lambda_j|
                - "relative": |lambda_i - lambda_j| / (|lambda_i| + |lambda_j| + eps)
                - "log": |log(lambda_i + eps) - log(lambda_j + eps)|
            crossing_penalty: Extra cost for non-identity permutations
                (encourages eigenvalue tracking without crossings)
            degeneracy_threshold: Eigenvalues closer than this are
                considered degenerate (requiring special handling)
        """
        self.cost_type = cost_type
        self.crossing_penalty = crossing_penalty
        self.degeneracy_threshold = degeneracy_threshold

    def compute_cost_matrix(
        self,
        lambda_source: NDArray[np.floating],
        lambda_target: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute pairwise matching costs between eigenvalues.

        Args:
            lambda_source: Source eigenvalues (k_s,)
            lambda_target: Target eigenvalues (k_t,)

        Returns:
            Cost matrix of shape (k_s, k_t)
        """
        k_s = len(lambda_source)
        k_t = len(lambda_target)

        # Broadcast to compute all pairs
        lambda_s = lambda_source[:, None]  # (k_s, 1)
        lambda_t = lambda_target[None, :]  # (1, k_t)

        if self.cost_type == "absolute":
            cost = np.abs(lambda_s - lambda_t)

        elif self.cost_type == "relative":
            eps = 1e-10
            cost = np.abs(lambda_s - lambda_t) / (np.abs(lambda_s) + np.abs(lambda_t) + eps)

        elif self.cost_type == "log":
            eps = 1e-10
            cost = np.abs(np.log(lambda_s + eps) - np.log(lambda_t + eps))

        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

        # Add crossing penalty (penalize off-diagonal matches)
        if self.crossing_penalty > 0 and k_s == k_t:
            identity_bonus = np.eye(k_s) * (-self.crossing_penalty)
            cost = cost + self.crossing_penalty - identity_bonus

        return cost

    def match(
        self,
        lambda_source: NDArray[np.floating],
        lambda_target: NDArray[np.floating],
    ) -> Tuple[NDArray[np.integer], NDArray[np.integer], float]:
        """
        Find optimal matching between eigenvalue sets.

        Uses Hungarian algorithm for optimal assignment.

        Args:
            lambda_source: Source eigenvalues (k_s,)
            lambda_target: Target eigenvalues (k_t,)

        Returns:
            source_indices: Matched indices in source
            target_indices: Corresponding matched indices in target
            total_cost: Sum of matching costs
        """
        cost = self.compute_cost_matrix(lambda_source, lambda_target)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        total_cost = cost[row_ind, col_ind].sum()

        return row_ind, col_ind, float(total_cost)

    def match_with_permutation(
        self,
        lambda_source: NDArray[np.floating],
        lambda_target: NDArray[np.floating],
    ) -> Tuple[NDArray[np.integer], float]:
        """
        Match eigenvalues and return permutation for target.

        Assumes k_s == k_t. Returns permutation P such that
        lambda_target[P] is aligned with lambda_source.

        Args:
            lambda_source: Source eigenvalues (k,)
            lambda_target: Target eigenvalues (k,)

        Returns:
            permutation: Index array such that lambda_target[permutation]
                aligns with lambda_source
            total_cost: Matching cost
        """
        k_s = len(lambda_source)
        k_t = len(lambda_target)

        if k_s != k_t:
            raise ValueError(
                f"Dimension mismatch: source has {k_s}, target has {k_t}. "
                "Use match() for different-sized eigenvalue sets."
            )

        source_ind, target_ind, cost = self.match(lambda_source, lambda_target)

        # Build permutation array
        # permutation[i] = j means lambda_target[j] matches lambda_source[i]
        permutation = np.zeros(k_s, dtype=np.int64)
        permutation[source_ind] = target_ind

        return permutation, cost

    def detect_crossings(
        self,
        lambda_sequence: List[NDArray[np.floating]],
    ) -> List[Tuple[int, int, int]]:
        """
        Detect eigenvalue crossings in a temporal sequence.

        Args:
            lambda_sequence: List of eigenvalue arrays over time

        Returns:
            List of (time_index, mode_i, mode_j) where crossing occurred
        """
        crossings = []

        for t in range(len(lambda_sequence) - 1):
            lambda_t = lambda_sequence[t]
            lambda_t1 = lambda_sequence[t + 1]

            perm, _ = self.match_with_permutation(lambda_t, lambda_t1)

            # Check if permutation is non-identity
            identity = np.arange(len(perm))
            if not np.array_equal(perm, identity):
                # Find which modes crossed
                for i in range(len(perm)):
                    if perm[i] != i:
                        # Mode i at time t matches mode perm[i] at time t+1
                        crossings.append((t, i, perm[i]))

        return crossings

    def detect_degeneracies(
        self,
        eigenvalues: NDArray[np.floating],
    ) -> List[Tuple[int, int]]:
        """
        Find pairs of near-degenerate eigenvalues.

        Args:
            eigenvalues: Eigenvalue array (k,)

        Returns:
            List of (i, j) pairs where |lambda_i - lambda_j| < threshold
        """
        k = len(eigenvalues)
        degeneracies = []

        for i in range(k):
            for j in range(i + 1, k):
                if np.abs(eigenvalues[i] - eigenvalues[j]) < self.degeneracy_threshold:
                    degeneracies.append((i, j))

        return degeneracies


class SignConvention:
    """
    Establish consistent sign conventions for eigenvectors without
    using node-to-node comparison.

    Eigenvectors are defined only up to sign (phi and -phi are both valid).
    We need a convention that:
    1. Is deterministic given the spectral data
    2. Doesn't require node correspondence
    3. Produces consistent signs for "similar" eigenvectors

    Methods:
    - "max_entry": Sign of maximum absolute entry is positive
    - "sum_positive": Sum of entries is non-negative
    - "first_entry": First non-negligible entry is positive
    - "moment": Based on statistical moments of the distribution
    """

    def __init__(self, method: str = "max_entry", threshold: float = 1e-10):
        """
        Initialize sign convention.

        Args:
            method: Convention to use
            threshold: Numerical threshold for "non-negligible"
        """
        self.method = method
        self.threshold = threshold

    def compute_sign(self, phi: NDArray[np.floating]) -> float:
        """
        Compute the canonical sign for a single eigenvector.

        Args:
            phi: Eigenvector of shape (n,)

        Returns:
            +1 or -1
        """
        if self.method == "max_entry":
            # Sign of entry with maximum absolute value
            idx = np.argmax(np.abs(phi))
            return np.sign(phi[idx]) if np.abs(phi[idx]) > self.threshold else 1.0

        elif self.method == "sum_positive":
            # Make sum non-negative
            s = np.sum(phi)
            return np.sign(s) if np.abs(s) > self.threshold else 1.0

        elif self.method == "first_entry":
            # First entry above threshold is positive
            for val in phi:
                if np.abs(val) > self.threshold:
                    return np.sign(val)
            return 1.0

        elif self.method == "moment":
            # Use third moment (skewness direction)
            # This is invariant to sign flip of the data but sensitive to
            # the asymmetry of the eigenvector distribution
            mean = np.mean(phi)
            centered = phi - mean
            third_moment = np.mean(centered ** 3)
            return np.sign(third_moment) if np.abs(third_moment) > self.threshold else 1.0

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def canonicalize(
        self,
        Phi: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Apply canonical sign convention to all eigenvectors.

        Args:
            Phi: Eigenvector matrix (n, k)

        Returns:
            Phi_canonical: Sign-corrected eigenvectors
            signs: Array of signs applied (k,)
        """
        k = Phi.shape[1]
        signs = np.array([self.compute_sign(Phi[:, j]) for j in range(k)])

        # Flip signs where needed to achieve canonical form
        flip = np.where(signs < 0, -1.0, 1.0)
        Phi_canonical = Phi * flip

        return Phi_canonical, flip

    def align_to_reference(
        self,
        Phi_ref: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Align target signs to match reference convention.

        This uses the canonical form of each as an intermediary,
        avoiding direct node comparison.

        Args:
            Phi_ref: Reference eigenvectors (n, k)
            Phi_target: Target eigenvectors (n, k) - same n required

        Returns:
            Phi_aligned: Target with signs matching reference convention
            signs: Signs applied
        """
        # Get canonical forms
        _, signs_ref = self.canonicalize(Phi_ref)
        _, signs_target = self.canonicalize(Phi_target)

        # To align target to ref:
        # target_canonical = target * signs_target
        # ref_canonical = ref * signs_ref
        # We want target_aligned such that its canonical form matches ref's
        # So: target_aligned * signs_target = target_canonical should have
        #     same sign pattern as ref_canonical = ref * signs_ref

        # The relative sign flip needed
        relative_signs = signs_ref * signs_target

        Phi_aligned = Phi_target * relative_signs

        return Phi_aligned, relative_signs


class SpectralAligner:
    """
    Full spectral alignment pipeline.

    Combines eigenvalue matching and sign conventions to align
    spectral frames without node-to-node comparison.
    """

    def __init__(
        self,
        matcher: Optional[SpectralMatcher] = None,
        sign_convention: Optional[SignConvention] = None,
    ):
        """
        Initialize aligner.

        Args:
            matcher: SpectralMatcher instance
            sign_convention: SignConvention instance
        """
        self.matcher = matcher or SpectralMatcher()
        self.sign_convention = sign_convention or SignConvention()

    def align(
        self,
        Phi_source: NDArray[np.floating],
        lambda_source: NDArray[np.floating],
        Phi_target: NDArray[np.floating],
        lambda_target: NDArray[np.floating],
    ) -> AlignedSpectralPair:
        """
        Align target spectral frame to source.

        Steps:
        1. Match eigenvalues to establish mode correspondence
        2. Permute target eigenvectors accordingly
        3. Apply sign convention for consistency

        Args:
            Phi_source: Source eigenvectors (N_s, k)
            lambda_source: Source eigenvalues (k,)
            Phi_target: Target eigenvectors (N_t, k)
            lambda_target: Target eigenvalues (k,)

        Returns:
            AlignedSpectralPair with aligned target
        """
        k = len(lambda_source)

        if len(lambda_target) != k:
            raise ValueError(
                f"Eigenvalue count mismatch: {k} vs {len(lambda_target)}"
            )

        # Step 1: Match eigenvalues
        permutation, matching_cost = self.matcher.match_with_permutation(
            lambda_source, lambda_target
        )

        # Step 2: Permute target
        Phi_permuted = Phi_target[:, permutation]
        lambda_permuted = lambda_target[permutation]

        # Step 3: Canonicalize signs
        Phi_source_canonical, _ = self.sign_convention.canonicalize(Phi_source)
        Phi_target_canonical, _ = self.sign_convention.canonicalize(Phi_permuted)

        # Compute relative signs
        signs_source = np.array([
            self.sign_convention.compute_sign(Phi_source[:, j])
            for j in range(k)
        ])
        signs_target = np.array([
            self.sign_convention.compute_sign(Phi_permuted[:, j])
            for j in range(k)
        ])

        relative_signs = signs_source * signs_target
        Phi_aligned = Phi_permuted * relative_signs

        # Compute eigenvalue gaps
        eigenvalue_gaps = np.abs(lambda_source - lambda_permuted)

        return AlignedSpectralPair(
            Phi_source=Phi_source,
            Phi_target_aligned=Phi_aligned,
            lambda_source=lambda_source,
            lambda_target_aligned=lambda_permuted,
            eigenvalue_permutation=permutation,
            sign_flips=relative_signs,
            matching_cost=matching_cost,
            eigenvalue_gaps=eigenvalue_gaps,
        )

    def align_sequence(
        self,
        Phi_sequence: List[NDArray[np.floating]],
        lambda_sequence: List[NDArray[np.floating]],
    ) -> List[AlignedSpectralPair]:
        """
        Align a temporal sequence of spectral frames.

        Each frame is aligned to its predecessor, creating a
        chain of consistent representations.

        Args:
            Phi_sequence: List of eigenvector matrices
            lambda_sequence: List of eigenvalue arrays

        Returns:
            List of AlignedSpectralPair for consecutive pairs
        """
        T = len(Phi_sequence)

        if len(lambda_sequence) != T:
            raise ValueError("Phi and lambda sequences must have same length")

        alignments = []

        for t in range(T - 1):
            pair = self.align(
                Phi_sequence[t], lambda_sequence[t],
                Phi_sequence[t + 1], lambda_sequence[t + 1],
            )
            alignments.append(pair)

        return alignments

    def compute_spectral_velocity(
        self,
        alignment: AlignedSpectralPair,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Compute velocity from aligned spectral pair.

        This is the target for flow matching: the direction from
        source to aligned target.

        Args:
            alignment: AlignedSpectralPair

        Returns:
            v_lambda: Eigenvalue velocity (k,)
            v_Phi: Eigenvector velocity (N, k) - tangent vector, or None if sizes differ
        """
        from temporal_spectral_flow.stiefel import StiefelManifold

        # Eigenvalue velocity (simple difference in R^k)
        v_lambda = alignment.lambda_target_aligned - alignment.lambda_source

        # Eigenvector velocity (log map on Stiefel)
        # Only valid if N_source == N_target
        if alignment.Phi_source.shape[0] == alignment.Phi_target_aligned.shape[0]:
            v_Phi = StiefelManifold.log_map(
                alignment.Phi_source,
                alignment.Phi_target_aligned,
            )
        else:
            # Different sizes - can't compute standard velocity
            # Would need transport/interpolation
            v_Phi = None

        return v_lambda, v_Phi
