"""
Stiefel manifold geometry utilities.

The Stiefel manifold St(n, k) is the set of n x k matrices with orthonormal columns:
    St(n, k) = {X in R^{n x k} : X^T X = I_k}

This module provides the geometric operations needed for learning dynamics
on this manifold, including:
- Tangent space projections
- Exponential and logarithmic maps
- Geodesic computations
- Retraction operations
- Parallel transport
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, logm, svd, qr
from scipy.sparse import csr_matrix


class StiefelManifold:
    """
    Geometric operations on the Stiefel manifold St(n, k).

    The Stiefel manifold is equipped with the canonical metric inherited
    from the Euclidean embedding space. This class provides differential
    geometric operations for optimization and dynamics on this space.

    Attributes:
        n: Ambient dimension (number of rows)
        k: Subspace dimension (number of columns)
    """

    def __init__(self, n: int, k: int):
        """
        Initialize Stiefel manifold.

        Args:
            n: Ambient dimension
            k: Subspace dimension (must be <= n)
        """
        if k > n:
            raise ValueError(f"k ({k}) must be <= n ({n})")
        self.n = n
        self.k = k

    @staticmethod
    def is_on_manifold(X: NDArray[np.floating], tol: float = 1e-6) -> bool:
        """
        Check if a matrix lies on the Stiefel manifold.

        Args:
            X: Matrix of shape (n, k)
            tol: Numerical tolerance

        Returns:
            True if X^T X is approximately identity
        """
        k = X.shape[1]
        gram = X.T @ X
        return np.allclose(gram, np.eye(k), atol=tol)

    @staticmethod
    def project_to_manifold(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Project a matrix onto the Stiefel manifold via polar decomposition.

        Args:
            X: Matrix of shape (n, k)

        Returns:
            Closest point on St(n, k) in Frobenius norm
        """
        U, _, Vt = svd(X, full_matrices=False)
        return U @ Vt

    @staticmethod
    def project_to_tangent(
        X: NDArray[np.floating],
        V: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Project a vector onto the tangent space at X.

        The tangent space at X is:
            T_X St(n,k) = {V : X^T V + V^T X = 0} (skew-symmetric)

        Args:
            X: Point on St(n, k)
            V: Arbitrary vector in R^{n x k}

        Returns:
            Projected tangent vector
        """
        # Project to ensure V lies in tangent space
        # V_tan = V - X @ sym(X^T V) where sym(A) = (A + A^T) / 2
        XtV = X.T @ V
        sym_XtV = (XtV + XtV.T) / 2
        return V - X @ sym_XtV

    def random_point(self, rng: Optional[np.random.Generator] = None) -> NDArray[np.floating]:
        """
        Generate a random point on St(n, k).

        Args:
            rng: Random number generator

        Returns:
            Random orthonormal matrix
        """
        if rng is None:
            rng = np.random.default_rng()

        A = rng.standard_normal((self.n, self.k))
        Q, _ = np.linalg.qr(A, mode="reduced")
        return Q

    def random_tangent(
        self,
        X: NDArray[np.floating],
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.floating]:
        """
        Generate a random tangent vector at X.

        Args:
            X: Point on St(n, k)
            rng: Random number generator

        Returns:
            Random tangent vector at X
        """
        if rng is None:
            rng = np.random.default_rng()

        V = rng.standard_normal(X.shape)
        return self.project_to_tangent(X, V)

    @staticmethod
    def exp_map(
        X: NDArray[np.floating],
        V: NDArray[np.floating],
        t: float = 1.0,
    ) -> NDArray[np.floating]:
        """
        Retraction (QR-based) exponential-like map on Stiefel.

        This is a QR retraction: exp_X(tV) := qf(X + tV)
        with a fixed sign convention so results are deterministic.

        Args:
            X: Point on St(n,k)
            V: Tangent vector at X
            t: Step size / time parameter
        """
        X = np.asarray(X, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)

        Q, R = qr(X + float(t) * V, mode="economic")

        # Enforce deterministic sign convention: diag(R) >= 0
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Q * d  # scale columns

        return Q


    @staticmethod
    def log_map(
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> NDArray[np.floating]:
        """
        Retraction-inverse log map for the QR retraction exp_map.

        We construct V such that:
            X + V = Y R
        where R is upper-triangular.
        Then exp_map(X, V) = qf(X + V) = qf(YR) = Y (up to QR sign convention).

        NOTE: This is NOT the exact Riemannian logarithm map.
        It is a consistent inverse for the QR retraction used in exp_map.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # A is k×k
        A = X.T @ Y

        # QR factorization of A gives an upper-triangular R
        Qa, Ra = qr(A, mode="economic")

        # Make R have nonnegative diagonal for deterministic behaviour
        d = np.sign(np.diag(Ra))
        d[d == 0] = 1.0
        Ra = (d[:, None] * Ra)  # flip rows so diag is >= 0

        # Construct V so that X + V = Y @ R
        V = (Y @ Ra) - X

        return V



    @staticmethod
    def geodesic_distance(
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> float:
        """
        Compute geodesic distance between two points on St(n, k).

        Based on the principal angles between the subspaces.
        """
        # SVD of X^T Y gives principal angles
        _, s, _ = svd(X.T @ Y, full_matrices=False)

        # Clamp singular values to valid range
        s = np.clip(s, -1.0, 1.0)

        # Treat numerically-equal values as exact
        s = np.where(1.0 - s < 1e-12, 1.0, s)

        # Principal angles
        angles = np.arccos(s)

        # Geodesic distance (Frobenius metric)
        d = float(np.linalg.norm(angles))

        # Snap near-zero distances to zero
        if abs(d) < 1e-10:
            return 0.0

        return d


    @staticmethod
    def retract_qr(
        X: NDArray[np.floating],
        V: NDArray[np.floating],
        t: float = 1.0,
    ) -> NDArray[np.floating]:
        """
        QR-based retraction: efficient approximation to exp map.

        Args:
            X: Point on St(n, k)
            V: Tangent vector at X
            t: Step size

        Returns:
            Point on St(n, k)
        """
        Y = X + t * V
        Q, R = np.linalg.qr(Y, mode="reduced")


        # Ensure consistent orientation
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs

        return Q

    @staticmethod
    def retract_polar(
        X: NDArray[np.floating],
        V: NDArray[np.floating],
        t: float = 1.0,
    ) -> NDArray[np.floating]:
        """
        Polar retraction: projects X + tV onto St(n, k).

        Args:
            X: Point on St(n, k)
            V: Tangent vector at X
            t: Step size

        Returns:
            Point on St(n, k)
        """
        return StiefelManifold.project_to_manifold(X + t * V)

    @staticmethod
    def parallel_transport(
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        V: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Parallel transport tangent vector V from T_X to T_Y.

        Uses vector transport by projection (approximate parallel transport).

        Args:
            X: Starting point
            Y: Target point
            V: Tangent vector at X

        Returns:
            Transported tangent vector at Y
        """
        return StiefelManifold.project_to_tangent(Y, V)

    @staticmethod
    def inner_product(
        X: NDArray[np.floating],
        V1: NDArray[np.floating],
        V2: NDArray[np.floating],
    ) -> float:
        """
        Canonical inner product on tangent space.

        Args:
            X: Base point (unused for canonical metric)
            V1: First tangent vector
            V2: Second tangent vector

        Returns:
            Inner product value
        """
        return np.sum(V1 * V2)

    @staticmethod
    def norm(X: NDArray[np.floating], V: NDArray[np.floating]) -> float:
        """
        Norm of tangent vector.

        Args:
            X: Base point
            V: Tangent vector

        Returns:
            Norm of V
        """
        return np.sqrt(StiefelManifold.inner_product(X, V, V))


class StiefelInterpolator:
    """
    Interpolation along geodesics on the Stiefel manifold.

    Useful for generating intermediate frames between spectral snapshots.
    """

    def __init__(self, manifold: Optional[StiefelManifold] = None):
        """
        Initialize interpolator.

        Args:
            manifold: StiefelManifold instance (created on-demand if None)
        """
        self.manifold = manifold

    def interpolate(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        t: float,
    ) -> NDArray[np.floating]:
        """
        Geodesic interpolation between X and Y.

        Args:
            X: Starting point on St(n, k)
            Y: Ending point on St(n, k)
            t: Interpolation parameter in [0, 1]

        Returns:
            Point on geodesic: exp_X(t * log_X(Y))
        """
        if t <= 0:
            return X.copy()
        if t >= 1:
            return Y.copy()

        V = StiefelManifold.log_map(X, Y)
        return StiefelManifold.exp_map(X, V, t)

    def interpolate_sequence(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        n_steps: int,
    ) -> list[NDArray[np.floating]]:
        """
        Generate sequence of points along geodesic.

        Args:
            X: Starting point
            Y: Ending point
            n_steps: Number of intermediate points

        Returns:
            List of points including X and Y
        """
        V = StiefelManifold.log_map(X, Y)
        t_values = np.linspace(0, 1, n_steps + 2)

        points = []
        for t in t_values:
            if t == 0:
                points.append(X.copy())
            elif t == 1:
                points.append(Y.copy())
            else:
                points.append(StiefelManifold.exp_map(X, V, t))

        return points
