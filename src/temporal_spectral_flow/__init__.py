"""
Temporal Spectral Flow for Spectral Dynamics.

A framework for learning continuous-time vector fields that model how spectral
representations evolve over time on the Stiefel manifold.

Key concepts:
- Maps temporal snapshots to spectral representations (Phi, lambda) on St(N,k) x R^k
- Learns smooth flows via endpoint prediction with Grassmann-invariant loss
- NO alignment during training - uses raw consecutive pairs
- Integrates learned field with QR retraction for manifold-aware dynamics
- Computes Temporal Intrinsic Dimension (TID) to identify stable modes
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import for optional dependencies."""
    # Core modules (always available)
    if name == "GraphConstructor":
        from temporal_spectral_flow.graph import GraphConstructor
        return GraphConstructor
    elif name == "SpectralEmbedding":
        from temporal_spectral_flow.spectral import SpectralEmbedding
        return SpectralEmbedding
    elif name == "StiefelManifold":
        from temporal_spectral_flow.stiefel import StiefelManifold
        return StiefelManifold
    elif name == "TemporalIntrinsicDimension":
        from temporal_spectral_flow.tid import TemporalIntrinsicDimension
        return TemporalIntrinsicDimension

    # Core: Spectral alignment (NumPy/SciPy only)
    elif name == "SpectralMatcher":
        from temporal_spectral_flow.alignment import SpectralMatcher
        return SpectralMatcher
    elif name == "SignConvention":
        from temporal_spectral_flow.alignment import SignConvention
        return SignConvention
    elif name == "SpectralAligner":
        from temporal_spectral_flow.alignment import SpectralAligner
        return SpectralAligner
    elif name == "AlignedSpectralPair":
        from temporal_spectral_flow.alignment import AlignedSpectralPair
        return AlignedSpectralPair

    # Optional: Requires POT
    elif name == "TransportAlignment":
        try:
            from temporal_spectral_flow.transport import TransportAlignment
            return TransportAlignment
        except ImportError as e:
            raise ImportError(
                f"TransportAlignment requires the 'pot' package. "
                f"Install with: pip install pot\nOriginal error: {e}"
            ) from e
    elif name == "BasisAligner":
        from temporal_spectral_flow.transport import BasisAligner
        return BasisAligner

    # Optional: Requires PyTorch
    elif name == "SpectralFlowModel":
        try:
            from temporal_spectral_flow.flow import SpectralFlowModel
            return SpectralFlowModel
        except ImportError as e:
            raise ImportError(
                f"SpectralFlowModel requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "FlowTrainer":
        try:
            from temporal_spectral_flow.training import FlowTrainer
            return FlowTrainer
        except ImportError as e:
            raise ImportError(
                f"FlowTrainer requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e

    # Optional: Joint spectral flow (requires PyTorch)
    elif name == "JointSpectralFlow":
        try:
            from temporal_spectral_flow.joint_flow import JointSpectralFlow
            return JointSpectralFlow
        except ImportError as e:
            raise ImportError(
                f"JointSpectralFlow requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "EigenvalueVelocityField":
        try:
            from temporal_spectral_flow.joint_flow import EigenvalueVelocityField
            return EigenvalueVelocityField
        except ImportError as e:
            raise ImportError(
                f"EigenvalueVelocityField requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "EigenvectorVelocityField":
        try:
            from temporal_spectral_flow.joint_flow import EigenvectorVelocityField
            return EigenvectorVelocityField
        except ImportError as e:
            raise ImportError(
                f"EigenvectorVelocityField requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e

    # Grassmann-invariant losses (require PyTorch)
    elif name == "principal_angle_loss":
        try:
            from temporal_spectral_flow.losses import principal_angle_loss
            return principal_angle_loss
        except ImportError as e:
            raise ImportError(
                f"principal_angle_loss requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "projection_loss":
        try:
            from temporal_spectral_flow.losses import projection_loss
            return projection_loss
        except ImportError as e:
            raise ImportError(
                f"projection_loss requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "eigenvalue_mse_loss":
        try:
            from temporal_spectral_flow.losses import eigenvalue_mse_loss
            return eigenvalue_mse_loss
        except ImportError as e:
            raise ImportError(
                f"eigenvalue_mse_loss requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    elif name == "TrainingConfig":
        try:
            from temporal_spectral_flow.training import TrainingConfig
            return TrainingConfig
        except ImportError as e:
            raise ImportError(
                f"TrainingConfig requires PyTorch. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "GraphConstructor",
    "SpectralEmbedding",
    "StiefelManifold",
    "TemporalIntrinsicDimension",
    # Spectral alignment (for analysis only, NOT training)
    "SpectralMatcher",
    "SignConvention",
    "SpectralAligner",
    "AlignedSpectralPair",
    # Optimal transport (for analysis only, NOT training)
    "TransportAlignment",
    "BasisAligner",
    # Flow models (require PyTorch)
    "SpectralFlowModel",
    "FlowTrainer",
    "TrainingConfig",
    "JointSpectralFlow",
    "EigenvalueVelocityField",
    "EigenvectorVelocityField",
    # Grassmann-invariant losses (require PyTorch)
    "principal_angle_loss",
    "projection_loss",
    "eigenvalue_mse_loss",
]
