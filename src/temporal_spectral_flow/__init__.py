"""
Temporal Spectral Flow with Transport-Consistent Alignment.

A framework for learning dynamical models of how the intrinsic geometric structure
of high-dimensional data evolves over time.

Key concepts:
- Maps temporal snapshots to spectral representations on the Stiefel manifold
- Learns smooth flows representing temporal dynamics
- Uses optimal transport for alignment during training
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "GraphConstructor",
    "SpectralEmbedding",
    "StiefelManifold",
    "TemporalIntrinsicDimension",
    # Spectral alignment
    "SpectralMatcher",
    "SignConvention",
    "SpectralAligner",
    "AlignedSpectralPair",
    # Optimal transport
    "TransportAlignment",
    "BasisAligner",
    # Flow models (require PyTorch)
    "SpectralFlowModel",
    "FlowTrainer",
    "JointSpectralFlow",
    "EigenvalueVelocityField",
    "EigenvectorVelocityField",
]
