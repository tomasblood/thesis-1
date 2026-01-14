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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphConstructor",
    "SpectralEmbedding",
    "StiefelManifold",
    "TransportAlignment",
    "BasisAligner",
    "SpectralFlowModel",
    "TemporalIntrinsicDimension",
    "FlowTrainer",
]
