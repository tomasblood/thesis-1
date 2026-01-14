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

from temporal_spectral_flow.graph import GraphConstructor
from temporal_spectral_flow.spectral import SpectralEmbedding
from temporal_spectral_flow.stiefel import StiefelManifold
from temporal_spectral_flow.transport import TransportAlignment
from temporal_spectral_flow.flow import SpectralFlowModel
from temporal_spectral_flow.tid import TemporalIntrinsicDimension
from temporal_spectral_flow.training import FlowTrainer

__version__ = "0.1.0"

__all__ = [
    "GraphConstructor",
    "SpectralEmbedding",
    "StiefelManifold",
    "TransportAlignment",
    "SpectralFlowModel",
    "TemporalIntrinsicDimension",
    "FlowTrainer",
]
