# Architecture

## Overview

Temporal Spectral Flow learns how geometric structures evolve over time by:

1. Computing spectral embeddings at each timestep
2. Aligning embeddings via optimal transport (inner alignment oracle)
3. Training a flow model to predict dynamics (outer flow)

## Module Structure

```
src/temporal_spectral_flow/
├── spectral.py      # Spectral embedding computation
├── graph.py         # Graph construction, Laplacian
├── stiefel.py       # Stiefel manifold geometry
├── transport.py     # Optimal transport alignment
├── alignment.py     # Spectral alignment utilities
├── flow.py          # Neural velocity field
├── joint_flow.py    # Joint spectral flow model
├── training.py      # Training pipeline
└── tid.py           # Temporal Intrinsic Dimension
```

## Key Classes

### SpectralFlowModel
Main model that combines velocity field with integration.

### StiefelVelocityField
Neural network predicting tangent vectors on St(N, k).

### TransportAlignment
Inner alignment oracle using optimal transport.

### FlowTrainer
Training loop with velocity matching and endpoint losses.

## Design Principles

1. **Manifold-aware**: All operations respect Stiefel geometry
2. **Transport-consistent**: Alignment preserves mass
3. **Decoupled**: Inner alignment separate from outer flow
