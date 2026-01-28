# Spectral Alignment Visualization Redesign

**Date:** 2026-01-23
**Status:** Approved
**Goal:** Replace static `visualize_toy_examples.py` with an interactive Streamlit app demonstrating temporal spectral alignment

---

## Overview

The current `visualize_toy_examples.py` generates static PNGs showing transport-based alignment concepts. This redesign creates a multi-page Streamlit app that demonstrates how spectral embeddings (Phi, lambda) evolve over time and how eigenvalue matching + sign conventions keep them aligned.

### Key Features
- Time slider to scrub through temporal evolution
- Three synchronized panels: eigenvalue trajectories, eigenvector heatmap, 2D spectral embedding
- Two toy examples: merging clusters and evolving graph connectivity
- Shows "before vs after" alignment comparison

---

## Architecture

### File Structure

```
src/viz/
├── __init__.py
├── styles.py              # (existing) Color palettes, styling
├── static.py              # (existing) Static matplotlib helpers
├── app.py                 # (existing) Flow matching demo - keep as-is
├── spectral_app.py        # NEW: Entry point for spectral demos
├── components/
│   ├── __init__.py
│   ├── eigenvalue_panel.py    # Eigenvalue trajectory plot
│   ├── eigenvector_panel.py   # Eigenvector heatmap with alignment
│   └── embedding_panel.py     # 2D spectral embedding scatter
└── pages/
    ├── 1_merging_clusters.py  # Toy example: clusters merge/split
    └── 2_evolving_graph.py    # Toy example: graph connectivity
```

### Entry Point

Run with: `streamlit run src/viz/spectral_app.py`

---

## Panel Components

### 1. Eigenvalue Panel (`eigenvalue_panel.py`)

Shows lambda_1, lambda_2, ..., lambda_k as trajectories over full time range.

**Features:**
- Lines colored by mode index (consistent after alignment)
- Vertical line indicating current time
- Highlights eigenvalue crossings with markers
- Optional "before alignment" faded background
- Y-axis: eigenvalue magnitude, X-axis: time

**Function signature:**
```python
def plot_eigenvalue_trajectories(
    lambda_sequence: List[NDArray],        # Raw eigenvalues per timestep
    lambda_aligned: List[NDArray],         # Aligned eigenvalues
    current_time: float,                   # Current timestep [0, 1]
    show_raw: bool = False,                # Show unaligned as background
    style: VizStyle = None,
) -> plt.Figure:
```

### 2. Eigenvector Panel (`eigenvector_panel.py`)

Heatmap of Phi_t (nodes x spectral dimensions) at current time.

**Features:**
- Rows = nodes, columns = eigenvector index
- RdBu diverging colormap (negative blue, positive red)
- Optional side-by-side: raw Phi_t vs aligned Phi_t
- Sign flips visible as column color inversions

**Function signature:**
```python
def plot_eigenvector_heatmap(
    Phi: NDArray,                          # Eigenvectors at current time (n, k)
    Phi_aligned: NDArray,                  # Aligned eigenvectors
    show_comparison: bool = False,         # Side-by-side view
    style: VizStyle = None,
) -> plt.Figure:
```

### 3. Embedding Panel (`embedding_panel.py`)

2D scatter using first two eigenvectors as coordinates.

**Features:**
- Points colored by original cluster/component membership
- Ghost trails showing previous positions
- Configurable trail length
- Optional arrows showing spectral velocity direction

**Function signature:**
```python
def plot_spectral_embedding(
    Phi_sequence: List[NDArray],           # All timesteps for trails
    current_idx: int,                      # Current timestep index
    labels: NDArray,                       # Cluster/component labels for coloring
    trail_length: int = 5,                 # Number of past positions to show
    show_velocity: bool = False,           # Show velocity arrows
    style: VizStyle = None,
) -> plt.Figure:
```

---

## Toy Examples

### Example 1: Merging Clusters (`1_merging_clusters.py`)

Two Gaussian clusters gradually move toward each other and merge.

**Visual progression:**
```
t=0.0          t=0.5          t=1.0
  ***    ***     ***  ***       ******
  ***    ***      ******        ******
  ***    ***       ****         ******
[separated]    [touching]     [merged]
```

**Spectral signature:**
- t=0: Two clusters -> lambda_2 near 0 (near-disconnected)
- As clusters merge, lambda_2 increases (gap closes)
- Eigenvectors transition from "localized on each cluster" to "global modes"

**Parameters:**
- `n_points`: Number of points per cluster (default: 50)
- `cluster_std`: Gaussian standard deviation (default: 0.3)
- `separation_start`: Initial cluster distance (default: 4.0)
- `separation_end`: Final cluster distance (default: 0.0)
- `k_neighbors`: For k-NN graph construction (default: 10)
- `n_eigenvectors`: Number of spectral dimensions (default: 6)
- `n_timesteps`: Temporal resolution (default: 50)

**Data generator:**
```python
def generate_merging_clusters(
    n_points: int = 50,
    cluster_std: float = 0.3,
    separation_start: float = 4.0,
    separation_end: float = 0.0,
    k_neighbors: int = 10,
    n_eigenvectors: int = 6,
    n_timesteps: int = 50,
    seed: int = 42,
) -> Tuple[List[SpectralSnapshot], NDArray]:
    """
    Returns:
        snapshots: List of SpectralSnapshot (Phi, eigenvalues) per timestep
        labels: Cluster membership array for coloring
    """
```

### Example 2: Evolving Graph (`2_evolving_graph.py`)

Two ring graphs gradually connect via edges of increasing weight.

**Visual progression:**
```
t=0.0              t=0.5              t=1.0
 o-o               o-o                o-o
o   o   o-o      o   o---o-o        o   o===o-o
 o-o     o-o      o-o     o-o        o-o     o-o
[disconnected]   [weakly coupled]   [strongly coupled]
```

**Spectral signature:**
- Disconnected: Two copies of ring spectrum (degenerate eigenvalues)
- Coupling creates eigenvalue crossings as modes "see" each other
- New global modes emerge that span both rings
- Clear eigenvalue crossings when ring sizes differ

**Parameters:**
- `n_nodes_ring1`: Nodes in first ring (default: 30)
- `n_nodes_ring2`: Nodes in second ring (default: 20)
- `coupling_start`: Initial coupling weight (default: 0.0)
- `coupling_end`: Final coupling weight (default: 1.0)
- `coupling_schedule`: "linear", "exponential", or "step" (default: "linear")
- `n_eigenvectors`: Number of spectral dimensions (default: 6)
- `n_timesteps`: Temporal resolution (default: 50)

**Data generator:**
```python
def generate_evolving_graph(
    n_nodes_ring1: int = 30,
    n_nodes_ring2: int = 20,
    coupling_start: float = 0.0,
    coupling_end: float = 1.0,
    coupling_schedule: str = "linear",
    n_eigenvectors: int = 6,
    n_timesteps: int = 50,
) -> Tuple[List[SpectralSnapshot], NDArray]:
    """
    Returns:
        snapshots: List of SpectralSnapshot (Phi, eigenvalues) per timestep
        labels: Ring membership array for coloring
    """
```

---

## Page Layout

Each page follows the same structure:

```
+------------------------------------------------------------------+
|  [Logo]  Temporal Spectral Alignment: [Example Name]             |
+------------------------------------------------------------------+
|  <--------------------- Time Slider ----------------------->     |
|  t = 0.35            [> Play]  [|< Reset]    Speed: [1x v]       |
+------------------------------------------------------------------+
|                                                                  |
|  +----------------+  +----------------+  +----------------+      |
|  |   Eigenvalue   |  |   Eigenvector  |  |    Spectral    |      |
|  |  Trajectories  |  |    Heatmap     |  |   Embedding    |      |
|  |                |  |                |  |                |      |
|  |   lambda_1-k   |  |  Phi_t matrix  |  |   2D scatter   |      |
|  |      |         |  |    (n x k)     |  |   with trails  |      |
|  +----------------+  +----------------+  +----------------+      |
|                                                                  |
+--------------------------- Sidebar ------------------------------+
|  Parameters:                                                     |
|  +- n_points: [100]        +- Show alignment: [x]               |
|  +- k_neighbors: [10]      +- Show raw comparison: [ ]          |
|  +- n_eigenvectors: [6]    +- Trail length: [5]                 |
|  +- [Regenerate]           +- Colormap: [RdBu v]                |
+------------------------------------------------------------------+
```

---

## Session State

```python
# Cached in st.session_state
{
    "spectral_sequence": List[SpectralSnapshot],  # Raw snapshots
    "aligned_pairs": List[AlignedSpectralPair],   # From align_sequence()
    "labels": NDArray,                             # Cluster/component labels
    "current_time": float,                         # In [0, 1]
    "playing": bool,                               # Animation state
    "params_hash": str,                            # For cache invalidation
}
```

**State transitions:**
1. Parameter change -> Regenerate sequence, recompute alignments, reset time
2. Time slider move -> Update panels, no recomputation
3. Play button -> Auto-increment time with st.rerun() loop
4. Reset button -> Set time to 0, stop animation

---

## Integration with Existing Code

Uses these existing modules:
- `temporal_spectral_flow.alignment.SpectralAligner` - align_sequence()
- `temporal_spectral_flow.alignment.SpectralMatcher` - eigenvalue matching
- `temporal_spectral_flow.alignment.SignConvention` - sign consistency
- `temporal_spectral_flow.graph.GraphConstructor` - k-NN graph building
- `temporal_spectral_flow.spectral.SpectralEmbedding` - eigendecomposition
- `viz.styles.ColorPalette, VizStyle` - consistent styling

---

## Migration

### Files to Delete
- `visualize_toy_examples.py` - replaced by Streamlit app
- `viz_1_sign_flip.png` - static output
- `viz_2_rotation.png` - static output
- `viz_3_birth_of_mode.png` - static output
- `viz_4_noise_vs_signal.png` - static output
- `viz_5_permutation.png` - static output
- `viz_6_curvature_change.png` - static output
- `viz_summary.png` - static output

### Files to Create
1. `src/viz/spectral_app.py`
2. `src/viz/components/__init__.py`
3. `src/viz/components/eigenvalue_panel.py`
4. `src/viz/components/eigenvector_panel.py`
5. `src/viz/components/embedding_panel.py`
6. `src/viz/pages/1_merging_clusters.py`
7. `src/viz/pages/2_evolving_graph.py`

### Files to Keep
- `src/viz/app.py` - existing flow matching demo (separate purpose)
- `src/viz/styles.py` - reused for consistent styling
- `src/viz/static.py` - utility functions may be useful
- `tests/test_toy_examples.py` - update if needed

---

## Testing

### Component Testing
Each component module includes `if __name__ == "__main__"` block:
```python
if __name__ == "__main__":
    # Generate mock data
    # Call plotting function
    # plt.show() for visual verification
```

### Integration Testing
```bash
# Run the app and manually verify:
streamlit run src/viz/spectral_app.py

# Check:
# - Time slider updates all panels
# - Play/pause animation works
# - Parameter changes regenerate data
# - Both example pages load correctly
```

---

## Implementation Order

1. Create `components/` package with three panel functions
2. Create data generators for both toy examples
3. Create `spectral_app.py` entry point with home page
4. Create `pages/1_merging_clusters.py` with full interactivity
5. Create `pages/2_evolving_graph.py` with full interactivity
6. Delete old `visualize_toy_examples.py` and PNG outputs
7. Update any references in tests or documentation
