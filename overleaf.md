# Overleaf Integration

Workflow for syncing code outputs to thesis.

---

## Figure Export

```python
import matplotlib.pyplot as plt

def save_figure(fig, name: str, folder: str = "figures"):
    """Save figure for Overleaf. PDF for vectors, PNG for rasters."""
    path = f"{folder}/{name}"
    fig.savefig(f"{path}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(f"{path}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
```

---

## Figure Naming Convention

```
figures/
├── method_overview.pdf        # Architecture diagrams
├── loss_curves.pdf            # Training plots
├── ablation_depth.pdf         # Ablation: varying depth
├── ablation_lr.pdf            # Ablation: learning rate
├── comparison_baseline.pdf    # vs baselines
├── qualitative_samples.pdf    # Generated samples
└── tsne_embeddings.pdf        # Embedding visualizations
```

---

## LaTeX Reference

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/method_overview.pdf}
    \caption{Overview of the proposed method.}
    \label{fig:method}
\end{figure}
```

For subfigures:
```latex
\begin{figure}[t]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/ablation_depth.pdf}
        \caption{Varying depth}
        \label{fig:ablation-depth}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/ablation_lr.pdf}
        \caption{Varying learning rate}
        \label{fig:ablation-lr}
    \end{subfigure}
    \caption{Ablation studies.}
    \label{fig:ablations}
\end{figure}
```

---

## Syncing Options

### Option 1: Git Integration (Overleaf Premium)
1. Link Overleaf project to GitHub repo
2. Push figures to `figures/` branch or folder
3. Pull in Overleaf

### Option 2: Manual Sync
1. Export figures locally
2. Upload to Overleaf via web interface

### Option 3: Dropbox/Google Drive
1. Save figures to synced folder
2. Link in Overleaf

---

## Paper Citations in Code

Reference equations directly:

```python
# From Lipman et al. (2023) Eq. 7
v_t = (x_1 - x_t) / (1 - t)

# From Chen & Lipman (2024) Theorem 3.2
# Optimal transport map preserves geodesic structure
```

---

## Results Tables

Export as CSV, format in LaTeX:

```python
import pandas as pd

results = pd.DataFrame({
    'Method': ['Ours', 'Baseline A', 'Baseline B'],
    'Accuracy': [0.95, 0.89, 0.87],
    'FID': [12.3, 18.7, 22.1],
})
results.to_csv('figures/results.csv', index=False)
```

```latex
\begin{table}[t]
    \centering
    \begin{tabular}{lcc}
        \toprule
        Method & Accuracy $\uparrow$ & FID $\downarrow$ \\
        \midrule
        Ours & \textbf{0.95} & \textbf{12.3} \\
        Baseline A & 0.89 & 18.7 \\
        Baseline B & 0.87 & 22.1 \\
        \bottomrule
    \end{tabular}
    \caption{Quantitative comparison.}
    \label{tab:results}
\end{table}
```

---

## Matplotlib Style for Papers

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': (3.5, 2.5),  # Single column width
    'figure.dpi': 300,
})
```
