# Experiments

Track experiment configurations and results here.

## Experiment Template

```yaml
name: experiment_name
date: YYYY-MM-DD
config:
  k: 10
  n_epochs: 100
  learning_rate: 1e-4
  batch_size: 32
results:
  final_loss: 0.0
  notes: ""
```

## Planned Experiments

- [ ] Baseline: static spectral embedding
- [ ] Temporal flow on synthetic data
- [ ] Real dataset evaluation
- [ ] Ablation: alignment methods
- [ ] Ablation: integration methods
