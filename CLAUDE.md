# Global Research Standards

## Why This Exists
Code should read like the paper it implements. Mistakes become documentation — fix errors, add rules here.

**Note:** Claude may ignore this file if contents aren't relevant. Keep universally applicable. Task-specific docs go in `agent_docs/`.

---

## Gatekeeper Rules (ABSOLUTE)
- **NEVER** commit `.env`, credentials, API keys, tokens
- **NEVER** hardcode secrets in source files or logs
- **NEVER** commit without verifying `.gitignore` includes `.env`
- **NEVER** publish sensitive data to git/npm/docker

---

## Identity
- GitHub: `tomasblood` (SSH: `git@github.com:tomasblood/thesis-1.git`)
- Overleaf: [your-project-link]
- Sync config: `cd ~/dotfiles && stow claude`

---

## Memory Hierarchy
| Level | Location | Purpose |
|-------|----------|---------|
| Global | `~/.claude/CLAUDE.md` | This file — all projects |
| Project | `./CLAUDE.md` | Team-shared project rules |
| Local | `./CLAUDE.local.md` | Personal overrides |

---

## Philosophy: Code ≈ Paper
- **Clarity > Abstraction** — modify any module in minutes, not hours
- **Math-First** — variables match paper notation (`x_t`, `sigma`, `alpha`), not (`noisy_data`, `std_dev`)
- **Divergence = Bug** — if implementation doesn't match the equation, it's wrong

---

## Tech Stack
| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ (type hints required) |
| Package Manager | `uv` |
| Deep Learning | PyTorch 2.x |
| Tensor Ops | `einops` (mandatory) |
| Type Checking | `beartype` + `jaxtyping` |
| Acceleration | `accelerate` |
| Logging | `loguru` |
| Config | YAML + dataclasses |
| CLI | `fire` |

---

## The "Don't" List
- ❌ `x.view()`, `x.permute()`, `x.transpose()` — use `einops`
- ❌ Vanilla `LayerNorm` — use `RMSNorm`
- ❌ Training by epochs — use steps
- ❌ Missing `@torch.no_grad()` on inference
- ❌ Hardcoded device — use `accelerate`
- ❌ Precision drift in sampling — explicit `dtype`
- ❌ `print()` for logging — use `loguru`

---

## Defense in Depth
| Layer | Mechanism |
|-------|-----------|
| 1 | CLAUDE.md rules (behavioral) |
| 2 | `settings.json` deny list (access) |
| 3 | `.gitignore` (git safety) |
| 4 | Hooks (deterministic) |

---

## Workflow

### Single-Purpose Chats
39% performance degradation when mixing topics. Stay focused.

| Situation | Action |
|-----------|--------|
| New feature | New chat |
| Unrelated bug | `/clear`, new task |
| Research vs implementation | Separate chats |
| 20+ turns | Start fresh |

### Before Commit
1. Run linter: `uv run ruff check .`
2. Run tests: `python tests/test_model.py`
3. Verify no secrets staged

---

## Project Scaffolding
When creating new projects:

```
project/
├── pyproject.toml
├── config.yaml
├── src/project_name/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── train.py
├── sample.py
├── tests/
├── figures/              # Overleaf exports
├── agent_docs/           # Progressive disclosure
│   ├── patterns.md       # Code patterns (RMSNorm, GEGLU, etc.)
│   ├── architecture.md   # Project-specific design
│   └── experiments.md    # Experiment notes
├── .env.example
├── .gitignore
└── CLAUDE.md             # Project-specific
```

---

## Progressive Disclosure

**Don't put code snippets in CLAUDE.md.** Point to where they live:

| Topic | Location |
|-------|----------|
| Code patterns (RMSNorm, GEGLU, Fourier) | `agent_docs/patterns.md` |
| Flow matching / OT math | `agent_docs/math_specs.md` |
| Architecture details | `agent_docs/architecture.md` |
| Overleaf workflow | `agent_docs/overleaf.md` |
| Experiment configs | `agent_docs/experiments.md` |

Claude reads these **only when relevant** to the current task.

---

## Testing Philosophy
Shape + gradient checks via `assert`. No pytest ceremony.

Run: `python tests/test_model.py`

---

## Overleaf Integration
- Figures: `figures/` folder, PDF + PNG exports
- Naming: `method_overview.pdf`, `loss_curves.pdf`, `ablation_*.pdf`
- Sync: Git integration or manual
- Citations in code: `# From [Author] ([Year]) Eq. [N]`

Details in `agent_docs/overleaf.md`.

---

## Finding Information
| Topic | Source |
|-------|--------|
| einops | einops.aimo.ai |
| Reference implementations | lucidrains GitHub |
| Project patterns | `agent_docs/patterns.md` |
| Skills | `~/.claude/skills/` (check what's installed) |

---

*Research code that reads like the paper it implements.*

*Last updated: [DATE]*
