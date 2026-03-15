# S01: Hyperparameter Optimization.

**Goal:** optimize hyperparameters for 128 agents and 4096 batch size.
**Demo:** Agent trains without crashing and learns basic policies.

## Must-Haves
- Stable training.
- 128 agents, 4096 batch configuration applied.

## Tasks

- [x] **T01: Configure 128 agents and 4096 batch size.** `est:1`
- [x] **T02: Run hyperparameter sweep and select best.** `est:2`

## Files Likely Touched
- `train.py`
- `arcpro_env_cfg.py`
