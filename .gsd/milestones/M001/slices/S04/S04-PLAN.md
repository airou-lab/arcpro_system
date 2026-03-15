# S04: Infrastructure & VRAM Optimization.

**Goal:** Reduce memory footprint to support 128 agents on an RTX 3060.
**Demo:** Monitor `nvidia-smi` during the stress test; ensure VRAM usage stays below 12GB.

## Must-Haves
- No OOM on RTX 3060.

## Tasks

- [x] **T01: Refactor ARCProSceneCfg in arcpro_env_cfg.py to support enable_cameras flag.** `est:1`
- [x] **T02: Add 128x128 God View camera attached to env_0 for visual monitoring.** `est:1`
- [x] **T03: Perform VRAM stress test (128 agents, 1000 steps).** `est:1`

## Files Likely Touched
- `arcpro_env_cfg.py`
