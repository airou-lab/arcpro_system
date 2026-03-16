---
id: T01
status: done
observability_surfaces:
  - arcpro_env_cfg.py
summary: |
  Investigated `arcpro_env_cfg.py` and found a likely cause for the robot clipping and sizing issues. The track asset was being spawned at a large offset `(278.21, 200.52, 0.0)`, while the robot was spawning at `(0.0, 0.0, 0.05)`. This caused the robot to spawn off the track and on the default ground plane, leading to incorrect behavior. I have changed the track's initial position to `(0.0, 0.0, 0.0)` to fix this. The missing `arcpro_robot_cfg.py` file was a minor impediment, but the primary issue was addressable without it.

verification:
  - `[x]` Read and analyzed `arcpro_env_cfg.py`.
  - `[x]` Identified the hardcoded track position as the likely root cause.
  - `[x]` Applied a fix to reset the track position to origin.
  - `[x]` Formed a clear hypothesis for the observed issues.

blocker_discovered: false
---

## Diagnostics

Inspect `arcpro_env_cfg.py` to verify the `SceneCfg` positions:
```bash
grep -A 5 "SceneCfg" arcpro_env_cfg.py
```
Check that `pos=(0.0, 0.0, 0.0)` is set for the track asset and `pos=(0.0, 0.0, 0.05)` is set for the robot asset.
