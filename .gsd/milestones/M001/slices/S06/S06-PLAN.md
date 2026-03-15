# S06: Autonomous Verification (HUD & 10 Lap target).

**Goal:** Prove the trained policy meets the project's autonomous driving targets.
**Demo:** Robot completes 10 laps while HUD displays real-time telemetry.

## Must-Haves
- Vectorized lap counting logic handles wrap-around correctly.
- HUD displays real-time data for at least one agent (Env 0).
- 10 laps completed without crash or off-track reset.
- Max lateral error < 0.3m.

## Tasks

- [ ] **T01: Vectorized Lap Counting in TrackManager.** `est:1`
- [ ] **T02: Isaac Lab 2.0 HUD Overlay (omni.ui).** `est:2`
- [ ] **T03: scripts/verify_policy.py inference script.** `est:1`
- [ ] **T04: 10-lap Autonomous Verification Run.** `est:1`

## Files Likely Touched
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py`
- `scripts/verify_policy.py`
