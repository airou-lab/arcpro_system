# S03: Isaac Lab 2.0 HUD Overlay (omni.ui)

**Goal:** Create an `omni.ui` overlay for simulation telemetry.
**Demo:** Viewport shows Lap Count, Speed, and Lateral Error for `env_0`.

## Must-Haves
- `omni.ui.Window` overlay without scrollbars or title bars.
- Updates cleanly on environment steps without dropping FPS.

## Tasks

- [ ] **T01: Create mdp/hud.py with ARCProHUD class.** `est:1`
- [ ] **T02: Implement update(env) pulling from TrackManager and robot velocity.** `est:2`
- [ ] **T03: Hook HUD initialization into ARCProEnvCfg if enabled.** `est:1`

## Files Likely Touched
- `mdp/hud.py`
- `arcpro_env_cfg.py`
