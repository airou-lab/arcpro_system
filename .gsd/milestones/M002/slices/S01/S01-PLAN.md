# S01: USD World & Robot Sizing Debugging

**Goal:** Diagnose and fix the robot clipping/sizing issues in `arcpro_RL_open_street_sim.usd`.
**Demo:** The car spawns accurately without falling through the floor and its size is visually correct on the track.

## Must-Haves
- Verify the robot's scale relative to the track bounds.
- Confirm ground plane and track collision meshes are active and correct.
- Ensure the initial spawn coordinates `pos=(0.0, 0.0, 0.05)` don't cause instant collision/clipping.

## Tasks

- [ ] **T01: Inspect arcpro_env_cfg.py for scale and spawn issues.** `est:1`
- [ ] **T02: Open or query the USD files to check dimensions and collision shapes.** `est:2`
- [ ] **T03: Run a dummy environment test to visually or programmatically verify robot placement.** `est:1`

## Files Likely Touched
- `arcpro_env_cfg.py`
- `arcpro_robot_cfg.py`
