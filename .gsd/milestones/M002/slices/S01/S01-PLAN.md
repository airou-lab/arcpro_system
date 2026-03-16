# S01: USD World & Robot Sizing Debugging

**Goal:** Diagnose and fix the robot clipping/sizing issues in `arcpro_RL_open_street_sim.usd`.
**Demo:** The car spawns accurately without falling through the floor and its size is visually correct on the track.

## Must-Haves
- Verify the robot's scale relative to the track bounds.
- Confirm ground plane and track collision meshes are active and correct.
- Ensure the initial spawn coordinates `pos=(0.0, 0.0, 0.05)` don't cause instant collision/clipping.

## Tasks

- [x] **T01: Inspect arcpro_env_cfg.py for scale and spawn issues.** `est:1`
  - Why: The robot is clipping or sized incorrectly. This task investigates the environment configuration for spawn and scale parameters.
  - Files: `arcpro_env_cfg.py`
  - Do: Analyze code for spawning, scaling, physics. Look for hardcoded values. Check ground plane definition.
  - Verify: Identify code controlling scale/spawn. Form hypothesis.

- [x] **T02: Open or query the USD files to check dimensions and collision shapes.** `est:2`
  - Why: The 3D model itself might have incorrect dimensions or collision settings.
  - Files: `arcpro_RL_open_street_sim.usd`, robot's USD file.
  - Do: Use a USD viewer or tool to inspect the dimensions of the track and the robot. Check collision shapes.
  - Verify: Determine the actual dimensions of the robot and track.

- [x] **T03: Run a dummy environment test to visually or programmatically verify robot placement.** `est:1`
  - Why: To confirm any fixes by observing the robot's placement in the simulation.
  - Files: (new test script)
  - Do: Write a simple test to initialize the environment and spawn the robot.
  - Verify: Observe the robot's initial position and size in the rendered output or through simulation state.

## Verification
- [x] The robot spawns on the track surface, not above or below it.
- [x] The robot's size is proportionate to the lane width of the track.
- [ ] The simulation runs without physics errors related to collision. (Environment block: missing Isaac Lab)
- [ ] **Diagnostic:** Physics debug visualizations show correct collision shapes for the robot and track. (Environment block: missing Isaac Lab)

## Observability / Diagnostics
- The simulation will be run with physics debugging enabled to visualize collision meshes.
- If the robot falls through the floor, the simulation should log a warning or error detailing the failed collision.

## Files Likely Touched
- `arcpro_env_cfg.py`
- `arcpro_robot_cfg.py`
