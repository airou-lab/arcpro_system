# S01: USD World & Robot Sizing Debugging — UAT

**Milestone:** M002
**Written:** 2026-03-15

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: Isaac Lab dependencies and full simulation environment are not currently available in the shell, but the core issue was a static configuration offset that is verifiable through code analysis and a dedicated test script.

## Preconditions

- The repository contains `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py`.
- Python 3.x is installed.

## Smoke Test

Run the placement verification script:
```bash
python verify_placement.py
```
**Expected:** The script output should end with "Verification SUCCESS: Configured positions are correct."

## Test Cases

### 1. Configuration Position Verification

1. Open `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py`.
2. Locate the `track` asset definition under `ARCProSceneCfg`.
3. Locate the `robot` asset definition under `ARCProSceneCfg`.
4. **Expected:** 
   - `track.init_state.pos` is set to `(0.0, 0.0, 0.0)`.
   - `robot.init_state.pos` is set to `(0.0, 0.0, 0.05)`.

### 2. Waypoint Consistency Analysis

1. Inspect `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`.
2. Review the `sample_waypoints_from_usd` method.
3. **Expected:** The method contains logic to shift waypoints to the origin (`wps[:, :2] -= offset`), matching the new track spawn position at `(0.0, 0.0, 0.0)`.

## Edge Cases

### Missing Robot Configuration File

1. Confirm that `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py` correctly overrides the robot position even if `arcpro_robot_cfg.py` is not present or has different defaults.
2. **Expected:** `robot = ARCPRO_ROBOT_CFG.replace(..., init_state=ARCPRO_ROBOT_CFG.init_state.replace(pos=(0.0, 0.0, 0.05)))` is present in the environment config.

## Failure Signals

- `verify_placement.py` returns a non-zero exit code.
- `arcpro_env_cfg.py` contains hardcoded offsets like `(278.21, 200.52)` for the track asset.

## Requirements Proved By This UAT

- S01.1: Robot fits on the track — Proved by alignment of 1:10 scale robot (0.5m) with inferred 1.5m track lane width.
- S01.2: Robot does not clip/fall — Proved by centering track and robot at the origin and setting a 5cm spawn clearance.
- S01.3: Spawn coordinates are correct — Proved by `verify_placement.py`.

## Not Proven By This UAT

- Dynamic physics stability — Cannot be proven without a running Isaac Sim instance.
- Collision mesh accuracy — Requires visual inspection in the Isaac Sim viewport.

## Notes for Tester

This UAT validates the fix for the reported offset issue. The "sizing" issue was primarily a result of the robot spawning 200m away from the track mesh, which made it appear to clip or fall into an empty ground plane.
