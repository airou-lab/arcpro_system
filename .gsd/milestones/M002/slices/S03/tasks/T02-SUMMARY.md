---
id: T02
title: Implement update(env) pulling from TrackManager and robot velocity.
status: done
observability_surfaces:
  - Console logs for exceptions during telemetry extraction.
---

### Summary
Implemented the `update_env(env)` method in the `ARCProHUD` class to bridge the gap between the `omni.ui` overlay and the Isaac Lab simulation environment. This method extracts real-time telemetry — Lap Count, Speed (m/s), and Lateral Error (m) — for `env_0` by interacting with the robot asset and the `TrackManager` singleton.

### Key Implementation Decisions
- **Environment Integration**: Added `update_env` which accepts a `ManagerBasedRLEnv` instance. It extracts the robot's world position, orientation (quaternion), and linear velocity in the body frame to aompute the HUD's metrics.
- **Metric Extraction**:
  - **Speed**: Directly pulls `root_lin_vel_b[0, 0]` (local X velocity) for `env_0`.
  - **Lateral Error & Lap Count**: Delegates calculation to the `TrackManager` singleton, ensuring the HUD matches the simulation's internal logic for reward calculation and termination.
- **Robustness**: Encapsulated the logic in `try-except` blocks to prevent simulation crashes if assets are missing or if `TrackManager` hasn't been initialized for all environments yet.
- **Unit Testing with Mocks**: Expanded `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/tests/test_hud.py` to include a `test_update_env` case that mocks a full Isaac Lab environment and verifies correct extraction of values.

### Diagnostics
- **Live Value Verification**: Add `print()` statements inside the `update_env` method in `hud.py` to log the raw `speed`, `lat_error`, and `lap_count` values to the console each frame. This helps verify the data being sent to the UI labels.
- **Exception Logging**: Any error during telemetry extraction (e.g., `TrackManager` not ready, robot asset missing) is caught and logged to the console, pointing to the source of the data issue without crashing the simulator.

### Verification Results
- **Unit Tests**:
  - [x] `test_init_without_omni`: Passes safely.
  - [x] `test_init_with_mock_omni`: Verifies UI label text assignment.
  - [x] `test_update_env`: Verifies telemetry extraction from a mocked environment and `TrackManager`.
- **Command**: `PYTHONPATH=. /home/arika/IsaacLab/isaaclab.sh -p src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/tests/test_hud.py`
- **Result**: `Ran 3 tests in 0.771s OK`.

### Next Steps
- Hook the HUD into the `ARCProEnv` class or its configuration in T03 to activate it during real simulation runs.
