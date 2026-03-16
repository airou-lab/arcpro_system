# T02: Implement update(env) pulling from TrackManager and robot velocity

## Goal
Update the `ARCProHUD` class to extract telemetry data directly from the Isaac Lab environment object.

## Proposed Changes

### `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`
- Add `update_env(self, env: ManagerBasedRLEnv)` method.
- This method will:
    1. Access the robot asset (defaulting to name "robot").
    2. Extract the robot's world position and local velocity.
    3. Calculate the robot's yaw from its orientation quaternion.
    4. Use `TrackManager` to get the current lap count and lateral error for `env_0`.
    5. Call the existing `update` method with these values.
- Add error handling to ensure HUD updates don't crash the simulation if data is missing.

## Verification Plan

### Automated Tests
- Update `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/tests/test_hud.py`.
- Add a test case that mocks a `ManagerBasedRLEnv` and its assets to verify `update_env` pulls the correct values.
- Use `unittest.mock` to simulate `TrackManager` and environment data.

### Manual Verification
- This will be fully verified in T03 when the HUD is integrated into the environment.
- For T02, successful unit tests with mocked environment state are the primary verification.
