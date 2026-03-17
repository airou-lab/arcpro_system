# Slice S02: Physical Verification Loop Integration

## Goal
Upgrade `scripts/verify_policy.py` from a mock loop to a real physical verification loop using Isaac Lab.

## Requirements
- **R002**: Physical 10-Lap Verification
    - Use `ARCProEnv` to step physics simulation.
    - Load `rsl_rl` checkpoint.
    - Track lateral error (<0.3m) and count 10 laps.
    - Emit telemetry JSON.
    - Run headlessly.
    - Fail loudly if dependencies (Isaac Lab, rsl_rl) are missing.

## Verification
### Automated
- `scripts/verify_policy.py` should be executable (e.g., via `./isaaclab.sh python scripts/verify_policy.py` or equivalent).
- Unit tests for checkpoint discovery and telemetry parsing.
- Verification script must exit with non-zero code if lateral error exceeds 0.3m.

### Manual
- Run `python3 scripts/verify_policy.py --checkpoint-dir logs/rsl_rl/arcpro_retraining` and verify it attempts to load Isaac Lab.
- (If GPU/Isaac Lab available) Observe the simulation running for 10 laps and completing with a success summary.

## Tasks
- **T01: Dependency & AppLauncher Setup**
    - Why: To ensure the script correctly initializes Isaac Lab and fails with a clear message if dependencies are missing.
    - Files: `scripts/verify_policy.py`
    - Do: Update the script to use `AppLauncher`, add CLI arguments for headless mode, and wrap Isaac Lab imports in try-except blocks.
    - Verify: Run script with `--help` to confirm Isaac Lab initialization logic is triggered.
    - Done when: Script initializes Isaac Sim (headless) and reports missing dependencies if any.

- **T02: Environment & Model Loading**
    - Why: To instantiate the target environment and the trained policy.
    - Files: `scripts/verify_policy.py`, `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env.py`
    - Do: Use `gym.make` to create `ARCProEnv-v0` and implement a policy loading function for `rsl_rl` checkpoints.
    - Verify: Script successfully initializes the environment and loads the `.pt` weights.
    - Done when: Environment is ready and policy network is populated with trained weights.

- **T03: Real Step Loop & Telemetry Integration**
    - Why: To execute the full verification loop and report metrics from the physics engine.
    - Files: `scripts/verify_policy.py`
    - Do: Replace the mock `run_lap` loop with a real `while` loop that steps the environment, polls `TrackManager` for laps and errors, and collects telemetry.
    - Verify: Script runs 10 laps and outputs the required JSON summary.
    - Done when: 10-lap verification is complete with real lateral error enforcement.
