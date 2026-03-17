# Slice S02: Physical Verification Loop Integration - UAT

## Preconditions
1. The project structure `src/examples/ARCPro_RL/` must be present.
2. `logs/rsl_rl/arcpro_retraining/` must contain at least one `.pt` checkpoint file.
3. Isaac Lab environment must be available (for a full successful run).
4. For automated logic tests, a standard Python 3.10+ environment is sufficient.

## Test Cases

### TC-01: Dependency Failure (No Isaac Lab)
**Goal**: Verify the script fails loudly when run outside the Isaac Lab environment.
1. **Steps**:
    - Run `python3 scripts/verify_policy.py` from a standard shell without Isaac Lab.
2. **Expected Outcome**:
    - The script prints a JSON error object with `"status": "error"`.
    - The error message mentions "Isaac Lab not found" or "torch not found".
    - Exit code is non-zero.

### TC-02: Checkpoint Discovery Logic
**Goal**: Verify that the latest model checkpoint is correctly identified.
1. **Steps**:
    - Run `python3 tests/test_verify_policy_logic.py`.
2. **Expected Outcome**:
    - The test script prints "Checkpoint discovery tests passed!".
    - All assertions for finding the highest numbered `model_*.pt` or the most recent modification time pass.

### TC-03: Real Verification Run (Requires Isaac Lab)
**Goal**: Verify 10-lap completion with lateral error enforcement.
1. **Steps**:
    - Run `./isaaclab.sh python scripts/verify_policy.py --checkpoint-dir logs/rsl_rl/arcpro_retraining --laps 2 --headless`.
2. **Expected Outcome**:
    - Isaac Sim initializes (headless).
    - Environment `ARCPro-v0` is instantiated.
    - Policy weights are loaded from `logs/rsl_rl/arcpro_retraining/model_100.pt`.
    - Script steps the simulation and prints lap completion summaries (time, lateral error).
    - After 2 laps, it prints a success JSON summary.
    - If lateral error exceeds 0.3m, the script exits early with a failure JSON object.

### TC-04: Help and Argument Parsing
**Goal**: Verify CLI options are integrated correctly with AppLauncher.
1. **Steps**:
    - Run `python3 scripts/verify_policy.py --help`.
2. **Expected Outcome**:
    - The output displays usage for `--checkpoint-dir`, `--laps`, and Isaac Lab standard arguments (e.g., `--headless`, `--gpu`).
    - Note: This might still trigger a dependency error for `torch` or `isaaclab`, but the argparse setup is verified to be present.

## Edge Cases
- **Missing Checkpoint Directory**: Running with a non-existent `--checkpoint-dir` should return a "No policy checkpoint found" error JSON.
- **Corrupt Checkpoint**: A non-torch `.pt` file should trigger a load error JSON.
- **Robot Crash**: If the robot leaves the track (terminated/truncated in Isaac Lab), the script should report "Robot crashed" and exit with a non-zero code.
