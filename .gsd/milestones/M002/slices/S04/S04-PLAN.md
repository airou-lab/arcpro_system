# S04: 10-lap Autonomous Verification

**Goal:** Final 10-lap proof-of-capability for the trained agent.
**Demo:** Robot completes 10 laps while maintaining < 0.3m lateral error.

## Must-Haves
- Dedicated inference script that runs the policy without training overhead.
- Telemetry output summarizing final run metrics.

## Tasks

- [x] **T01: Create scripts/verify_policy.py.** `est:1`
    - **Why:** To create a standalone script for running the trained policy without the overhead of the training environment.
    - **Files:** `scripts/verify_policy.py`
    - **Do:** Create a new Python script that can load and run a trained policy.
    - **Verify:** The script exists and is executable.
- [x] **T02: Configure it to load the best checkpoint from logs/rsl_rl/arcpro_retraining.** `est:1`
    - **Why:** To ensure the verification script uses the most successful version of the trained model.
    - **Files:** `scripts/verify_policy.py`
    - **Do:** Modify the script to identify and load the best-performing checkpoint from the specified directory.
    - **Verify:** The script correctly loads the intended checkpoint.
- [x] **T03: Execute the verification run.** `est:1`
    - **Why:** To generate the final performance data and prove the agent meets the specified criteria.
    - **Files:** `scripts/verify_policy.py`
    - **Do:** Run the script to start the 10-lap verification run.
    - **Verify:** The run completes and generates telemetry data.

## Files Likely Touched
- `scripts/verify_policy.py`

## Observability / Diagnostics
- **Telemetry output:** The verification script must produce structured JSON or CSV output detailing lateral error per lap, max error, and total time.
- **Failure visibility:** If the policy fails to load, the robot crashes, or the lateral error exceeds 0.3m, the script must exit with a non-zero code and output a clear error message identifying the failure reason.
- **Progress logs:** Log the start and end of each lap to `stdout` to ensure progress can be tracked during the 10 laps.

## Verification
- [x] Run `python3 scripts/verify_policy.py --help` to verify the script is executable and parseable.
- [x] Induce a failure (e.g., missing checkpoint) and verify the script exits with a non-zero status and structured error output.
- [x] Full 10-lap run completes with `< 0.3m` max lateral error and proper telemetry summary generated.