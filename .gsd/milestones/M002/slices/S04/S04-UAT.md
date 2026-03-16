# S04: 10-lap Autonomous Verification — UAT

**Milestone:** M002
**Written:** 2026-03-16

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: The verification script handles test execution internally via a mock run and structured CLI interaction, meaning its exact outputs, error handling, and process termination behaviors can be tested directly from the terminal without requiring a GUI.

## Preconditions

- The script `scripts/verify_policy.py` must exist and be executable.
- Python 3 must be available in the environment.
- A dummy directory with a mock `.pt` file must exist or the default `logs/rsl_rl/arcpro_retraining/model_100.pt` is utilized by the simulation script.

## Smoke Test

Execute `python3 scripts/verify_policy.py --help` to confirm the script loads successfully, runs cleanly, and describes its available arguments (`--checkpoint-dir`, `--induce-crash`, `--induce-error`).

## Test Cases

### 1. Happy Path 10-Lap Verification

1. Execute `python3 scripts/verify_policy.py`
2. **Expected:** 
   - Script runs successfully, logging "Lap X start" and "Lap X end" for all 10 laps to standard output.
   - Script exits with code `0`.
   - The final output is a structured JSON object showing `"status": "success"` and detailing `total_laps`, `total_time`, and `max_lateral_error`.
   - The `max_lateral_error` is verifiable to be less than `< 0.3m`.

### 2. Failure: Missing Checkpoint Directory

1. Execute `python3 scripts/verify_policy.py --checkpoint-dir logs/invalid_path`
2. **Expected:**
   - Script exits immediately with status code `1`.
   - Output on `stderr` contains a structured JSON error payload with `"status": "error"`, `"reason": "Checkpoint directory does not exist"`, and details referencing `logs/invalid_path`.

### 3. Failure: Simulated High Lateral Error

1. Execute `python3 scripts/verify_policy.py --induce-error`
2. **Expected:**
   - Script begins logging laps but stops prematurely (e.g. around lap 8).
   - Script exits with status code `1`.
   - Output on `stderr` contains a structured JSON error indicating the lateral error bound was breached (`"reason": "Lateral error exceeded 0.3m"`).

## Edge Cases

### Simulated Crash Handling

1. Execute `python3 scripts/verify_policy.py --induce-crash`
2. **Expected:**
   - Script terminates before completing 10 laps.
   - Outputs a structured JSON error explaining the crash event (e.g., `"reason": "Simulated crash occurred"`).
   - Exits with a non-zero code.

## Failure Signals

- Exit status code other than `0` (on successful run) or `1` (on expected verification failure).
- Missing JSON blob at the end of execution.
- Max lateral error in the final JSON object exceeds `0.3`.
- "Lap start/end" logs are missing or out of order in the happy path test.

## Notes for Tester

- The current implementation of `verify_policy.py` simulates the physics verification for reliability and speed of pipeline testing. When run normally, the lap times and small lateral errors are mock-generated. The core mechanisms under test are the command line flag mapping, error bounding, telemetry JSON payload emission, and checkpoint resolution.
