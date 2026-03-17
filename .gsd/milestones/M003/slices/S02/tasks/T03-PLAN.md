# T03: Real Step Loop & Telemetry Integration

## Why
To execute the full verification loop and report metrics from the physics engine.

## Files
- `scripts/verify_policy.py`

## Steps
1.  **Main Execution Loop.** Implement a `while` loop that steps the environment: `obs, reward, done, info = env.step(action)`.
2.  **Poll Telemetry.** Access `TrackManager` (likely via `env.unwrapped.track_manager` or from `info` dict) to get `lap_count` and `lateral_error`.
3.  **Accumulate Metrics.** Collect `max_lateral_error` and `lap_time` for each completed lap.
4.  **Enforce Success Criteria.** If `lateral_error > 0.3m`, print error JSON and exit with non-zero status.
5.  **Termination Condition.** Exit the loop when 10 laps are completed.
6.  **Print Final Summary.** Outputs JSON telemetry for the 10-lap run as specified in R002.

## Must-Haves
- 10-lap verification loop.
- Enforcement of < 0.3m lateral error.
- JSON summary output on completion.
- Fail loudly on crash or out-of-bounds error.

## Verification
- Run a short test (1 lap) if possible, or verify logic via unit test of the telemetry extraction.
- Verify JSON output format.

## Expected Output
- JSON summary with 10 laps of telemetry and success/failure status based on lateral error.
