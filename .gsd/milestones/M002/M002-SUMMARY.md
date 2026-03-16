---
id: M002
provides:
  - Correct USD track and robot placement at the origin
  - Vectorized lap counting in TrackManager
  - Real-time telemetry HUD overlay using omni.ui
  - 10-lap autonomous verification script with checkpoint discovery
key_decisions:
  - D001: Center track and robot at (0,0,0) in environment configuration.
  - D002: Use a boolean flag enable_hud in ARCProEnvCfg to control UI visibility.
  - D003: Create a custom environment class ARCProEnv to manage the HUD lifecycle.
  - D004: Implement a configurable mock run within verify_policy.py for verification testing.
  - D005: Prefer model_*.pt sorted numerically by iteration for checkpoint discovery.
patterns_established:
  - origin-centered environment configuration for vectorized simulations
  - Config-driven UI and graceful degradation for simulation overlays
  - Structured final JSON telemetry reporting for autonomous task verification
observability_surfaces:
  - verify_placement.py (static config check)
  - structured JSON telemetry output from verify_policy.py
  - HUD overlay in Isaac Lab viewport
requirement_outcomes: []
duration: 1 day
verification_result: failed
completed_at: 2026-03-16
---

# M002: Autonomous Verification Phase

**Established the foundation for a 10-lap autonomous verification pipeline, origin-centered simulation geometry, vectorized lap counting, and real-time telemetry, though full physical verification remains blocked by environment dependencies.**

## What Happened

This milestone focused on validating that the trained policy could complete 10 laps in Isaac Sim without crashing while maintaining a tight lateral error boundary. 
- In **S01**, we diagnosed world geometry clipping by identifying a 200m track offset and re-centering the environment to the origin. 
- In **S02**, we implemented vectorized lap counting in `TrackManager` capable of detecting forward start line crossings and properly handling environment resets. 
- In **S03**, we decoupled the UI logic to build a config-driven `omni.ui` HUD that dynamically pulls laps, speed, and lateral error from the environment. 
- In **S04**, we created a standalone `verify_policy.py` script to discover model checkpoints and run a structured 10-lap evaluation that emits JSON telemetry on completion and robust JSON errors on failure. 

However, all executions throughout this milestone mapped to mock simulations or static configuration checks due to missing underlying Isaac Lab / Isaac Gym physics dependencies in the runtime environment.

## Cross-Slice Verification

The milestone definition of done and success criteria were NOT fully met.

- **Car completes 10 laps autonomously without crashing:** Not met in actual simulation. `verify_policy.py` was built and runs successfully, but uses a configurable mock loop instead of the real Isaac Sim physics step loop (noted as a deviation in S04).
- **Max lateral error remains < 0.3m:** Not met in actual simulation. Demonstrated structurally within the mock script but not validated by a trained policy in a physics-enabled environment.
- **Real-time telemetry (laps, speed, error) is visible in the viewport via an omni.ui HUD:** Partially met. Verified via mocked unit tests for `omni.ui`, but not observable visually due to blocked simulation rendering.
- **The car size and clipping issues in the USD world are diagnosed and resolved:** Met statically. Diagnosed and verified via `verify_placement.py` script, though full physical visual verification is blocked.
- **Definition of Done Check:** Failed. The `.gsd/milestones/M002/slices/S02/S02-SUMMARY.md` file was missing (though its tasks were completed). 

## Requirement Changes

None.

## Forward Intelligence

### What the next milestone should know
- `verify_policy.py`'s internal `run_laps` function must be swapped out for the actual Isaac Lab environment step loops once the physics stack is production-ready for pure inference.
- The `TrackManager` now expects the track to be at `(0,0)`, and any further waypoint logic should align with this origin-centered assumption.

### What's fragile
- The simulation lap time bounds and error bounds are hardcoded in the `verify_policy.py` mock. When integrating real physics, ensure the environment step loop populates real `max_lateral_error` fields per lap.
- The `TrackManager` fallback straight-line track is now origin-centered, but might need further alignment with complex track layouts.

### Authoritative diagnostics
- Final `stdout` output on success, or `stderr` JSON output on failure from `verify_policy.py`. They emit strictly schema-compliant outputs (`{status, reason, details}` for failure, `{status, total_laps, total_time, max_lateral_error, laps[]}` for success).
- Run `python verify_placement.py` to check if the track and robot spawn coordinates are correctly set to origin.

### What assumptions changed
- **Original Assumption:** The robot was sized incorrectly, causing clipping.
- **Actual Finding:** The track was placed 200m away from the robot's spawn point. 
- **Original Assumption:** The verification script would directly boot Isaac Sim and perform a real physics test.
- **Actual Finding:** Heavy physics and Isaac Gym hooks were not available in the current codebase state, so the verification harness was proven against a mock simulation instead.

## Files Created/Modified

- `.gsd/milestones/M002/M002-SUMMARY.md` — Milestone summary documenting the unfulfilled criteria and missing slice summary.