# Requirements

## Active
- **R001**: **Missing M002 S02 Summary**
  - **Status**: Active
  - **Class**: documentation
  - **Description**: S02 completed its tasks but the summary was never written.
  - **Owner**: M003/S01
  - **Source**: Execution (M002 failure)
- **R002**: **Physical 10-Lap Verification**
  - **Status**: Active
  - **Class**: feature
  - **Description**: `verify_policy.py` must use the real `ARCProEnv` from Isaac Lab to step the physics simulation instead of a mock loop. It will load the checkpoint, step observations, track lateral error (<0.3m), count laps, and emit telemetry. It will run headlessly and fail if Isaac Lab dependencies are missing.
  - **Owner**: M003/S02
  - **Source**: User (M002 failure fix)

## Validated
(None)

## Deferred
(None)

## Out of Scope
- **R003**: **HUD in Verification Run**
  - **Status**: Out of Scope
  - **Description**: The 10-lap verification will run headlessly. HUD integration visually during this step is explicitly excluded.
  - **Source**: User decision