# S03: Isaac Lab 2.0 HUD Overlay (omni.ui)

**Goal:** Create an `omni.ui` overlay for simulation telemetry.
**Demo:** Viewport shows Lap Count, Speed, and Lateral Error for `env_0`.

## Must-Haves
- `omni.ui.Window` overlay without scrollbars or title bars.
- Updates cleanly on environment steps without dropping FPS.

## Tasks

- [x] **T01: Create mdp/hud.py with ARCProHUD class.** `est:1`
  - Why: Modularize the HUD UI for easy integration and maintenance.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`
  - Do: Create `ARCProHUD` class with `omni.ui.Window`.
  - Verify: Unit test that can be run with `python.sh` and mocks `omni.ui` if necessary.

- [x] **T02: Implement update(env) pulling from TrackManager and robot velocity.** `est:2`
  - Why: Display live telemetry from the simulation.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`
  - Do: Add `update` method that takes `env` and updates UI labels.
  - Verify: Run simulation and check for updated text.

- [x] **T03: Hook HUD initialization into ARCProEnvCfg if enabled.** `est:1`
  - Why: Configuration-driven UI activation.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py`
  - Do: Add `enable_hud` flag to config and initialize `ARCProHUD` if true.
  - Verify: Enable/disable HUD from config and see the result.

## Observability / Diagnostics
- **Runtime Signals**: HUD logs initialization and update status to the console if `debug` mode is enabled.
- **Inspection Surfaces**: Use `omni.ui` debug tools to inspect the widget hierarchy.
- **Failure Visibility**: HUD should catch and log exceptions during `update` to avoid crashing the simulator.
- **Redaction**: No sensitive data is processed by the HUD.

## Verification
- [x] **V1: HUD window appears in Isaac Lab.**
- [x] **V2: Telemetry updates (Lap, Speed, Lateral Error) match internal state.**
- [x] **V3: HUD handles environment reset without errors.**
- [x] **V4: HUD logs an error but does not crash if telemetry data is missing.**

## Files Likely Touched
- `mdp/hud.py`
- `arcpro_env_cfg.py`
