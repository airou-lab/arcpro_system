---
id: T01
title: Create mdp/hud.py with ARCProHUD class.
status: done
observability_surfaces:
  - Console log on `omni.ui` import failure.
---

### Summary
Created the `ARCProHUD` class in `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`. This class uses `omni.ui` to create a transparent overlay window in the Isaac Sim viewport. The HUD is designed to display Lap Count, Speed (m/s), and Lateral Error (m) for `env_0`. It includes a safe initialization path for environments where `omni.ui` is not available (e.g., unit tests or headless CLI runs).

### Key Implementation Decisions
- **Class-based Design**: Encapsulated the HUD logic in `ARCProHUD` for easier integration with the main environment loop.
- **Graceful Degradation**: Added a `try-except` block for `import omni.ui` and checked for its existence before creating the window. This allows the simulation to run without the HUD if necessary.
- **Window Styling**: Configured the `omni.ui.Window` with flags to remove the title bar, scrollbars, and resize handles, and set a semi-transparent background to create a clean overlay look.
- **Update Logic**: Provided an `update` method that takes raw telemetry values, decoupling the UI from the environment's internal data structures (though T02 will bridge this).

### Diagnostics
- **Inspect UI Hierarchy**: If the HUD appears incorrectly, use the `omni.ui` debugger (`Window > UI > Debugger`) to inspect the visual tree, check styling properties, and ensure the `ARCProHUD` window is present.
- **Check for Import Errors**: The console will show a "ARCProHUD: omni.ui not found. HUD will be disabled." log message if the required UI library isn't present, confirming graceful degradation.

### Verification Results
- **Unit Tests**: Created `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/tests/test_hud.py` which verifies:
  - [x] HUD initializes safely without `omni.ui`.
  - [x] HUD correctly updates internal label text when `omni.ui` is mocked.
- **Command**: `PYTHONPATH=. python3 src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/tests/test_hud.py` passed with 2/2 tests OK.

### Next Steps
- Implement the `update(env)` method in T02 to pull real telemetry data from the `env` object (specifically `TrackManager` and robot velocity).
- Hook the HUD into the `ARCProEnv` lifecycle in T03.
