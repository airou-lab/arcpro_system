---
id: T01
title: Create mdp/hud.py with ARCProHUD class.
---

### Why
Create a modular HUD for Isaac Lab 2.0 to show simulation telemetry like Lap Count, Speed, and Lateral Error for `env_0`.

### Files
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/hud.py`

### Do
1.  Define `ARCProHUD` class that initializes an `omni.ui.Window`.
2.  Set window properties (no scrollbars, no title bar, overlays viewport).
3.  Implement basic layout using `omni.ui.VStack` and `omni.ui.Label`.
4.  Add placeholders for Lap, Speed, and Lateral Error.

### Verify
1.  Verify file existence.
2.  Run a simple script with `python.sh` that imports the class (mocking `omni.ui` if necessary).
