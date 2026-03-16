# S03 UAT: Isaac Lab 2.0 HUD Overlay

This UAT script verifies that the `omni.ui` Heads-Up Display (HUD) correctly initializes, displays real-time telemetry, and respects its configuration settings within the Isaac Lab simulation environment.

## Preconditions
- The ARCPro Isaac Lab project is loaded.
- The `ARCPro-v0` environment is registered and can be launched.
- The simulation environment can be configured via a Python script (e.g., `arcpro_main.py`).

## Test Case 1: HUD Enabled by Default

**Objective:** Verify the HUD appears and updates correctly when enabled in the configuration.

**Steps:**

1.  **Configure Environment:** Ensure the environment configuration file (e.g., `arcpro_env_cfg.py` or the script that creates `ARCProEnvCfg`) has `enable_hud` set to `True`. This is the default, so no changes may be needed.
2.  **Launch Simulation:** Run the main simulation script that loads the `ARCPro-v0` environment.
    ```bash
    ./isaaclab.sh -p src/examples/ARCPro_RL/arc_rl_isacc_sim/arcpro_main.py
    ```
3.  **Observe Viewport:** Once the simulation starts and the robot is spawned, look at the top-left corner of the viewport.
4.  **Let Simulation Run:** Allow the simulation to run for at least 30 seconds to observe the values changing.

**Expected Outcomes:**

- **A)** A small, semi-transparent window (the HUD) is visible in the top-left of the viewport.
- **B)** The HUD contains three labels: `Lap:`, `Speed (m/s):`, and `Lat. Error (m):`.
- **C)** Initial values are `0`, `0.00`, and a small non-zero number, respectively.
- **D)** As the robot moves, the `Speed (m/s)` and `Lat. Error (m)` values update continuously in real-time.
- **E)** If a lap is completed, the `Lap:` count increments.

## Test Case 2: HUD Disabled

**Objective:** Verify that the HUD does not appear when it is disabled in the configuration.

**Steps:**

1.  **Configure Environment:** Modify the script that creates the `ARCProEnvCfg` to explicitly set `enable_hud=False`.
    ```python
    # In arcpro_main.py or similar
    env_cfg = ARCProEnvCfg(num_envs=1, enable_hud=False)
    ```
2.  **Launch Simulation:** Relaunch the simulation with the modified configuration.
    ```bash
    ./isaaclab.sh -p src/examples/ARCPro_RL/arc_rl_isacc_sim/arcpro_main.py
    ```
3.  **Observe Viewport:** Once the simulation starts, look at the top-left corner of the viewport.

**Expected Outcomes:**

- **A)** The HUD window is **not** visible.
- **B)** The simulation runs normally otherwise.
- **C)** The console may contain a log message like "ARCProEnv: HUD disabled by config."

## Test Case 3: Robustness to Reset

**Objective:** Verify the HUD handles an environment reset without crashing.

**Steps:**

1.  **Enable HUD:** Ensure `enable_hud` is set to `True`.
2.  **Launch Simulation:** Start the simulation.
3.  **Induce a Reset:** Manually trigger a reset if possible, or wait for the robot to fail the task (e.g., by crashing or going too far off-track), which will trigger an automatic reset.
4.  **Observe HUD:** Pay attention to the HUD when the robot respawns.

**Expected Outcomes:**

- **A)** The simulation does not crash or throw UI-related errors in the console.
- **B)** After the reset, the HUD's values reset to their initial state (Lap: 0, Speed: 0.00, etc.).
