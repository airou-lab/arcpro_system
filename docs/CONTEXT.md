# Phase 2: Policy Loading and Inference - CONTEXT

## Goal
Load the trained Stable Baselines3 (SB3) policy (`final_model.zip`) and run inference within the stabilized `openStreetUSD` environment using the 34-joint high-fidelity F1Tenth model.

## Decisions

### 1. Observation Alignment (Inputs)
- **Image Resolution:** Forced to **160x90** pixels.
- **Color Space:** **RGB** (3-channel).
- **Telemetry Vector:** 12 elements matching the ROS2 contract:
    - `[turn_bias, reserved, goal_dist_masked, speed, yaw_rate, last_steer, last_throttle, last_brake, lat_err_zero, hdg_err_zero, kappa_zero, total_dist]`
- **Normalization:** Image pixels (0-255) must be handled by the SB3 policy's internal feature extractor (usually `NatureCNN`).

### 2. Inference Execution Model (The Loop)
- **Control Frequency:** Target **20Hz** (50ms between policy steps).
- **Execution:** **Synchronous** inference (Simulation steps wait for the policy decision).
- **Inference Library:** **Stable Baselines3 (SB3)**.
- **Latency:** Simulate **20ms** artificial delay between action selection and physics application to mimic real-world hardware.

### 3. Verification & Feedback
- **Feedback:** Terminal **Summary** after each episode (Total reward, Lap time, Avg speed).
- **Failure Condition:** Auto-reset on **Collision** or **Leaving the Track**.
- **Metrics:** Track "Distance Travelled" and "Time to Collision" as primary KPIs.

### 4. Action Space Mapping (Outputs)
- **Steering:** Map policy output [-1.0, 1.0] to **[-0.5, 0.5] radians** (+/- 28.6 degrees).
- **Throttle:** Map policy output [0.0, 1.0] to **Max Acceleration (3.0 m/s²)**.
- **Braking/Reverse:** Map negative throttle outputs to **Reverse** motion.
- **Smoothing:** Apply a **Low-pass filter** (alpha=0.2) to steering actions to prevent high-frequency oscillations in the 26-joint suspension.

## Success Criteria
- Policy loads without error from `final_model.zip`.
- Robot can navigate between the lines for at least 100 steps.
- Zero "NaN" errors during inference.
