# Phase 2 Technical Report: Retraining & Stability Hardening

This report documents the non-obvious technical hurdles, silent crashes, and architectural refinements implemented during Phase 2.1 and 2.2 of the ARCPro RL project.

## 1. The "Silent Crash" Log
During the launch of Phase 2.2, four distinct types of silent failures were encountered. These did not produce Python tracebacks in background mode (`nohup`), requiring manual foreground unit-testing to resolve.

### Issue A: GPU Context Hijacking (Initialization Order)
*   **Problem:** Python scripts would hang or segfault immediately upon start.
*   **Cause:** Isaac Sim's `SimulationApp` is a singleton that initializes the Vulkan/CUDA context. If `torch` or `cv2` were imported *before* `SimulationApp`, they would attempt to lock the GPU device, causing a collision when Isaac Sim tried to do the same.
*   **Solution:** Refactored `train_direct.py` to ensure `SimulationApp` is the very first non-standard import.

### Issue B: Custom LSTM State Mismatch (The Tuple Bug)
*   **Problem:** The script would initialize but crash the moment the first frame was processed by the policy.
*   **Cause:** The `HierarchicalPathPlanningPolicy` expected its internal LSTM states to be a `NamedTuple` (`RNNStates`). However, the Stable Baselines 3 `RecurrentPPO` algorithm passes a standard Python `tuple` of `(hidden_state, cell_state)`.
*   **Solution:** Patched `forward()` and `evaluate_actions()` in `hierarchical_policy.py` with `try...except` blocks to handle both `NamedTuple` and standard tuple indexing.

### Issue C: Vectorized Environment Protocol
*   **Problem:** Data handover between `IsaacDirectEnv` and `RecurrentPPO` failed silently.
*   **Cause:** SB3's Recurrent policies are optimized for vectorized environments. Passing a single raw environment instance caused internal indexing errors in the LSTM buffer.
*   **Solution:** Wrapped the environment in `DummyVecEnv([lambda: env])` to satisfy the API requirements.

### Issue D: Nohup/Interactive Deadlock
*   **Problem:** Script would stop progressing after initialization when run in the background.
*   **Cause:** The `progress_bar=True` parameter in `model.learn()` attempts to update a TTY terminal. In `nohup` mode, there is no TTY, causing the `tqdm` library to block indefinitely.
*   **Solution:** Explicitly disabled `progress_bar` for background training runs.

---

## 2. Visual Stability: 360p Downsampling
Previously, we rendered at **160x90** to match the policy input. This caused high-frequency jitter and artifacts because the internal DLSS/Denoiser systems in Isaac Sim require more pixel data to function correctly.

### Implementation:
1.  **Rendering:** Environment now renders at **640x360** (360p).
2.  **Downsampling:** Integrated `cv2.resize` inside the `_capture_camera` method.
3.  **Interpolation:** Used `cv2.INTER_AREA`. This is specifically chosen because it provides superior edge preservation when shrinking images, ensuring lane lines remain sharp for the CNN.

---

## 3. Physics Hardening (34-Joint Model)
The F1Tenth model is exceptionally complex (26 suspension joints). We implemented a "Runtime Stability Pass" to prevent physics explosions:
*   **Solver Beefing:** Increased Position Iterations to **64** and Velocity Iterations to **32**.
*   **Mass Injection:** Forced rigid body masses (3.0kg Chassis) at runtime to override USD file corruption.
*   **Active Damping:** Forced **50.0 Damping** on all suspension joints via the `ArticulationView` API to stop high-speed oscillation.

---

## 4. Telemetry & Reward Realignment
*   **Indices 8-11:** Reclaimed these previously masked indices to provide the agent with Lateral Offset, Confidence, and Distance Travelled.
*   **"Passivity Trap" Fix:** Implemented a hard penalty (`-1.0`) for speeds below 0.1 m/s to force the agent to explore the throttle space.
*   **Linear vs Gaussian:** Added support for both Linear lane bonuses (easier to find) and Gaussian precision rewards (better for high-speed racing).

---

## 5. Current Performance Baseline
*   **Hardware:** NVIDIA GeForce RTX 3060 (12GB)
*   **Throughput:** 7 FPS
*   **Bottleneck:** High-fidelity physics + 360p Rendering + LSTM temporal backprop.
*   **Next Step:** Isaac Lab Integration (Phase 3) aims to push this to 2,000+ FPS via vectorization.
