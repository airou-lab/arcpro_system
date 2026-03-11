# ARCPro Technical Report: Phase 2.1 Debug & Reward Evolution

## 1. Vision System Audit & USD Stabilization

### Problem: Intermittent Detection
During scouting, the `SimpleLaneDetector` was losing confidence (0.0) in specific track sections.
- **Discovery:** Frames captured in `no_graph_sim.usd` showed texture "dead zones" where contrast was insufficient for Canny edge detection.
- **Fix:**
    - **Spawn Shift:** Moved robot from `(-127.76, 60.32)` to `(-125.0, 62.0)`.
    - **Lighting Boost:** Increased DomeLight intensity to `2,000,000` and Exposure to `12.0`.
    - **Ground Contrast:** Injected a dark-gray display color (`0.2, 0.2, 0.2`) to the ground plane to highlight white lane markers.

---

## 2. Telemetry Alignment (The 12-Dim Vector)

The `HierarchicalPathPlanningPolicy` expects a specific 12-dimensional vector (`vec`). We identified a critical mismatch between training and inference:

| Index | Name | Training Usage | Inference (Old) | **Current (Fixed)** |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `turn_bias` | Nav Intent (-1 to 1) | 0.0 | **0.05** (Trigger) |
| 3 | `speed` | Normalized | Raw m/s | **Raw m/s** |
| 8 | `lateral_err` | **Masked (0.0)** | Raw pixels | **Masked (0.0)** |
| 9 | `confidence` | **Masked (0.0)** | 0.0 to 1.0 | **Masked (0.0)** |

**The "Refusal" Discovery:** We found that injecting `lateral_error` into Index 8 was confusing the LSTM, as the policy was trained to ignore telemetry and rely *only* on the CNN for lane staying.

---

## 3. Reward Architecture Math

### A. The "Couch Potato" Trap (Original Model)
The original reward function was:
$R = (1.0 \text{ if in\_lane else } -2.0 \cdot |err|) + (0.3 \cdot speed)$

- **Standing Still:** $R = 1.0$ (Safe, guaranteed)
- **Moving at 1m/s:** $R = 1.3$ (Risky, deviation could lead to -2.0)
- **AI Decision:** Staying still is the global optimum.

### B. Evolution Reward (Our Fix)
$R = \text{Penalty}(-1.0 \text{ if speed } < 0.1) + (1.0 \text{ if moving\_in\_lane}) + (2.0 \cdot speed)$
- **Standing Still:** $R = -1.0$
- **Moving at 1m/s:** $R = 3.0$
- **Result:** Forces the AI to move to achieve any positive score.

### C. Hybrid Racer (Gaussian Magnetic Lane)
Uses a continuous bell curve instead of binary bonuses:
$R_{lane} = 2.0 \cdot e^{-\frac{lateral\_error^2}{0.25}}$
- **Center:** $2.0$
- **0.5m Offset:** $0.73$
- **1.0m Offset (Edge):** $0.03$
- **Goal:** Teaches the AI that "Center is better than just In-Lane," leading to smoother paths.

---

## 4. Control Loop Architecture

### Direct API vs. ROS2 Bridge
We bypassed the ROS2 Action Graph for training to achieve:
1. **Deterministic Timing:** Physics and Python script stay in sync (no network lag).
2. **High Throughput:** 100+ FPS vs 20-30 FPS via the bridge.

### Ackermann Math
The environment converts the AI's `[steer, throttle]` into joint commands using the kinematic model:
- $R_{radius} = \frac{Wheelbase}{\tan(\delta)}$
- $\omega_{inner} = v \cdot \frac{R - half\_track}{R} / radius_{wheel}$
- $\omega_{outer} = v \cdot \frac{R + half\_track}{R} / radius_{wheel}$

---

## 5. Training Stability Constraint
**Crucial Finding:** Isaac Sim's `SimulationApp` is a singleton. Standard RL libraries (like Stable Baselines3) often use `DummyVecEnv` which can try to fork or re-initialize.
- **Stable Config:** Use a single-process environment when using the Direct API to prevent silent crashes during GPU resource allocation.
