For full documentation please visit the docs site: https://airou-lab.github.io/general_wiki_website/

To clone run:
```bash
git clone -j8 --recurse-submodules=':!src/examples' git@github.com:airou-lab/arcpro_system.git
```

(To add build passing tags)

(To add current version deployment)

(To add current maintainer)

[//]: # (git submodule add -f git@github.com:airou-lab/twist_to_ackermann.git src/base/twist_to_ackermann)

Maintainer until Dec 2027: arikak@ou.edu

When building on new device run in project root:
```bash
rosdep install --from-paths src -y --ignore-src
```


# Running Examples:
```bash
# waypointer example:
 ./waypoint.sh 
 # RL example:
./arcpro_rl.sh 
```

## Reinforcement Learning (Isaac Lab - Phase 3)
The ARCPro RL system is currently migrating to **Isaac Lab** to enable massive parallel training and resolve simulation stability bottlenecks.

- **Status:** Phase 3 Migration in Progress.
- **Goal:** Vectorized training with 128+ agents using the Manager-Based RL API.
- **Digital Twin:** Verified metric asset (25cm WB, 4.092kg) calibrated for high-fidelity physics.

### Simulation & Hardware Alignment (Digital Twin)
The simulation model has been meticulously calibrated to match ARCPro hardware for Zero-Shot Sim2Real transfer.

The F1Tenth simulation model in `src/examples/ARCPro_RL/arc_rl_isacc_sim/f1tenth_trainer/assets/F1Tenth.usd` has been refactored to match exact ARCPro hardware specifications for Zero-Shot Sim2Real transfer.

### Hardware Specifications Applied
- **Total Weight:** 4092 g (4.092 kg)
- **Wheelbase:** 25.0 cm (Kinematic length between axles)
- **Track Width:** 24.0 cm (Horizontal width between wheels)
- **Wheel Radius:** 5.0 cm
- **LIDAR Offset:** [235, 0, 265.23] mm (Forward, Left, Up from chassis root)
- **Camera Offset:** [145, 0, 195] mm (Realsense D435 centered)

### Stability Fixes (Isaac Sim 2025)
To resolve physics explosions and high-frequency jitter (NaN errors) common with 34-joint high-fidelity models in Isaac Sim 2025, the following "Runtime Stability Pass" is implemented in `isaac_direct_env.py`:

1.  **Mass Force-Injection:** Explicitly sets rigid body masses (3.0kg chassis, 0.15kg wheels) at environment startup to eliminate zero-mass errors.
2.  **Damping Override:** Bypasses deprecated USD attributes by forcing **50.0 damping** and **1000.0 stiffness** across all 34 joints via the `ArticulationView` API.
3.  **Solver Beefing:** Increases physics precision to **64 position iterations** and **32 velocity iterations** to eliminate mathematical drift and stabilize the 26-joint suspension.

### Vision & Policy Stability
The following refinements were applied to ensure reliable data flow and training convergence:

1.  **Vision Verification:** Confirmed stable frame capture via `IsaacDirectEnv`. Relocated default spawn to `(-125.0, 62.0)` to ensure the robot always starts on high-contrast textured track segments, resolving "black-screen" initialization bugs.
2.  **Policy Gradient Patch:** Patched `HierarchicalPathPlanningPolicy` to recreate the optimizer *after* custom hierarchical heads (Planning/Control) are initialized. This ensures all sub-networks are registered for backpropagation, which was previously blocked by the standard SB3 `super().__init__` sequence.
3.  **Single-Process Direct API:** Refactored `train_direct.py` to initialize `SimulationApp` **before** any other imports. This prevents GPU context collisions and ensures the Direct API maintains a stable singleton connection to the physics engine.

to just view the xacro file, run 
```bash
./src/base/robot/urdf/models/rsp_xacro_test.sh

# AND in host terminal for gui
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```
