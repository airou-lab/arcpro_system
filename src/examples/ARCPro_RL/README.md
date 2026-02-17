# ARCPro Reinforcement Learning with Isaac Sim

This project integrates a Reinforcement Learning (RL) policy with an NVIDIA Isaac Sim robot model using ROS2 as a bridge.

## Project Structure

- **`arc_rl_isacc_sim/`**: Contains simulation assets and scripts.
    - `isacc_sim_usd/`: USD world and robot models.
    - `setup_ros2_bridge.py`: Script to generate the Action Graph in Isaac Sim.
    - `launch_isaac_ros2.py`: Automated launcher for Isaac Sim with ROS2 bridge.
- **`arc_rl_isacc_policy/`**: Contains the RL environment and inference logic.
    - `isaac_ros2_env.py`: Gymnasium-compatible environment interface.
    - `verify_policy_link.py`: Plumbing test script for environment/sim interaction.

## Current Setup & Findings

### Isaac Sim Configuration
- **Robot Path**: `/mushr_tx2/mushr_fixed/base_footprint`
- **Articulation Root**: Located at the base footprint.
- **Joint Names**:
    - Steering: `front_left_wheel_steer`, `front_right_wheel_steer`
    - Throttle: `front_left_wheel_throttle`, `front_right_wheel_throttle`, `back_left_wheel_throttle`, `back_right_wheel_throttle`

### Known Issues
- **Physical Movement**: The robot currently does not move despite receiving ROS2 commands. 
    - **Hypothesis**: Joints are configured with `acceleration` drive type and high stiffness in the USD, which may conflict with the `AckermannController` outputs.
    - **Action Graph**: Some warnings persist in the generated graph that require manual refinement in the Isaac Sim GUI.

## Quick Start (Manual Setup)

1. Open Isaac Sim and load `World0.usd`.
2. Open **Window -> Script Editor**.
3. Copy and run the contents of `arc_rl_isacc_sim/setup_ros2_bridge.py`.
4. Press **Play** in Isaac Sim.
5. In a terminal, run the verification script:
   ```bash
   python3 arc_rl_isacc_policy/verify_policy_link.py
   ```
