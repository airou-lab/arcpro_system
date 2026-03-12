# ARCPro Reinforcement Learning

This directory contains the high-performance Reinforcement Learning (RL) stack for the ARCPro system, utilizing NVIDIA Isaac Sim and the Stable Baselines3 framework.

## Project Architecture

- **`arc_rl_isacc_sim/`**: Isaac Sim environment logic and digital twin assets.
    - `openStreetUSD/`: USD world models and track geometry.
    - `run_inference.py`: Standard inference script for model verification.
- **`arc_rl_isacc_policy/`**: RL policy definitions and training protocols.
    - `isaac_direct_env.py`: Fast Direct-API Gymnasium environment.
    - `policies/`: Custom hierarchical and recurrent neural network architectures.
    - `train_policy_ros2.py`: Training entry point (supports both Direct and ROS2 modes).

## Key Features

## Phase 3: Isaac Lab Migration (Active)
We are currently migrating from the Direct API to **Isaac Lab** to resolve simulation stability issues and enable vectorized training.

- **Manager-Based RL:** Transitioning to modular Reward, Observation, and Action managers.
- **Massive Vectorization:** Target throughput of 2,000+ FPS using parallel agents.
- **Improved Stability:** Leveraging hardened Isaac Lab schemas for the 34-joint ARCPro robot.

## Getting Started

To run a verification lap with the pre-trained model:
```bash
./arcpro_rl.sh
```

For detailed reward and training information, see:
- [Policy Documentation](arc_rl_isacc_policy/README.md)
- [Reward Architectures](arc_rl_isacc_policy/README_REWARDS.md)
