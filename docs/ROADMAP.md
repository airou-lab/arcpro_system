# Master Roadmap: ARCPro RL System

This document tracks the strategic and tactical progression of the ARCPro Reinforcement Learning integration.

## Phase 1: Environment & Robot Foundation (COMPLETE)
- [x] **1.0: Connectivity** - Verified ROS2 installation and bridge connectivity.
- [x] **1.1: Robot Integration** - Sourced and stabilized F1Tenth USD/URDF assets.
- [x] **1.2: Direct API Refactor** - Removed ROS2 overhead for pure Isaac Sim training.
- [x] **1.3: Track Alignment** - Restored OpenStreetUSD geometry and spawn orientations.
- [x] **1.3.5: Vision & Physics Hardening** - Implemented SDF Collisions and standardized GlobalLight (Intensity 1000) to resolve "blind" CNN inputs.

## Phase 2: Training & Policy Development (CURRENT)
- [x] **2.1: Camera Pipeline** - Implemented 360p -> 160x90 downsampling with `cv2.INTER_AREA`.
- [x] **2.1.5: Reward Strategy** - Implemented "Passivity Trap" penalty (-1.0 for < 0.1m/s) to resolve the Couch Potato local minima.
- [x] **2.2: Stability Patching** - Fixed the `HierarchicalPathPlanningPolicy` LSTM tuple bug and enabled `DummyVecEnv` wrapping.
- [x] **2.3: Baseline Training** - Completed 200,000-step `RecurrentPPO` run with confirmed throttle convergence.
- [ ] **2.4: Inference Verification** - Implement `run_inference_v2.py` (Coupled) to confirm 80m+ lap completion without GUI overhead.
- [ ] **2.5: Visual Analytics** - Develop `run_gui_inference.py` with `omni.ui` Canvas for real-time HUD and 2D mapping.

## Phase 3: Scalability Refactor (Isaac Lab)
- [ ] **3.1: Task Definition** - Refactor USD and track into Isaac Lab `DirectRLEnv` schema.
- [ ] **3.2: Parallelization** - Transition to multi-robot vectorized training (Goal: 2,000+ FPS).

## Phase 4: Domain Randomization & Tuning
- [ ] **4.1: Precision Tuning** - Implement Gaussian precision rewards for high-speed racing.
- [ ] **4.2: Robustness** - Add lighting, texture, and friction randomization to the environment.

## Phase 5: Sim2Real Bridge (FINAL)
- [ ] **5.1: Bridge Node** - Develop standalone ROS2 node for SB3 model deployment.
- [ ] **5.2: Hardware Integration** - Connect to physical VESC/Sensors on the ARCPro robot.
- [ ] **5.3: Real-World UAT** - Autonomous lap completion on the physical track.
