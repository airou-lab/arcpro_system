# Master Roadmap: ARCPro RL System

This document tracks the strategic and tactical progression of the ARCPro Reinforcement Learning integration.

## Phase 1: Environment & Robot Foundation (COMPLETE)
- [x] **1.0: Connectivity** - Verified ROS2 installation and bridge connectivity.
- [x] **1.1: Robot Integration** - Sourced and stabilized F1Tenth USD/URDF assets.
- [x] **1.2: Direct API Refactor** - Removed ROS2 overhead for pure Isaac Sim training.
- [x] **1.3: Track Alignment** - Restored OpenStreetUSD geometry and spawn orientations.
- [x] **1.3.5: Vision & Physics Hardening** - Implemented SDF Collisions and standardized GlobalLight.

## Phase 2: Isaac Lab Migration (IMMEDIATE PRIORITY)
- [ ] **2.1: Task Definition** - Refactor USD into Isaac Lab schema (PhysX/Articulation schemas).
- [ ] **2.2: Manager Configuration** - Implement Observation, Action, and Reward Managers.
- [ ] **2.3: Vectorization** - Transition to multi-robot parallel training (128+ agents).
- [ ] **2.4: Infrastructure Verification** - Achieve stable 1M+ step training without simulation crashes.

## Phase 3: Training & Policy Development (PENDING INFRASTRUCTURE)
- [ ] **3.1: Camera Pipeline** - Port 360p -> 160x90 downsampling to Tiled Rendering.
- [ ] **3.2: Reward Strategy** - Implement Gaussian "Magnetic" Lane rewards.
- [ ] **3.3: High-Fidelity Retraining** - Execute 200,000+ step training in vectorized Isaac Lab env.
- [ ] **3.4: Inference Verification** - Confirm 80m+ autonomous lap completion.
- [ ] **3.5: Visual Analytics** - Develop Isaac Lab compliant GUI/HUD for real-time debugging.
