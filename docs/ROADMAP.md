# Master Roadmap: ARCPro RL System

This document tracks the strategic and tactical progression of the ARCPro Reinforcement Learning integration.

## Phase 1: Environment & Robot Foundation (COMPLETE)
- [x] **1.0: Connectivity** - Verified ROS2 installation and bridge connectivity.
- [x] **1.1: Robot Integration** - Sourced and stabilized F1Tenth USD/URDF assets.
- [x] **1.2: Direct API Refactor** - Removed ROS2 overhead for pure Isaac Sim training.
- [x] **1.3: Track Alignment** - Restored OpenStreetUSD geometry and spawn orientations.
- [x] **1.3.5: Vision & Physics Hardening** - Implemented SDF Collisions and standardized GlobalLight.

## Phase 2: Training & Policy Development (PENDING VERIFICATION)
- [ ] **2.1: Camera Pipeline** - Implemented 360p -> 160x90 downsampling.
- [ ] **2.1.5: Reward Strategy** - Implemented "Passivity Trap" penalty.
- [ ] **2.2: Stability Patching** - Fixed HPPO LSTM tuple bug.
- [ ] **2.3: Initial Training** - Completed 200k step run (Identified scale bug).
- [ ] **2.4: Documentation & Cleanup** - Audited READMEs and purged debug scripts.
- [ ] **2.5: Metric Asset Calibration** - Authored verified Atomic Digital Twin (25cm WB, 4.092kg).
- [ ] **2.6: High-Fidelity Retraining** - Re-run the 200,000-step training using the verified metric model.
- [ ] **2.7: Inference Verification** - Confirm 80m+ autonomous lap completion.
- [ ] **2.8: Visual Analytics** - Develop `run_gui_inference.py` with HUD.

## Phase 3: Isaac Lab Migration (IMMEDIATE PRIORITY)
- [ ] **3.1: Task Definition** - Refactor USD into Isaac Lab schema (PhysX/Articulation schemas).
- [ ] **3.2: Manager Configuration** - Implement Observation, Action, and Reward Managers.
- [ ] **3.3: Vectorization** - Transition to multi-robot parallel training (128+ agents).
- [ ] **3.4: Verification** - Achieve stable 1M+ step training without simulation crashes.
