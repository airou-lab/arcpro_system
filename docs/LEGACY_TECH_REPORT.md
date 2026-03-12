# Legacy Technical Report (Phase 1)

## Executive Summary
This report summarizes the technical findings during the initial development of the ARCPro RL system using the Isaac Sim Direct API.

## Key Technical Findings
1.  **High-Fidelity Physics:** The 34-joint robot requires high solver iterations (64 pos, 32 vel) and aggressive joint damping (50.0) to remain stable.
2.  **Asset Scaling Bug:** Discovered that joint local positions were not correctly scaled when mesh vertices were shrunk to metric size. This caused "disconnected" wheels. Fixed by scaling `physics:localPos` by 0.01.
3.  **Direct API Limits:** The Isaac Sim Direct API (legacy) is prone to segmentation faults when resetting environments with complex hierarchical articulations.

## Infrastructure Decision
Due to the instability of the Direct API, the project has moved to **Isaac Lab** for Phase 2 to leverage modern vectorized environment management.
