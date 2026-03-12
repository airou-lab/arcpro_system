# Historical Context Summary (Phases 1 & 2)

> **LEGACY:** This document minifies the context from Phases 1 and 2 for historical reference.

## Phase 1: Foundations
*   **Goal:** Establish ROS2 connectivity and source high-fidelity F1Tenth assets.
*   **Result:** Verified track geometry and spawn orientations in `openStreetUSD`.

## Phase 2: Direct API Development
*   **Goal:** Train HPPO policy using Isaac Sim Direct API.
*   **Bottleneck:** Complex 34-joint robot caused persistent Segmentation Faults and memory instability.
*   **Asset Repair:** Identified and corrected a 100x scale error in joint local positions within `F1Tenth_Metric_Baked.usd`.
*   **Decision:** Moved to Phase 3 (Isaac Lab) to resolve infrastructure limitations.
