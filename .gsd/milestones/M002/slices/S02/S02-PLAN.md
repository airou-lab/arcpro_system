# S02: Vectorized Lap Counting in TrackManager

**Goal:** Implement robust, vectorized lap counting logic.
**Demo:** Agent's lap count increments correctly when crossing the start line.

## Must-Haves
- Vectorized logic (no Python loops) to track `lap_count` across 128+ agents.
- Handle wrap-around from last waypoint to first waypoint safely.

## Tasks

- [x] **T01: Add lap_count and last_wp_idx tensors to TrackManager.** `est:1`
  - Why: Track per-agent progress along the centerline.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`
  - Do: Add `lap_count` and `last_wp_idx` as torch tensors. Initialize lazily in `get_closest_waypoint_data` or `compute_errors` based on input shape.
  - Verify: Instantiate `TrackManager`, call `compute_errors` with dummy positions, and confirm tensors are created with correct size and type.
- [x] **T02: Implement update_laps(pos) tracking forward crossings.** `est:2`
  - Why: Vectorized lap counting by detecting forward crossings of the start line.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`
  - Do: Implement logic to increment `lap_count` when `last_wp_idx` jumps from near-end to near-start.
  - Verify: Unit test with synthetic trajectories that cross the start line.
- [x] **T03: Ensure lap logic resets safely when the agent resets.** `est:1`
  - Why: Prevent lap counts from carrying over between episodes.
  - Files: `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`
  - Do: Add `reset_laps(env_ids)` method.
  - Verify: Call `reset_laps` and confirm tensors are zeroed for specified indices.

## Files Likely Touched
- `mdp/track_manager.py`
