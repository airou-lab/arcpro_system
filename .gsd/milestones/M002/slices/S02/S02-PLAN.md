# S02: Vectorized Lap Counting in TrackManager

**Goal:** Implement robust, vectorized lap counting logic.
**Demo:** Agent's lap count increments correctly when crossing the start line.

## Must-Haves
- Vectorized logic (no Python loops) to track `lap_count` across 128+ agents.
- Handle wrap-around from last waypoint to first waypoint safely.

## Tasks

- [ ] **T01: Add lap_count and last_wp_idx tensors to TrackManager.** `est:1`
- [ ] **T02: Implement update_laps(pos) tracking forward crossings.** `est:2`
- [ ] **T03: Ensure lap logic resets safely when the agent resets.** `est:1`

## Files Likely Touched
- `mdp/track_manager.py`
