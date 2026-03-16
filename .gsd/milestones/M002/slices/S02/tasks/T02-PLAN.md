# Task T02 Plan - Implement update_laps(pos) tracking forward crossings

## Goal
Implement vectorized lap counting by detecting forward crossings of the start line.

## Proposed Changes
### `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`
- Update `get_closest_waypoint_data` to return the indices as well, or create a new method `get_closest_waypoint_indices(pos)`.
- Implement `update_laps(pos)`:
    1. Get current closest waypoint indices `curr_wp_idx`.
    2. Detect "forward" jumps across the start/finish line. A forward jump happens if `curr_wp_idx` is small (near start) and `last_wp_idx` was large (near end).
    3. Increment `self.lap_count` for those environments.
    4. Update `self.last_wp_idx` with `curr_wp_idx`.
- Implement `reset_laps(env_ids)`:
    1. Reset `self.lap_count[env_ids]` to 0.
    2. Reset `self.last_wp_idx[env_ids]` to 0 (or find closest waypoint and set to that).

## Verification Plan
### Automated Tests
- Create a unit test `tests/unit/test_track_manager_laps.py`.
- Test `update_laps` with:
    - Normal forward motion (no lap increase).
    - Crossing the finish line (lap count increases).
    - Stationary agent (no lap increase).
    - Large jumps (should be careful here, maybe limit how large a jump can be to count as a lap).

### Manual Verification
- N/A (Unit tests should suffice for this logic).
