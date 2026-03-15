# S02: TrackManager Implementation.

**Goal:** Create a robust, vectorized track-following utility.
**Demo:** Run `generate_track.py` and visualize waypoints.

## Must-Haves
- No hardcoded X-aligned math.
- Vectorized performance.

## Tasks

- [x] **T01: Create mdp/track_manager.py with sample_waypoints_from_usd logic.** `est:2`
- [x] **T02: Implement compute_errors(pos, yaw) using vectorized Pytorch math.** `est:1`
- [x] **T03: Create standalone scripts/generate_track.py for offline waypoint verification.** `est:1`

## Files Likely Touched
- `mdp/track_manager.py`
- `scripts/generate_track.py`
