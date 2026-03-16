# T03: Run a dummy environment test to visually or programmatically verify robot placement.

## Goal
Verify that the robot spawns correctly on the track surface at the origin after the fix in T01.

## Must-Haves
- A test script that initializes the environment.
- The robot should spawn at `(0.0, 0.0, 0.05)`.
- The track should be at `(0.0, 0.0, 0.0)`.
- Visual or programmatic verification of the robot's height relative to the track.

## Steps
1. Create a test script `verify_placement.py` that uses the `arcpro_env_cfg.py` configuration.
2. Run the script with Isaac Sim/Omniverse (assuming it's available or use a mock if not, but the goal is "real thing").
3. Programmatically check if the robot's Z-coordinate is above the track surface and not falling.
4. If possible, capture a screenshot or log the distance from the ground.

## Verification
- [ ] Robot Z-coordinate remains stable at ~0.05m (or slightly above track surface).
- [ ] No "collision" or "falling" errors in the logs.
- [ ] Robot is visually on the track (if GUI is available).
