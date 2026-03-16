# M002: Autonomous Verification Phase

**Vision:** Prove the trained policy meets the project's autonomous driving targets in Isaac Sim.

## Success Criteria
- Car completes 10 laps autonomously without crashing.
- Max lateral error remains < 0.3m.
- Real-time telemetry (laps, speed, error) is visible in the viewport via an omni.ui HUD.
- The car size and clipping issues in the USD world are diagnosed and resolved.

## Slices

- [x] **S01: USD World & Robot Sizing Debugging.** `risk:high` `depends:[]`
  > After this: We are certain the robot fits on the track, does not clip, and the spawn coordinates are correct.
- [ ] **S02: Vectorized Lap Counting in TrackManager.** `risk:medium` `depends:[S01]`
  > After this: TrackManager correctly tracks laps and handles wrap-around without loops.
- [ ] **S03: Isaac Lab 2.0 HUD Overlay (omni.ui).** `risk:medium` `depends:[S02]`
  > After this: HUD displays real-time telemetry over the simulation.
- [ ] **S04: 10-lap Autonomous Verification.** `risk:medium` `depends:[S03]`
  > After this: `scripts/verify_policy.py` confirms the policy completes 10 laps smoothly.
