# M003: Physical Verification Fixes — Context

**Intent:** Address the failures from M002 by writing the missing slice documentation and upgrading the verification script to use a real physics simulation loop instead of a mock.

**Scope & Constraints:**
- The verification script (`scripts/verify_policy.py`) must use the real `ARCProEnv` to step the simulation.
- It must run headlessly without attempting to render the `omni.ui` HUD.
- If Isaac Lab dependencies are missing, the script should fail loudly rather than falling back to the mock.
- The success criteria remain: 10 laps completed, max lateral error < 0.3m.

**Risks:**
- Missing Isaac Lab dependencies might still block execution if the environment isn't fully configured.
- The `TrackManager` and environment might need minor tweaks to surface the required data (lap counts, lateral errors) to the headless script loop.