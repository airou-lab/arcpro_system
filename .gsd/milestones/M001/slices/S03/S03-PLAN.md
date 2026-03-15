# S03: MDP Logic Refinement.

**Goal:** Synchronize observations and rewards with the new TrackManager.
**Demo:** Print observation indices 8 & 9 during a test run and confirm they change realistically.

## Must-Haves
- No hardcoded X-aligned math.
- Vectorized performance.

## Tasks

- [x] **T01: Update mdp/observations.py to call TrackManager for indices 8 & 9.** `est:1`
- [x] **T02: Update mdp/rewards.py to mask all rewards until env.episode_length_buf >= 20.** `est:1`
- [x] **T03: Verify action normalization in ActionManager config.** `est:1`

## Files Likely Touched
- `mdp/observations.py`
- `mdp/rewards.py`
- `arcpro_env_cfg.py`
