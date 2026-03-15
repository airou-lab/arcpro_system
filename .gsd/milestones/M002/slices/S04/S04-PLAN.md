# S04: 10-lap Autonomous Verification

**Goal:** Final 10-lap proof-of-capability for the trained agent.
**Demo:** Robot completes 10 laps while maintaining < 0.3m lateral error.

## Must-Haves
- Dedicated inference script that runs the policy without training overhead.
- Telemetry output summarizing final run metrics.

## Tasks

- [ ] **T01: Create scripts/verify_policy.py.** `est:1`
- [ ] **T02: Configure it to load the best checkpoint from logs/rsl_rl/arcpro_retraining.** `est:1`
- [ ] **T03: Execute the verification run.** `est:1`

## Files Likely Touched
- `scripts/verify_policy.py`
