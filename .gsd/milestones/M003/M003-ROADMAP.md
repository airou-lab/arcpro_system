# M003: Physical Verification Fixes

**Vision:** Resolve the verification failures from M002 by ensuring documentation is complete and the verification script uses a genuine Isaac Lab physics step loop to validate the trained policy.

**Success Criteria:**
- S02 summary from M002 is written and accurate.
- `verify_policy.py` integrates `ARCProEnv` and runs a real environment step loop.
- `verify_policy.py` fails loudly if Isaac Lab dependencies are missing.
- `verify_policy.py` successfully measures 10 laps and enforces the < 0.3m lateral error bound using environment data.

---

## Slices

- [x] **S01: M002 Documentation Cleanup** `risk:low` `depends:[]`
  > After this: The missing S02 summary from M002 is written based on its completed tasks.

- [x] **S02: Physical Verification Loop Integration** `risk:high` `depends:[]`
  > After this: `scripts/verify_policy.py` initializes the Isaac Lab environment, steps it with the trained model, and reports telemetry.

---

## Boundary Map

### S01
Produces:
  - `.gsd/milestones/M002/slices/S02/S02-SUMMARY.md`

Consumes:
  - Task summaries from `.gsd/milestones/M002/slices/S02/tasks/`

### S02
Produces:
  - Updated `scripts/verify_policy.py` using `ARCProEnv` and `torch` models instead of a mock loop.

Consumes:
  - Existing `ARCProEnv` and `TrackManager` structure.