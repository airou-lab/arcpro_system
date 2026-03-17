# M003/S02 — Research

**Date:** 2026-03-16

## Summary

Slice S02 is tasked with upgrading the `scripts/verify_policy.py` script from a mock execution loop to a real physical verification loop using Isaac Lab. This requires integrating the `ARCProEnv` environment, loading the trained `rsl_rl` policy, and using the `TrackManager` to verify the success criteria (10 laps, lateral error < 0.3m).

## Recommendation

The verification script should be refactored to use the `AppLauncher` from Isaac Lab to initialize the simulation. It must explicitly disable UI/rendering to run headlessly (as per R003) and ensure that if Isaac Lab imports fail, the script terminates with a clear error message instead of silently falling back to mock logic. The loop should leverage the `TrackManager` singleton to poll `lap_count` and `lateral_error` directly from the environment state.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Environment Initialization | `ARCProEnv` and `ARCProEnvCfg` | Standardized Isaac Lab manager-based RL environment already defined in `arcproLab`. |
| Lap & Error Tracking | `TrackManager.update_laps` / `compute_errors` | Provides consistent vectorized telemetry already used by rewards and HUD. |
| Model Loading | `rsl_rl.OnPolicyRunner` or manual `torch.load` | The training logs in `logs/rsl_rl` indicate the RSL_RL library was used; its runner provides standard inference utilities. |

## Existing Code and Patterns

- `scripts/verify_policy.py` — Current entry point; contains checkpoint discovery logic that should be preserved.
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env.py` — The target environment class.
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py` — Source of truth for lap counting and lateral error.
- `logs/rsl_rl/arcpro_retraining/model_*.pt` — Training checkpoints to be loaded for verification.

## Constraints

- **Headless Execution**: Must run without a GUI. This should be enforced via `AppLauncher` arguments or `SimulationCfg`.
- **Loud Failure**: If Isaac Lab or dependencies are missing, the script must exit with a non-zero status and a descriptive message.
- **Accuracy**: Metrics (laps, error) must come from the physics state, not simulated random distributions.

## Common Pitfalls

- **Simulation Time Scaling**: Ensure the loop accounts for `dt` and `decimation` to measure real-world equivalent lap times.
- **TrackManager Initialization**: `TrackManager` requires a running simulation to sample waypoints from USD if the `.npy` cache is missing. Ensure the env is reset before the first telemetry poll.
- **GPU Context**: Isaac Sim requires a valid GPU context even in headless mode. The script must be run in an environment with appropriate drivers/Vulkan support.

## Open Risks

- **Policy Compatibility**: While the logs suggest `rsl_rl`, if the model was actually trained with Stable Baselines 3 (found in some legacy docs), the loading logic will differ significantly. The implementation must verify the checkpoint format.
- **Missing USD Assets**: The `ARCProEnv` depends on a specific USD track file. If this path is absolute or missing on the verification host, the simulation will crash.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Isaac Lab | [None] | Use standard Isaac Lab patterns for env creation. |
| RSL_RL | [None] | Use for policy inference. |

## Sources

- `scripts/verify_policy.py`
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/`
- `.gsd/milestones/M003/M003-ROADMAP.md`
- `.gsd/DECISIONS.md`
