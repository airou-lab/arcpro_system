# Decisions

<!-- Append-only register of architectural and pattern decisions -->

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| D001 | Center track and robot at (0,0,0) in environment configuration. | Resolves clipping and sizing issues caused by large track offsets. Alignment with `TrackManager` waypoint sampling which centers the path at (0,0). | 2026-03-15 |
| D002 | Use a boolean flag `enable_hud` in `ARCProEnvCfg` to control UI visibility. | Allows the simulation to run headlessly for training or testing without requiring UI libraries or impacting performance, making the HUD a purely optional, additive feature. | 2026-03-16 |
| D003 | Create a custom environment class `ARCProEnv` to manage the HUD lifecycle. | Decouples UI logic from the core simulation loop. `ARCProEnv` acts as a controller, instantiating and updating the HUD, which prevents the main training script from needing to be aware of the UI. | 2026-03-16 |
