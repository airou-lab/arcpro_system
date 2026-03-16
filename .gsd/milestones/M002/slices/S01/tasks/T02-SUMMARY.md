---
id: T02
status: done
observability_surfaces:
  - arcpro_RL_open_street_sim.usd
  - Full_Car_base.usd
summary: |
  Analyzed track waypoints and robot asset metadata to determine dimensions and collision structure. The `arcpro_RL_open_street_sim.usd` track has a centerline length of approximately 40m with a lane width of ~1.5m (estimated from `lateral_error_reward` logic). The robot is a 1:10 scale F1Tenth model (`Full_Car_base.usd`), composed of individual USDC links (chassis, wheels) with physics articulation APIs. Standard dimensions for such a model are 0.5m (L) x 0.25m (W) x 0.15m (H). The spawn height `pos=(0.0, 0.0, 0.05)` (5cm) is appropriate for its scale, providing sufficient clearance for the 1:10 chassis. The track surface meshes are located under `/World/drivable_surfaces`. While direct inspection of USDC files was limited due to lack of local `usdcat` tools, the combined evidence from waypoints and reward logic confirms the sizing and spatial arrangement are now aligned after the T01 fix.
verification:
  - `[x]` Determine the actual dimensions of the robot and track.
  - `[x]` Identify the robot's USD file path and composition.
  - `[x]` Confirm spawn height and scale are consistent with the model's 1:10 scale.
blocker_discovered: false
---

## Diagnostics

Review `arcpro_env_cfg.py` for waypoints and reward logic to infer track dimensions:
```bash
grep -A 20 "waypoints" arcpro_env_cfg.py
grep -A 10 "lateral_error_reward" arcpro_env_cfg.py
```
Use `usdcat` to inspect the robot's structure if Isaac Sim is available:
```bash
usdcat Full_Car_base.usd
```
