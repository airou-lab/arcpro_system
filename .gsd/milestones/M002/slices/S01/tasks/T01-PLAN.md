---
id: T01
title: Inspect arcpro_env_cfg.py for scale and spawn issues.
---

### Why
The robot is clipping through the floor or is incorrectly sized in the simulation. This task is to investigate the environment configuration to find the root cause of the scaling and spawning parameters.

### Files
- `arcpro_env_cfg.py`

### Do
1.  Read `arcpro_env_cfg.py`.
2.  Analyze the code related to vehicle spawning, scaling, and physics properties.
3.  Look for hardcoded values or incorrect calculations for the robot's initial position and size.
4.  Check how the ground plane is defined and if there are any mismatches with the robot's spawn height.

### Verify
1.  Identify the specific lines of code that control the robot's scale and spawn position.
2.  Form a hypothesis about why the clipping and sizing issues are occurring.
