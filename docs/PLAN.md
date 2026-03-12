# Phase 3 Plan: Isaac Lab Migration (Parallel Training)

## Objective
Migrate the single-robot Direct API environment to Isaac Lab to enable multi-robot vectorized training, significantly increasing training throughput.

## Tasks
- [ ] Define Isaac Lab `Asset` schemas for the 34-joint ARCPro robot.
- [ ] Implement `ARCProTask` inheriting from `RLTask` or `DirectRLEnv` in Isaac Lab.
- [ ] Port the verified `SimpleLaneDetector` reward logic to the Isaac Lab task.
- [ ] Configure `Hydra` for experiment management and scaling.
- [ ] Train a vectorized policy with 128+ parallel robots.

## Success Criteria
- Policy achieves same or better performance than Phase 2.6.
- Training time for 1M steps reduced by >10x via vectorization.
