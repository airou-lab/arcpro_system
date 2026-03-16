# T01: Add lap_count and last_wp_idx tensors to TrackManager

**Goal:** Initialize stateful tensors in `TrackManager` to enable per-agent lap counting.

## Why
`TrackManager` is currently stateless with respect to the agents. To count laps, it must remember the previous waypoint index for each agent to detect when they've completed a circuit.

## Files
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/mdp/track_manager.py`

## Implementation Details
1.  **Lazy Initialization**: Since `TrackManager` is a singleton and may be initialized before `num_envs` is known, we will initialize `lap_count` and `last_wp_idx` lazily within the first call to `update_laps` or `compute_errors`.
2.  **Stateful Tensors**:
    - `self.lap_count`: `torch.Tensor` of shape `(num_envs,)`, dtype `torch.int32`, device matching waypoints.
    - `self.last_wp_idx`: `torch.Tensor` of shape `(num_envs,)`, dtype `torch.long`, device matching waypoints.

## Steps
- [ ] Add `lap_count` and `last_wp_idx` attributes to `TrackManager.__init__` as `None`.
- [ ] Implement a `_check_state_init(num_envs)` helper method that initializes these tensors if they are `None` or have a different `num_envs`.
- [ ] Call `_check_state_init` inside `compute_errors`.

## Verification
- [ ] Test script that instantiates `TrackManager`.
- [ ] Call `compute_errors` with dummy data for 10 environments.
- [ ] Verify `tm.lap_count` and `tm.last_wp_idx` are initialized with size 10.
