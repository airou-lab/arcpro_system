# T02: Environment & Model Loading

## Why
To instantiate the target environment and the trained policy.

## Files
- `scripts/verify_policy.py`
- `src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env.py`

## Steps
1.  **Register Gym Env.** Import `src.examples.ARCPro_RL.arc_rl_isacc_sim.arcproLab` to trigger environment registration.
2.  **Create Environment.** Use `gym.make("ARCPro-v0", cfg=env_cfg, render_mode=None)` to instantiate the `ARCProEnv`.
3.  **Implement Policy Loading.** Develop a `load_policy(checkpoint_path, env)` function that:
    - Loads the `.pt` file using `torch.load`.
    - Instantiates the `rsl_rl` actor-critic network based on the saved config.
    - Loads the model weights into the network.
    - Sets the model to `eval()` mode.
4.  **Observation Normalization.** Ensure the loaded policy uses the same observation scaling as used during training (likely found in the checkpoint or its accompanying metadata).

## Must-Haves
- `ARCProEnv` is successfully created using `gym.make`.
- `rsl_rl` actor weights are correctly loaded from the `.pt` file.
- The policy is ready for inference.

## Verification
- Run script and verify it initializes the environment and loads the model without crashing.

## Expected Output
- Environment is ready and policy network is populated with trained weights.
