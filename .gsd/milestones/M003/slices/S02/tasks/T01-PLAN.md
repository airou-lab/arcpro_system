# T01: Dependency & AppLauncher Setup

## Why
To ensure the script runs in the Isaac Lab context and fails gracefully if not.

## Files
- `scripts/verify_policy.py`

## Steps
1.  **Initialize AppLauncher.** Add `from omni.isaac.lab.app import AppLauncher` at the top of the script.
2.  **Add CLI Arguments.** Add `AppLauncher.add_argparse_args()` to the script's `argparse` setup.
3.  **Loud Failure Logic.** Add a `try/except` block to import `omni.isaac.lab` and other dependencies.
4.  **Initialize Simulation.** Call `app_launcher = AppLauncher(args)` and then `from omni.isaac.lab.app import AppLauncher`.
5.  **Headless Configuration.** Ensure the `AppLauncher` is configured for headless operation by default for verification runs.

## Must-Haves
- `AppLauncher` is used to start the simulation.
- Descriptive error messages if `omni.isaac.lab`, `torch`, or `rsl_rl` are missing.
- Exit with non-zero code on missing dependencies.

## Verification
- Run `python3 scripts/verify_policy.py --help` and verify it identifies as an Isaac Lab script.
- Verify "loud failure" by running in an environment without Isaac Lab.

## Expected Output
- A script that correctly initializes Isaac Sim and reports missing dependencies if any.
