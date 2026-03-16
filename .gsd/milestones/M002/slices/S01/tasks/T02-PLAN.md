# T02: Open or query the USD files to check dimensions and collision shapes.

- **Why**: The 3D model itself might have incorrect dimensions or collision settings.
- **Files**: `arcpro_RL_open_street_sim.usd`, robot's USD file.
- **Do**: Use a USD viewer or tool to inspect the dimensions of the track and the robot. Check collision shapes.
- **Verify**: Determine the actual dimensions of the robot and track.

## Steps
1. Identify the robot's USD file path.
2. Search for tools to inspect USD binary files (e.g., `usdcat`, `usdview`).
3. If `usdcat` is available, convert or read the binary USD files to ASCII.
4. Extract dimensions and collision shape information for the track and the robot.
5. Record the dimensions and collision settings in the task summary.
