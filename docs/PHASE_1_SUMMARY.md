# Research Summary: Phase 1 - Environment Verification

## Isaac Sim ROS2 Bridge
- **Extension**: `isaacsim.ros2.bridge` must be enabled.
- **Action Graph Nodes**:
    - `ROS2Context`: Managed the ROS2 middleware context.
    - `ROS2SubscribeAckermannDrive`: Subscribes to `AckermannDrive` messages.
    - `AckermannController`: Converts Ackermann commands to joint angles and velocities.
    - `IsaacArticulationController`: Controls the robot's physical joints.
    - `ROS2CameraHelper`: Publishes camera feed (RGB, Depth, Info).

## Ackermann Configuration (Mushr)
- **Wheelbase**: ~0.19m - 0.33m (needs verification against URDF).
- **Track Width**: ~0.20m.
- **Joints**: 
    - Steering: `front_left_wheel_steer`, `front_right_wheel_steer`.
    - Throttle: `front_left_wheel_throttle`, `front_right_wheel_throttle`, `back_left_wheel_throttle`, `back_right_wheel_throttle`.

## ROS2 Topics
- **Command**: `/ackermann_cmd` (mapped from `/mushr/command/drive` in some scripts).
- **Image**: `/camera/image_raw`.
- **State**: `/vehicle_state`.

## Verification Strategy
1.  **Connectivity**: Use `ros2 topic list` to ensure topics are being published/subscribed by Isaac Sim.
2.  **Actuation**: Publish a manual `AckermannDrive` message and observe robot movement in simulation.
3.  **Sensors**: Use `ros2 run image_view image_view --ros-args -r image:=/camera/image_raw` to verify visual output.
