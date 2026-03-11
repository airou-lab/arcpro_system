---
id: verify-robot-movement
title: Verify Robot Random Movement in Simulation
area: simulation
status: pending
created: 2026-02-17
files:
  - arc_rl_isacc_policy/isaac_ros2_env.py
  - arc_rl_isacc_sim/launch_isaac_ros2.py
  - arc_rl_isacc_sim/setup_ros2_bridge.py
---

## Problem
During Phase 1 verification, the `verify_policy_link.py` script ran successfully with random actions, but the logs showed `Speed=0.00` for all steps. Visual inspection confirmed the robot is **NOT moving**.

## Context
The user verified:
1.  Play button WAS pressed.
2.  Manual ROS2 publication to `/drive_stamped` did NOT move the robot.
3.  Automated verification script sent commands, but robot remained stationary.

## Root Cause Hypotheses
1.  **Topic Mismatch**: User published to `/drive_stamped`, but bridge listens to `/ackermann_cmd`. Need to verify manual publication to `/ackermann_cmd`.
2.  **Joint Names Mismatch**: The Action Graph (`setup_ros2_bridge.py`) assumes specific joint names (`front_left_wheel_steer`, etc.). If the Mushr USD uses different names, the controller is driving nothing.
3.  **Command Type**: The `isaac_ros2_env.py` sends `acceleration` but sets `speed=0.0`. If the Action Graph relies on `speed` for the loopback or control, this might be an issue.

## Next Steps
1.  **Test Manual Topic**: Run `ros2 topic pub /ackermann_cmd ackermann_msgs/msg/AckermannDrive ...` to rule out topic mismatch.
2.  **Inspect USD Joints**: Open Isaac Sim, select the robot, and check the joint names in the Property panel.
3.  **Update Bridge Script**: Modify `setup_ros2_bridge.py` to match the actual joint names of the robot.
