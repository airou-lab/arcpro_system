# Phase 3 Plan: Sim2Real Bridge (ROS2 Integration)

## Objective
Develop a bridge node that abstracts the policy from the specific environment, allowing it to control either the simulation (via ROS2 Bridge) or the real ARCPro robot (via VESC/Sensors).

## Tasks
- [ ] Implement a standalone ROS2 Node `policy_bridge_node.py`.
- [ ] Subscribe to standard ROS2 topics: `/camera/image_raw`, `/vehicle_state`.
- [ ] Publish standard Ackermann commands: `/ackermann_cmd`.
- [ ] Ensure the node can load the SB3 model and run inference at a stable frequency.
- [ ] Test integration by switching Isaac Sim back to using its ROS2 Bridge and verifying the policy still works.

## Success Criteria
- Policy can control the robot over network/ROS2 topics.
- System is ready for real-world deployment on the ARCPro robot.
