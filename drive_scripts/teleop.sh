# ros2 topic pub /commands/servo/position std_msgs/msg/Float64 '{data: 0.0}' --once #if iw centre the wheels b4 hand
ros2 launch f1tenth_teleop teleop.launch.py joy_dev:=ttyUSB0
ros2 run twist_to_ackermann twist_to_ackermann --ros-args --params-file src/base/f1tenth_to_arcpro/f1tenth_teleop/config/teleop_twist_joy.yaml
