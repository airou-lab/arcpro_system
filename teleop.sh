# ros2 topic pub /commands/servo/position std_msgs/msg/Float64 '{data: 0.0}' --once #if iw centre the wheels b4 hand
cd ~/Vnavros2setup/workspaces/f1tenth_ws/src
source install/setup.sh
ros2 launch launches teleop.launch.py joy_dev:=ttyUSB0


#TODO merge into main repo later!