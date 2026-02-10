# 1. Start inference server
cd src/examples/arc_rl_interface

#todo, should find an easier way to parse a model in?
python inference_server_RNN.py --model final_model_fresh.zip --img_size 128 128 &

# go back to ws root js 2 not confuse if cancel
cd ../..

# tcp message -> ros2 (model built on unity so forced to use tcp, idt can directly use dds)
ros2 run bridge real_robot_bridge &

# realesense
ros2 launch realsense2_camera rs_launch.py &
# vesc
ros2 launch f1tenth_stack no_lidar_bringup_launch.py sim:=false &