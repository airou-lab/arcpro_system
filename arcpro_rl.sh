# 1. Start inference server
cd src/arc_rl_interface/
python inference_server_RNN.py --model final_model_fresh.zip --img_size 128 128 &

# 2. Move back to workspace root
cd ../..

# 3. Start robot bridge
ros2 run bridge real_robot_bridge &

# 4. Start RealSense camera
ros2 launch realsense2_camera rs_launch.py &

ros2 launch f1tenth_stack no_lidar_bringup_launch.py sim:=false &