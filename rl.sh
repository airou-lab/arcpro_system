cd src/arc_rl_interface/
python inference_server_RNN.py --model final_model_fresh.zip --img_size 128 128

cd ..
cd ..


ros2 run bridge real_robot_bridge
