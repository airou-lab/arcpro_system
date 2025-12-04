ros2 launch waypointer full.launch.py
#sleep 5
#
#ros2 topic pub /drive ackermann_msgs/msg/AckermannDriveStamped \
#'{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "base_link"},
# drive: {steering_angle: 0.0, steering_angle_velocity: 0.0, speed: .1, acceleration: 0.0, jerk: 0.0}}' \
#--once
