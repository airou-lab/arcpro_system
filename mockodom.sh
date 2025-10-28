#ros2 topic pub /commands/servo/position std_msgs/msg/Float64 '{data: 0.0}' --once

  ros2 topic pub /drive_stamped ackermann_msgs/msg/AckermannDriveStamped \
  '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "base_link"},
    drive: {steering_angle: 0.0, steering_angle_velocity: 0.0, speed:  0.4, acceleration: 0.0, jerk: 0.0}}' \
  -r 10

