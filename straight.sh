#!/bin/bash
# ==============================================================================
# ARC Pro - Out-of-the-box Straight Line Movement Test
# Description: Launches VESC hardware driver and drives the robot forward at 0.4 m/s.
# ==============================================================================

set -e

# Source ROS 2 environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace environments (legacy and new)
if [ -f "/home/arc/Vnavros2setup/workspaces/f1tenth_ws/install/setup.bash" ]; then
    source "/home/arc/Vnavros2setup/workspaces/f1tenth_ws/install/setup.bash"
fi
if [ -f "/home/arc/arcpro_system/install/setup.bash" ]; then
    source "/home/arc/arcpro_system/install/setup.bash"
fi

echo "=================================================="
echo "  ARC Pro Robot - Straight Drive Test"
echo "  Speed: 0.4 m/s | Steering: 0.0 rad"
echo "  Press Ctrl+C at any time to STOP the robot"
echo "=================================================="

# Function to stop the robot and cleanup on exit
cleanup() {
    echo ""
    echo "Stopping robot and cleaning up processes..."
    ros2 topic pub /ackermann_cmd ackermann_msgs/msg/AckermannDriveStamped \
      '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "base_link"}, drive: {steering_angle: 0.0, speed: 0.0}}' \
      --once >/dev/null 2>&1 || true

    ros2 topic pub /commands/motor/speed std_msgs/msg/Float64 '{data: 0.0}' --once >/dev/null 2>&1 || true

    if [ -n "${VESC_PID:-}" ]; then
        kill "$VESC_PID" >/dev/null 2>&1 || true
    fi
    killall -q -9 vesc_driver_node ackermann_to_vesc_node >/dev/null 2>&1 || true
    echo "Done. Robot stopped safely."
}
trap cleanup EXIT INT TERM

# Start VESC stack if not already running
if ! pgrep -x 'vesc_driver_node' >/dev/null; then
    echo "Starting VESC hardware driver..."
    ros2 launch launches vesc.launch.py &
    VESC_PID=$!
    echo "Waiting 3 seconds for VESC connection..."
    sleep 3
else
    echo "VESC hardware driver is already running."
fi

echo "Publishing forward drive command (speed=0.4 m/s)..."
ros2 topic pub /ackermann_cmd ackermann_msgs/msg/AckermannDriveStamped \
  '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "base_link"},
    drive: {steering_angle: 0.0, steering_angle_velocity: 0.0, speed: 0.4, acceleration: 0.0, jerk: 0.0}}' \
  -r 10
