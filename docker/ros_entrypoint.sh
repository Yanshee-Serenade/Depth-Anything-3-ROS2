#!/bin/bash
# ROS2 Docker Entrypoint Script
# This script sources the ROS2 environment and executes the command

set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Execute the command
exec "$@"
