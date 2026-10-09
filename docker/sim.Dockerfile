# Lean ROS 2 Jazzy + Nav2 image for running the obstacle-course stack in the 2D sim.
#   docker build -t tt-sim -f docker/sim.Dockerfile docker
#   see src/tt_course_sim/README section in the top-level README for usage
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-nav2-controller ros-jazzy-nav2-planner ros-jazzy-nav2-behaviors \
      ros-jazzy-nav2-bt-navigator ros-jazzy-nav2-lifecycle-manager \
      ros-jazzy-nav2-smac-planner ros-jazzy-nav2-mppi-controller \
      ros-jazzy-nav2-costmap-2d ros-jazzy-nav2-msgs \
      ros-jazzy-ackermann-msgs ros-jazzy-tf2-ros \
      python3-numpy python3-yaml python3-pytest python3-matplotlib \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
