# Gazebo Harmonic (the ROS 2 Jazzy pairing) on top of the lean sim image.
#   docker build -t tt-gz -f docker/gz.Dockerfile docker
# Run with --gpus all for rendered sensors (camera, depth, gpu_lidar).
FROM tt-sim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-opencv \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_DRIVER_CAPABILITIES=all
