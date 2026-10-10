#!/usr/bin/env bash
# Run the stack in the 3-D Gazebo world (GPU, headless).
#   docker/run_sim3d.sh [name] [launch args...]
#   docker/run_sim3d.sh full seed:=0 turn_radius:=1.2
# Results: /tmp/simout/<name>/{result.json,trajectory.csv,run.png}, log: /tmp/simout/<name>.log
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-run3d}"; shift || true
out=/tmp/simout
mkdir -p "$out"
rm -rf "${out:?}/$name"

if ! docker image inspect tt-gz >/dev/null 2>&1; then
  docker image inspect tt-sim >/dev/null 2>&1 || docker build -t tt-sim -f docker/sim.Dockerfile docker
  docker build --build-arg http_proxy="${http_proxy:-}" --build-arg https_proxy="${https_proxy:-}" \
    -t tt-gz -f docker/gz.Dockerfile docker
fi

# USER/GZ_PARTITION: without a passwd entry gz-transport derives an unstable partition and
# processes can't discover each other. render/video groups: EGL device access.
# Parallel runs: private build dirs and container-local DDS discovery.
docker run --rm --gpus all --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" -e USER=tt -e GZ_PARTITION="tt_$name" \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -v "$PWD":/ws -v "$out":/out -w /ws tt-gz bash -c "
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install --build-base /tmp/build --install-base /tmp/install \
        --paths src/rc_car_controller src/tt_obstacle_course src/tt_course_sim \
        > /tmp/build.log 2>&1 || { cat /tmp/build.log; exit 1; }
    source /tmp/install/setup.bash
    timeout 900 ros2 launch tt_course_sim sim3d.launch.py output_dir:=/out/$name $*
  " > "$out/$name.log" 2>&1 || true

grep -E '=== 3D RUN' "$out/$name.log" || echo "no result line; see $out/$name.log"
