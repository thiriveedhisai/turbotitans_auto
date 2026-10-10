#!/usr/bin/env bash
# Build (first time) and run the obstacle-course stack in the 2-D simulator.
#   docker/run_sim.sh [name] [launch args...]
#   docker/run_sim.sh nonav use_nav2:=false
#   docker/run_sim.sh norails rails:=false seed:=3
# Results: /tmp/simout/<name>/{result.json,trajectory.csv,run.png}, log: /tmp/simout/<name>.log
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-run}"; shift || true
out=/tmp/simout
mkdir -p "$out"
rm -rf "${out:?}/$name"

if ! docker image inspect tt-sim >/dev/null 2>&1; then
  docker build --build-arg http_proxy="${http_proxy:-}" --build-arg https_proxy="${https_proxy:-}" \
    -t tt-sim -f docker/sim.Dockerfile docker
fi

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -v "$PWD":/ws -v "$out":/out -w /ws tt-sim bash -c "
    source /opt/ros/jazzy/setup.bash
    [ -d install/tt_course_sim ] || colcon build --symlink-install \
        --paths src/rc_car_controller src/tt_obstacle_course src/tt_course_sim
    source install/setup.bash
    timeout 900 ros2 launch tt_course_sim sim_course.launch.py output_dir:=/out/$name $*
  " > "$out/$name.log" 2>&1 || true

grep -E '=== RUN' "$out/$name.log" || echo "no result line; see $out/$name.log"
