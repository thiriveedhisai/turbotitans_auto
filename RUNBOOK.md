# Obstacle course runbook (on the truck)

Quick field reference. Background and design: [README.md](README.md).

## 1. What goes on the device

The Jetson needs these packages from `src/`:

| Package | Why |
|---|---|
| `tt_obstacle_course` | The whole stack: nodes, launch, `config/tuning.yaml`, `config/mission.yaml` |
| `rc_car_controller` | Steering servo and ESC (PCA9685) |
| `sllidar_ros2` | RPLidar A1 driver |
| `zed-ros2-wrapper` | ZED 2i driver (submodule) |

Leave these off the truck (they are sim/laptop only): `tt_course_sim`, `docker/`, `markers/`
(print `markers/all_markers.pdf` instead), `build/ install/ log/`.

**First install** (internet on the Jetson): follow README §2, then build only what the truck uses:

```bash
cd ~/turbotitans_auto
git checkout feature/obstacle-course && git pull
colcon build --symlink-install --packages-up-to tt_obstacle_course rc_car_controller sllidar_ros2 zed_wrapper
source install/setup.bash
```

**Updating from a laptop** (no internet on the truck): copy the changed packages, then rebuild them.

```bash
# laptop
rsync -av --delete --exclude __pycache__ src/tt_obstacle_course src/rc_car_controller \
      <user>@<jetson-ip>:~/turbotitans_auto/src/
# jetson
cd ~/turbotitans_auto && colcon build --symlink-install --packages-select tt_obstacle_course rc_car_controller
```

With `--symlink-install`, edits to `.py` and `.yaml` files apply on the next launch with no rebuild.
New files, or changes to `setup.py`, still need a rebuild.

**Field-tuned files** (`src/tt_obstacle_course/config/`):
- `tuning.yaml`: vehicle dimensions, turn radius, lidar flag mask `scan_self_mask`, detector thresholds.
- `mission.yaml`: zone order, steps, and triggers.

Before every session:
- [ ] `python3 -c "import cv2; cv2.aruco"` works.
- [ ] `ls /dev/ttyUSB*` shows the lidar.
- [ ] The ZED is on a USB3 port.
- [ ] Battery is charged, and the wheels are off the ground for the first test.

## 2. Run the whole course

```bash
source ~/turbotitans_auto/install/setup.bash
mkdir -p ~/runs
ros2 launch tt_obstacle_course obstacle_course.launch.py turn_radius:=1.07 2>&1 | tee ~/runs/$(date +%m%d_%H%M).log
```

- Boot it **level and still**: the terrain monitor calibrates the IMU during the first 2 s.
- It waits for the green start light. A red disc in view holds the start.
- To start by hand, without the light:
  `ros2 topic pub --once /race_start std_msgs/msg/Bool "{data: true}"`

Useful launch arguments:

| Argument | Default | Use |
|---|---|---|
| `turn_radius:=1.07` | tuning value | Measured minimum turning radius (m) |
| `start_zone:=<zone>` | none | Begin at this zone (section 4) |
| `mission:=<file>` | package `mission.yaml` | Custom or cut-down mission |
| `tuning:=<file>` | package `tuning.yaml` | Alternative tuning copy |
| `motor:=false` | true | Everything runs, but the wheels never move (bench test) |
| `camera_nodes:=false` | true | No start light or AprilTag detection |
| `sensors:=false` | true | Don't start the ZED and lidar drivers (you run them yourself) |

## 3. Stop

| What | Command |
|---|---|
| **E-stop** (speed 0, latched) | `ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"` |
| Release e-stop | `ros2 topic pub --once /estop std_msgs/msg/Bool "{data: false}"` |
| Quit everything | `Ctrl-C` in the launch terminal |

- E-stop zeroes speed but holds the steering where it is.
- `Ctrl-C` makes `rc_car_controller` set the servo and ESC to neutral and switch the PWM outputs off.
- If the launch dies or stops publishing, the controller goes to neutral on its own after 0.5 s without commands.
- Keep the RC transmitter or the physical kill switch within reach anyway.

## 4. Run only part of the course

Zone names, in order:
`start ramp_up bridge helix tunnel narrow gravel bank_approach bank_uturn pothole wide buckets hoops car_wash finish`

**From one zone to the end:** place the truck at that zone's entrance, facing in, then:

```bash
ros2 launch tt_obstacle_course obstacle_course.launch.py start_zone:=buckets
ros2 topic pub --once /race_start std_msgs/msg/Bool "{data: true}"
```

- A misspelt zone name silently starts at `start`. Check the `=== ZONE ... ===` line.
- For `buckets` and `hoops`, the pose at start stands in for the entrance just passed. Line the truck up square to the field.

**One zone only:** make a cut-down mission and run it.

```bash
python3 - <<'EOF'
import yaml
src = 'src/tt_obstacle_course/config/mission.yaml'
m = yaml.safe_load(open(src))
m['zones'] = [z for z in m['zones'] if z['name'] in ('wide', 'buckets')]   # zones to keep
yaml.safe_dump(m, open('/tmp/only_buckets.yaml', 'w'), sort_keys=False)
EOF
ros2 launch tt_obstacle_course obstacle_course.launch.py mission:=/tmp/only_buckets.yaml start_zone:=wide
```

The run ends with `FINISHED` after the last zone you kept.

**One maneuver (behavior) by hand:** stop the mission manager so it doesn't override you, open the speed cap, then send a command.

```bash
pkill -f mission_manager
ros2 topic pub -r 10 /mission/speed_cap std_msgs/msg/Float32 "{data: 0.5}" &
ros2 topic pub --once /behavior/cmd std_msgs/msg/String \
  "{data: '{\"id\": 1001, \"behavior\": \"wall_center\", \"params\": {\"speed\": 0.4, \"distance\": 2.0}}'}"
ros2 topic echo /behavior/status          # running -> done / failed
```

- Use a new `id` for every command; the executor ignores a repeated one.
- Behaviors: `stop heading_hold wall_center wall_trace gap_follow k_turn arc_turn plan_drive backoff`.
- Their parameters are the ones used in `mission.yaml`.
- To stop: send `{"id": 1002, "behavior": "stop", "params": {}}`, or e-stop, or kill the speed-cap publisher (`kill %1`).

## 5. Watch what it is doing

**Live state, one line every 0.5 s** (zone, step index, phase, race time):

```bash
ros2 topic echo --field data /mission/state
# {"started": true, "finished": false, "zone": "buckets", "step": 1, "phase": null, "elapsed": 112.4}
```

**The story of the run.** Filter the saved launch log (from section 2) in a second terminal:

```bash
tail -f ~/runs/<file>.log | grep --line-buffered -E \
  'RACE START|GREEN|Armed|=== ZONE|step [0-9]+|behavior <-|-> (done|failed)|marker|gap found|exit seen|gate at|gate rejected|stuck|timed out|fallback|terrain ->|tracking (lost|back)|E-STOP|FINISHED'
```

| Log line | Means |
|---|---|
| `GREEN at 1.8 m -> race start!` / `Armed: red light...` | Start light seen / start held by the red light |
| `=== ZONE hoops (t=141.4 s) ===` | The zone the mission thinks it is in |
| `step 1: behavior=plan_drive until=[...]` | The maneuver it is trying, and what ends it |
| `behavior <- wall_center {...}` / `wall_center -> done` | Executor started / finished a maneuver |
| `marker -> zone wide` | AprilTag close enough (≤1.5 m) to count |
| `marker for <zone>: jumping from <zone>` | A tag moved the mission forward |
| `left gap found at (x, y)` | Bucket field entrance located (the wide zone traces the left wall ~0.5 m off, forward only, until this appears) |
| `exit seen at (x, y)` / `step 2 skipped: corridor<=1.15` | Bucket exit found / already in the exit connector |
| `gate at ...` / `gate rejected: ...` | Hoop chosen / candidate ignored, with the reason |
| `stuck in <behavior>, backing off (retry n)` | No progress; reversing to retry |
| `zone <z> timed out` / `...; fallback -> ...` | Recovery paths |
| `terrain -> ramp_up` | Ramp / bank / rough ground detected (IMU) |
| `ZED tracking lost` / `back after X m` | Visual odometry dropout, dead-reckoning meanwhile |
| `E-STOP` / `FINISHED in X s` | |

**Single topics:**

| Question | Command |
|---|---|
| Which maneuver right now, and its result | `ros2 topic echo --field data /behavior/status` |
| Exact maneuver parameters | `ros2 topic echo --field data /behavior/cmd` |
| AprilTags in view (id, zone, distance, bearing) | `ros2 topic echo --field data /marker/detections` |
| Tag that triggered a zone | `ros2 topic echo /zone_event` |
| Start light fired? | `ros2 topic echo /race_start` |
| Terrain / tilt | `ros2 topic echo /terrain/state`, `/terrain/attitude` (roll°, nose-up°, vibration) |
| Lidar trusted? (false when tilted ≥4°) | `ros2 topic echo /terrain/lidar_ok` |
| Odometry degraded? | `ros2 topic echo /odom/degraded` |
| Openings and hoops seen | `ros2 topic echo /perception/left_gap` (also `right_gap`, `front_gap`, `gate`) |
| Final command to the motors | `ros2 topic echo /ackermann_cmd` |
| Everything alive? | `ros2 node list`, `ros2 topic hz /scan`, `ros2 topic hz /odom/fused` |

**Bench check before driving.** Run with `motor:=false`, carry the truck through the first zones by hand, and watch the log:
- `marker -> zone ...` should appear at each tag.
- `terrain ->` should change on the ramp.
- `=== ZONE` lines should advance.

## 6. Quick fixes

| Symptom | Check |
|---|---|
| Never starts | `/race_start` never published: light out of 0.3-4 m, or red in view. Start by hand (section 2). |
| Stops in a lane for no reason | Lidar sees part of the truck. Check `scan_self_mask` in `tuning.yaml` (flag pole bearing, half-width, max range) and `vehicle.lidar_to_front`. |
| Zone jumps early or late | A tag is visible from another zone. Check `/marker/detections`; move or angle the tag. |
| Ramp/bank never detected | It wasn't level at boot. Restart it still and level. |
| Wrong turn tightness | Measure the real turning radius and pass `turn_radius:=`. |
