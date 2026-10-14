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
- [ ] Mosquitto runs (`systemctl is-active mosquitto`), the ESP8266 e-stop is powered, and
      `mosquitto_sub -v -t '#' -C 3 -W 10` prints its `esp/trigger {"heartbeat": .., "pin_state": ..}`
      messages. Nothing within 10 s: the ESP isn't reaching the broker. Mosquitto 2.x listens on
      localhost only by default; the ESP connects over Wi-Fi, so open it to the network once:
      `printf 'listener 1883 0.0.0.0\nallow_anonymous true\n' | sudo tee /etc/mosquitto/conf.d/esp.conf && sudo systemctl restart mosquitto`
      (team network only: anyone on it can publish).
- [ ] `python3 -c "import paho.mqtt"` works (else `sudo apt install python3-paho-mqtt`).

## 2. Run the whole course

**One command for everything** (sensors, full obstacle-course stack, dashboard with camera,
lidar, map, logs, Drive panel, E-STOP and map making):

```bash
source ~/turbotitans_auto/install/setup.bash && mkdir -p ~/runs
ros2 launch tt_obstacle_course obstacle_course.launch.py dash:=true turn_radius:=1.07 2>&1 | tee ~/runs/$(date +%m%d_%H%M).log
```

Open `http://<jetson-ip>:8080` on the laptop. Add `map:=<workspace>/maps/<map_dir>` to run on a
saved map (section 2a).

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
| `map:=<dir>` | none | Localize on a course map (section 2a) |
| `map_correct:=false` | true | With `map:=`, only localize and report; don't touch `/odom/fused` |
| `dash:=true` | false | Start the dashboard with Drive panel, E-STOP and map making |
| `maps_dir:=<dir>` | `<workspace>/maps` | Where the dashboard saves new maps |
| `hw_estop:=false` | true | Bench only: drive without the ESP8266 e-stop (see section 3) |

## 2a. Track day: odometry check, map, verify (trial: 30 min on the track)

Everything below runs from the dashboard (one command from section 2, race not started).
The odometry check comes first: the map is only as good as the odometry it is built on.

**Order of work on the track (about 25 min):**

| # | What | Time |
|---|---|---|
| 0 | Boot level and still at the start box: `ros2 launch ... dash:=true`, open the dashboard | 2 min |
| 1 | Odometry check (below); if the scale is off by >2%, set `odom_scale` and restart | 5 min |
| 2 | Mapping lap (below), check `map.png` | 7 min |
| 3 | Second mapping lap if the first map is bent (keep the better one) | 5 min |
| 4 | One check run on the map with `map_correct:=false`, watch the dashboard | 5 min |
| 5 | Copy `<workspace>/maps/` and `~/runs/` to the laptop | 1 min |

### Odometry check (Drive panel, "Odometry check" row)

Measures how far off the ZED's distance and heading are, on two straight legs with a known
turn between them. Teleop is capped at **0.3 m/s** while it runs (`tt_dash --check-speed`).

1. On a flat, open part of the floor lay **two tapes**: leg 1 straight, 5 m (measure it), and
   leg 2 starting near its end, **square to it** (90 deg, check with a set square or the
   course's own wall/floor grid), also 5 m. Enter the true lengths/turn if they differ.
2. **Start check.** The status line says what to do next:
   1. truck at the START of tape 1, lined up on it, **Mark**;
   2. drive slowly along the tape to its END, stop, **Mark**;
   3. turn onto tape 2 (a K-turn is fine), line up at its START, **Mark**;
   4. drive slowly to its END, stop, **Mark**.
3. The result (also in the log, and saved as `<workspace>/maps/odocheck_<time>.json`):
   - **scale** = true / odometry distance. 1.00 is perfect. If it is off by more than ~2%,
     put the suggested value in `tuning.yaml` (`odom_guard: odom_scale: ...`) and restart
     the launch. Run the check again: it should now read ~1.00.
   - **turn error** of the path, the ZED heading and the gyro vs the true 90 deg, and the
     **heading drift along each leg** (should be ~0 driving straight). A turn error over
     ~3 deg or drift over ~2 deg per 5 m: tell me the numbers (heading isn't auto-corrected).
   - "a leg was under 0.5 m": the truck didn't move enough between marks; redo it.

### Mapping lap

The map is a lidar picture of the course walls. With it the truck knows where it is even when
the ZED drifts (ramps, helix, tunnel, car wash): `map_localizer` matches each level scan to the
map and `odom_guard` pulls `/odom/fused` onto it, but only by a fifth of the distance driven
(plus 5 mm) per scan, so a running maneuver sees it as ordinary odometry error. It is paused in
`bank_approach`/`bank_uturn` (`map_freeze_zones` in tuning), where the fenced plan must not move.
Changed spots (moved buckets/hoops, shifted bucket entrance, extra bales) just don't match and
are ignored; the walls that stayed do the work.

**Mapping lap (~5 min).** Truck at the start box, facing in, exactly where races start, still.
Start the one command from section 2 (don't start the race), open the dashboard, then:

1. **Start map** (Drive panel). It records into `<workspace>/maps/map_<MMDD_HHMMSS>/`.
2. **Take control** and drive the lap with the arrow keys (or push the truck).
3. Mapping **stops by itself** back at the start/finish line: once 20 m+ were driven and the
   truck is within 2.5 m of where mapping began, it matches the scan against the start area,
   spreads the leftover drift back over the whole lap so start and finish meet
   (`map: back at the start, loop closed (...)` in the log), saves and stops.
   **Stop & save map** ends it any time (it is saved every 30 s anyway, and on shutdown).

The folder holds `map.png` (look at it), `map.npz` (what the stack loads), `map.pgm/yaml`
(Nav2 format) and `bag/` (the whole drive, to rebuild or update the map later).

What gets mapped: every wall the lidar sees while level. Ramps, the bridge, helix and bank are
driven blind (lidar gated), so their walls are missing, but odometry carries the path through
them, so the lap stays one connected map from start to finish.

- Drive **slowly** (~0.2-0.3 m/s), steady speed, gentle steering, no stops against walls, no
  wheel spin, no rocking: quick jerks are where camera odometry jumps.
- **Pause ~2 s at every lane end and corner**: walls across the lane fix how far along it the
  truck is (in a straight featureless lane only odometry knows that).
- In each **open area** (wide zone, bucket field, hoop U-turn): drive in centred, pause, turn
  ~90 deg each way so the lidar sees all four walls.
- **Ramps, bridge, helix, bank** (lidar off): slow and in one go, no stops or reversing on them.
- **Finish** by crossing the finish line slowly and stopping where mapping started: that is
  what closes the loop (`loop closed` in the log).
- Changeable items (buckets, hoops, extra bales): if allowed, map with them removed so the map
  holds only fixed walls; otherwise map as they are and refresh those zones later (`tt_map update`).
- **Check `map.png`**: lanes straight and square, the tunnel lane a clean
  corridor, no doubled walls, start and finish meet. If it is bent, drive a second lap (keep the
  better map) or rebuild from the bag (below).
- Two laps agree? `tt_map eval <lap 1 map> <lap 2 map>/bag`: quality median > 0.8 = consistent.
- Rebuild any time without the truck: `ros2 run tt_obstacle_course tt_map build <map_dir>/bag -o course2`

(Without the dashboard: `tt_map live -o course` does the same from a terminal.)

**Check it before trusting it (1 run).** Same start box:

```bash
ros2 launch tt_obstacle_course obstacle_course.launch.py dash:=true map:=<map_dir> map_correct:=false 2>&1 | tee ~/runs/$(date +%m%d_%H%M).log
```

In tt_dash's state panel, `map localization` should read `tracking`, quality > 0.7 on level
ground (it says `odometry only` on ramps/bank/helix: expected), and the green scan should sit on
the white map walls. Afterwards: `tt_map eval ~/maps/course ~/runs/bag_<that run>` (record it).

**Use it.** Same command with `map_correct:=true` (the default). The log shows
`map frame locked: ...` once at the start.

**Partial runs** (`start_zone:=`) work too: if the scan doesn't match at the start pose the
localizer searches the mapped drive for where it is (a few seconds, truck still).

**Maps are plain files** (`~/maps/<name>/`: `map.npz` for the stack, `map.pgm` + `map.yaml` in
Nav2 format, `map.png` to look at). Copy, rename or keep several; any of them runs with
`map:=~/maps/<name>`. `tt_map show <dir>` prints what's in one.

**Refresh the changeable zones from any run** (practice or heat). Record it with `tt_record`, then:

```bash
ros2 run tt_obstacle_course tt_map update ~/maps/course ~/runs/bag_<run> --zones buckets wide hoops -o course_heat1
```

The run is localized on the old map, and only cells inside those zones (zone names as labelled
on the mapping drive) are redone from it: moved buckets, the shifted entrance, extra bales,
hoops. Everything else and the original map stay as they were. Prints how much changed. In the
sim, updating from a run with moved buckets/hoops cut the hoop-zone error from 0.38 to 0.08 m.

**If anything looks wrong:** drop `map:=` and the stack is exactly as before.

**Not in the map / limits:**
- Ramps, bridge, helix, bank: lidar gated; odometry carries the pose, corrected again within ~1 s
  of level ground.
- The map is one level: the tunnel under the bridge is mapped, the bridge deck isn't.

## 3. Stop

| What | Command |
|---|---|
| **E-stop** (speed 0, latched) | `ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"` |
| Release e-stop | `ros2 topic pub --once /estop std_msgs/msg/Bool "{data: false}"` |
| Quit everything | `Ctrl-C` in the launch terminal |

- E-stop zeroes speed but holds the steering where it is.

**Hardware e-stop (ESP8266).** The ESP sends `{"heartbeat": int, "pin_state": int}` on MQTT
`esp/trigger` to mosquitto on the Jetson. `estop_bridge` turns it into `/estop/hw` (shown as
`hw e-stop` in the dashboard state panel) and the speed governor refuses to move unless it says RUN:
- STOP when `pin_state` is `stop_pin_state` (0, in `tuning.yaml`; check on the bench that pressing
  the button shows STOP), `heartbeat` is 0, or nothing arrived for 2.5 s (radio lost).
- Once it went RUN -> STOP it **stays latched** even if the button comes back: put the button back
  to RUN, then press **Release** on the dashboard (or `/estop` false above).
- The STOP it shows at boot (before the ESP is heard) is not latched.
- It stops teleop too. Bench work without the ESP: `hw_estop:=false`.
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
| `ros2 run tt_obstacle_course tt_dash --control` | Adds a **Drive** panel: "Take control", then arrow keys drive (space = stop, Esc = hand back), plus an E-STOP button. Commands go to `/teleop_cmd`; the speed governor gives them priority but keeps collision stops, e-stop, rollover and acceleration limits (`teleop_max_speed`, default 1.0 m/s). Perception, mission and logging keep running; 0.3 s after you hand back (or the page loses focus / connection) the stack drives again. Anyone who can open the page can drive: use it on the team network only. |
```

- Use a new `id` for every command; the executor ignores a repeated one.
- Behaviors: `stop heading_hold wall_center wall_trace gap_follow k_turn arc_turn plan_drive backoff`.
- Their parameters are the ones used in `mission.yaml`.
- To stop: send `{"id": 1002, "behavior": "stop", "params": {}}`, or e-stop, or kill the speed-cap publisher (`kill %1`).

## 5. Watch what it is doing

**Debug tools** (all read-only; start them before, during or after any run: full, `start_zone:=`, one zone, hand-sent behavior, `motor:=false` bench, sim, or a replayed bag):

| Command | What you get |
|---|---|
| `ros2 run tt_obstacle_course tt_dash` | Browser dashboard at `http://<jetson-ip>:8080`: camera with tag boxes and HUD, lidar (forward = up, self-mask grey, steering arc), map trail coloured by zone, live state (zone/step/maneuver/commands/caps/terrain/topic rates), and the whole log as one filterable timeline. Saves every event to `~/runs/dash_<time>.jsonl`. |
| `ros2 run tt_obstacle_course tt_cam` | Same, camera first: for placing tags. Each tag shows id, zone, distance, bearing, face angle, size in px and why it is rejected (too far / too small / unknown id). Green = would trigger its zone. |
| `ros2 run tt_obstacle_course tt_lidar` | Same, lidar and map first. |
| `ros2 run tt_obstacle_course tt_dash --log ~/runs/<file>.log` | A saved run's timeline in the browser (`--follow` while it is still growing; `--no-ros` on a laptop without ROS). |
| `ros2 run tt_obstacle_course tt_log ~/runs/<file>.log` | The same uniform timeline in the terminal. `-f` follows, `--zone buckets`, `--summary` per-zone table, `--all` includes chatter, `--grep 'gap|exit'`. Also reads `dash_*.jsonl`. Pure Python: works on a laptop. |
| `ros2 run tt_obstacle_course tt_record [--camera]` | Records a bag to `~/runs/bag_<time>`. Replay later with `ros2 bag play <bag>` + `tt_dash` and it looks exactly like live. Camera adds ~10-15 MB/s. |
| `ros2 run tt_obstacle_course tt_mission wide buckets` | Writes a cut-down mission and prints the launch + start commands. `--list` shows zones and steps. |
| `ros2 run tt_obstacle_course tt_map live / build / show / eval` | Course map: make it while driving or from a bag, redraw `map.png`, or replay a bag through the localizer and report match quality (section 2a). |

The camera view costs CPU only while a browser has it open (the image is not even subscribed otherwise), and its tag boxes come from the running `marker_detector`. Close the camera panel during timed runs.

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
