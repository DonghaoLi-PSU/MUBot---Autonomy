# MUBot: ROS 2 + Gazebo Harmonic

Simulation, learning and (soon) autonomy for **MUBot**, a multi-link, magnetic-electrically actuated swimming robot.

This repository is a port of the original **ROS 1 + Gazebo Classic** MUBot project to **ROS 2 Jazzy + Gazebo Harmonic**. It then extends the project with additional controllers and sensors toward autonomous swimming.

> **Status:** Phase 1 (migration) is working. Phase 2 (sensors and autonomy) is in progress. See the [Roadmap](#roadmap).

---

## Goals

1. **Migrate** the ROS 1 / Gazebo Classic simulation, hydrodynamics plugins and EM-PGPE gait-learning pipeline to ROS 2 / Gazebo Harmonic.
2. **Extend** the platform with more controllers and onboard sensors so the robot can move from open-loop gaits to autonomous behavior.

---

## Features

- **Hydrodynamics plugin** (`mubot::HydroPlugin`). Per-link reactive (added-mass), resistive (friction and cross-flow drag) and compensation forces. Tail-tip pressure force. Wrench clamping to keep the simulation stable.
- **Actuator plugin** (`mubot::ActuatorPlugin`). Voltage-driven magnetic-electric joint actuators with back-EMF, plus optional explicit joint springs. Includes a total-power limit and voltage and torque clamps.
- **Drift metrics.** Computed over whole oscillation cycles: mean CoG velocity, max lateral offset, RMS lateral offset and heading drift.
- **Gait learning with EM-PGPE.** Learns amplitudes, phases and frequency per joint. Each rollout runs in a fresh, isolated simulator process, so many workers can run in parallel.
- **Data-driven robot description.** Morphology is set by `mubot.urdf` plus CSV parameter files in a per-robot `Database/<id>/` folder.
- **Headless and GUI modes**, with pause/step/unpause through a `ros_gz_bridge` world-control service.

---

## Repository layout

```
Mubot/                      # colcon workspace
└── src/
    ├── mubot_sim/              # Gazebo Harmonic simulation + plugins + learning scripts
    ├── mubot_control/          # controllers (CPG, closed-loop, ...)   [in progress]
    └── mubot_interfaces/       # custom msgs / srvs / actions          [in progress]
```

### Per-robot `Database/<id>/` folder

The plugins locate a robot's data from its **model name**: a model named `mubot_<id>` reads from `Database/<id>/`. The model name must therefore start with `mubot_`.

| File | Purpose |
|---|---|
| `mubot.urdf` | Links, joints, inertials and the two plugin tags |
| `joint_param.csv` | One row per joint: `name, spring_stiffness, actuator_constant, stiffness_ratio` |
| `hydro_param.csv` | One row per link: `name` plus 12 hydrodynamic coefficients (`M_total, M_S, M_0, M_l, cog, lt, h, IH0..IH3, IP`) |
| `control_signal.csv` | One line of `2N` values: `amp_0, (amp_i, phase_i)*(N-1), frequency` (for sinusoidal voltage)|
| `record/` | Output folder written by the plugins (displacements, forces, voltages, drift metrics) |

CSV rows must match the model's links and joints exactly, in both count and name. A mismatch disables the plugin and logs the reason.

---

## Requirements

- Ubuntu 24.04
- ROS 2 **Jazzy**
- Gazebo **Harmonic** (gz-sim 8)
- `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_interfaces`
- Python 3 with `numpy` and `pandas`
- A C++17 compiler and `colcon`

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
                 ros-jazzy-ros-gz-interfaces python3-pandas python3-numpy
```

---

## Build

```bash
mkdir -p ~/Mubot/src && cd ~/Mubot/src
git clone <this-repo-url> .
cd ~/Mubot
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` matters: `mubot.launch.py` resolves the `Database/` folder relative to the source tree and refuses to run from `install/`. Alternatively, pass `database_dir:=/abs/path` explicitly.

### Database path resolution

The plugins find the Database root in this order:

1. `<database_dir>` in the plugin's SDF/URDF block
2. The `MUBOT_DATABASE_DIR` environment variable
3. `<share>/des_mubot/Database` (only if `ament_index_cpp` is available)

---

## Quick start

**Open the world with the GUI** (no robot, useful for checking the install):

```bash
ros2 launch mubot_sim mubot.launch.py
```

**Spawn a robot, headless, paused:**

```bash
ros2 launch mubot_sim mubot.launch.py robot_name:=mubot_1 headless:=true run:=false
```

Launch arguments:

| Argument | Default | Description |
|---|---|---|
| `database_dir` | `<src>/Database` | Root folder containing `<id>/` subfolders |
| `robot_name` | `mubot_1` | Must start with `mubot_` |
| `urdf_path` | `Database/<id>/mubot.urdf` | Robot description to spawn |
| `world` | `worlds/mubot.world` | World file (must not already include the robot) |
| `headless` | `false` | `true` runs the server only |
| `run` | `true` | `false` starts paused |

---

## Gait learning (EM-PGPE)

```bash
cd ~/Mubot/src/mubot_sim/scripts

# train from scratch
python start_empgpe.py --wk-id 1

# resume an interrupted run
python start_empgpe.py --wk-id 1 --mode continue

# re-run a stored controller (Database/<id>/controller.csv)
python start_empgpe.py --wk-id 1 --mode rerun --morph-id 17

# watch it in the GUI
python start_empgpe.py --wk-id 1 --gui
```

How a rollout works:

1. `start_empgpe.py` writes `control_signal.csv` for the sampled parameters.
2. `MubotEnv` launches `mubot.launch.py` as its own process group, paused.
3. It waits for both plugins to report that their CSVs match the model.
4. It unpauses the simulation via the `/world/<name>/control` bridge.
5. It waits for `terminate.txt` from the Hydro plugin, then reads `record/drift_metrics.csv`.
6. It pauses, kills the whole process group and cleans up.

Reward, computed from the drift metrics `[cogvel_x, abs_lateral, rms_lat, heading_drift]`:

```
reward = -cogvel_x - max(abs_lateral - 0.1, 0) - 0.2 * max(heading_drift - 0.1, 0)
```

Each worker uses its own `GZ_PARTITION`, `ROS_DOMAIN_ID` and log directory, so several workers (`--wk-id 1`, `--wk-id 2`, ...) can run side by side.

---

## Physics notes

- The world is **zero-gravity** (neutral buoyancy) with ODE at a **0.25 ms** step.
- To improve numerical accuracy, fluid and robot densities are **scaled up 1000x**. Actuator torques are scaled by the same factor.
- The evaluation window is the largest integer number of oscillation periods that fits between `record_time` (4.0 s) and `terminate_time` (6.1 s).

---

## Migration notes (ROS 1 / Classic to ROS 2 / Harmonic)

| Classic + ROS 1 | Harmonic + ROS 2 |
|---|---|
| `ModelPlugin::Load` / `OnUpdate` | `ISystemConfigure` / `ISystemPreUpdate` |
| `gazebo::physics::Link` force APIs | `gz::sim::Link::AddWorldWrench` |
| `rospy` services for pause/unpause | `ros_gz_bridge` + `ControlWorld` service |
| `roslaunch` | `ros2 launch` (Python launch files) |
| Plugin discovery via `GAZEBO_PLUGIN_PATH` | `GZ_SIM_SYSTEM_PLUGIN_PATH` |
| Resources via `GAZEBO_MODEL_PATH` | `GZ_SIM_RESOURCE_PATH` |

---

## Roadmap

**Phase 1: Migration**
- [x] Actuator and hydrodynamics plugins ported to gz-sim 8
- [x] Launch files, world and spawn flow
- [x] EM-PGPE training loop on ROS 2
- [x] Validate the ported plugins against Classic results (velocity, lateral drift, forces)
- [ ] CI: build and a headless smoke-test rollout
- [ ] Docker image

**Phase 2: Control**
- [ ] Open-loop CPG controller node in `mubot_control`
- [ ] Steering and speed commands (`/cmd_vel`-style interface)
- [ ] Closed-loop heading and depth control
- [ ] Learned or optimized gait library, with a switcher

**Phase 3: Sensing**
- [ ] IMU
- [ ] Depth / pressure
- [ ] Flow / lateral-line style sensing
- [ ] Camera and/or sonar
- [ ] Joint-state publishing and `robot_state_publisher` / TF

**Phase 4: Autonomy**
- [ ] State estimation (EKF)
- [ ] Waypoint following
- [ ] Obstacle avoidance
- [ ] Behavior layer (behavior trees or state machine)
- [ ] Custom messages, services and actions in `mubot_interfaces`

---

## Troubleshooting

- **"Plugin disabled" or "CSV/URDF mismatch":** the number and names of rows in `joint_param.csv` / `hydro_param.csv` must exactly match the URDF joints and links. `control_signal.csv` must have `2 * num_joints` columns.
- **"Cannot locate Database directory":** set `<database_dir>`, `MUBOT_DATABASE_DIR`, or build with `--symlink-install`.
- **Rollout times out waiting for `terminate.txt`:** check `Database/<id>/sim.log`. The simulation may not have unpaused, or a plugin failed to load.
- **Unstable or exploding simulation:** reduce amplitudes, check the inertials in the URDF, and see the clamp constants in the plugins.

---

## Citation

If you use this work, please cite the original MUBot publication:

```
Li, D., Deng, H., Bayiz, Y. E., & Cheng, B. (2022). Effects of design and hydrodynamic parameters on optimized swimming for simulated, fish-inspired robots. 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7500–7506. https://doi.org/10.1109/IROS47612.2022.9981478
Li, D., Deng, H., Panta, K., Panah, A., & Cheng, B. (2026). System identification of robotic fish swimming reveals roles of reactive and resistive fluid force in two distinct swimming gaits. Advanced Intelligent Systems. https://doi.org/10.1002/aisy.202501503
```

## Authors

- **Donghao Li**: original MUBot (Gazebo Classic + ROS 1) and the ROS 2 port
- Contributors welcome. See `CONTRIBUTING.md`.

## License

<choose a license, e.g. Apache-2.0 or MIT>
