# -*- coding: utf-8 -*-
"""
Original (ROS 1) and migrating (ROS 2) by donghao Li.
"""

import atexit
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node

from gazebo_connection import GazeboConnection
from mubot_step_client import StepClient, StepError

DT = 0.00025            # physics step, must equal max_step_size in mubot.world
CLAMP_FREQ = 20.0       # Hz, the value Hydro uses for its force clamp limits

READY_MARKERS = ("[Actuator] CSV and URDF matched",
                 "[Hydro] CSV and URDF matched")
FAIL_MARKERS = ("plugin disabled",
                "Failed to load system plugin",
                "Could not find shared library")


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _count_joints(folder):
    with open(Path(folder) / 'joint_param.csv') as f:
        return sum(1 for line in f if line.strip())


def _default_domain(wk_id):
    s = str(wk_id)
    return int(s) % 100 if s.isdigit() else 0


def drift_metrics_from_obs(traj, t_start=4.0, t_end=6.0):
    """[cogvel_x, abs_lateral, rms_lat, heading_drift] over a time window of a list of Obs."""
    w = [o for o in traj if t_start - 1e-9 <= o.sim_time <= t_end + 1e-9]
    if len(w) < 2:
        return np.full(4, np.nan)
    cog = np.array([o.cog_xy for o in w])
    fwd = np.array([w[0].head_pose[0] - w[0].tail_xy[0],
                    w[0].head_pose[1] - w[0].tail_xy[1]])
    nrm = np.linalg.norm(fwd)
    fwd = fwd / nrm if nrm > 1e-12 else np.array([1.0, 0.0])
    left = np.array([-fwd[1], fwd[0]])
    lat = (cog - cog[0]) @ left
    yaw = np.unwrap([o.cog_yaw for o in w])
    return np.array([np.mean([o.cog_vel[0] for o in w]),
                     np.max(np.abs(lat)),
                     np.sqrt(np.mean(lat ** 2)),
                     abs(yaw[-1] - yaw[0])])


def episode_reward(m):
    return float(-m[0] - np.clip(m[1] - 0.1, 0.0, None)
                 - np.clip(m[3] - 0.1, 0.0, None) * 0.2)


class MubotGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, worker_folder, wk_id=1, n_joints=None, ctrl_steps=40, episode_time=6.0,
                 v_max=15.0, headless=True, use_clock=True, clock_freq=2.0,
                 lat_free=0.1, yaw_free=0.1, w_lat=1.0, w_yaw=0.2, w_act=0.0,
                 window_start=4.0, max_retry=2, restart_every=300,
                 world_name='default', domain_id=None,
                 startup_timeout=60.0, reset_timeout=10.0):
        super().__init__()
        print("\33[92mInitializing MubotGymEnv\033[0m")
        folder = Path(worker_folder).resolve()
        if folder.name != str(wk_id):
            raise ValueError(f"worker folder '{worker_folder}' must be named '{wk_id}' "
                             f"(plugins resolve Database/<robot_name minus 'mubot_'>)")
        self.folder = folder
        self.worker_folder = str(folder) + "/"
        self.database_dir = str(folder.parent)
        self.wk_id = wk_id
        self.robot_name = f"mubot_{wk_id}"
        self.n = int(n_joints) if n_joints else _count_joints(folder)

        # task settings
        self.ctrl_steps = int(ctrl_steps)
        self.episode_time = float(episode_time)
        self.window_start = float(window_start)
        self.v_max = float(v_max)
        self.use_clock = bool(use_clock)
        self.clock_freq = float(clock_freq)
        self.lat_free, self.yaw_free = float(lat_free), float(yaw_free)
        self.w_lat, self.w_yaw, self.w_act = float(w_lat), float(w_yaw), float(w_act)
        self.max_retry = int(max_retry)
        self.ctrl_dt = self.ctrl_steps * DT
        self.max_k = int(round(self.episode_time / self.ctrl_dt))

        # simulator settings
        self.headless = bool(headless)
        self.world_name = world_name
        self.startup_timeout = float(startup_timeout)
        self.reset_timeout = float(reset_timeout)
        self.restart_every = int(restart_every)
        self.domain_id = _default_domain(wk_id) if domain_id is None else int(domain_id)
        self.ros_log_dir = f"/dev/shm/roslog_worker_{wk_id}"
        os.makedirs(self.ros_log_dir, exist_ok=True)

        os.environ['ROS_DOMAIN_ID'] = str(self.domain_id)
        if not rclpy.ok():
            rclpy.init(domain_id=self.domain_id)
        self.node = Node(f"mubot_env_{wk_id}")
        self._conn = None            # GazeboConnection: pause / unpause / reset
        self._client = None          # StepClient: lock-step service
        self._proc = None
        self._log = None
        self._n_since_start = 0
        atexit.register(self.shutdown)

        # Gym spaces
        dim = 3 * self.n + 8 + (2 if self.use_clock else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self._last_vec = np.zeros(dim, np.float32)
        self._traj = []
        self._k = 0

        self._write_clamp_file()
        self._sweep_stale()
        print("\33[92mMubotGymEnv initialized\033[0m")

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = None
        for attempt in range(self.max_retry + 1):
            try:
                obs = self._begin_episode()
                break
            except (RuntimeError, TimeoutError) as e:
                print(f"\033[91m[MubotGymEnv {self.wk_id}] reset failed "
                      f"(attempt {attempt + 1}/{self.max_retry + 1}): {e}\033[0m")
                self._stop_sim()
        if obs is None:
            raise RuntimeError("simulator could not be reset")

        # start frame: forward = unit vector tail -> head
        fwd = np.array([obs.head_pose[0] - obs.tail_xy[0], obs.head_pose[1] - obs.tail_xy[1]])
        nrm = np.linalg.norm(fwd)
        self._f0 = fwd / nrm if nrm > 1e-12 else np.array([1.0, 0.0])
        self._l0 = np.array([-self._f0[1], self._f0[0]])
        self._cog0 = np.array(obs.cog_xy, dtype=float)
        self._yaw0 = float(obs.cog_yaw)
        self._yaw_prev = float(obs.cog_yaw)
        self._yaw_unw = float(obs.cog_yaw)
        self._prev_a = np.zeros(self.n)
        self._traj = [obs]
        self._k = 0
        vec = self._vec(obs, 0.0)
        self._last_vec = vec
        return vec, {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        try:
            obs = self._client.step(a * self.v_max, self.ctrl_steps)
        except (RuntimeError, TimeoutError, AttributeError) as e:
            print(f"\033[91m[MubotGymEnv {self.wk_id}] step failed: {e}\033[0m")
            self._stop_sim()
            return (self._last_vec, 0.0, False, True,
                    {"sim_failed": True, "orig_reward": float("nan")})

        self._traj.append(obs)
        self._k += 1
        yaw_rel = self._track_yaw(obs)
        v_f = float(np.dot(obs.cog_vel, self._f0))
        lat = float(np.dot(np.asarray(obs.cog_xy) - self._cog0, self._l0))
        reward = (v_f
                  - self.w_lat * max(abs(lat) - self.lat_free, 0.0)
                  - self.w_yaw * max(abs(yaw_rel) - self.yaw_free, 0.0)
                  - self.w_act * float(np.sum((a - self._prev_a) ** 2)))
        self._prev_a = a
        vec = self._vec(obs, yaw_rel)
        self._last_vec = vec

        info = {}
        truncated = self._k >= self.max_k
        if truncated:
            info["sim_failed"] = False
            try:
                self._finish_episode()
            except (RuntimeError, TimeoutError) as e:
                print(f"\033[91m[MubotGymEnv {self.wk_id}] finish failed: {e}\033[0m")
                self._stop_sim()
                info["sim_failed"] = True
            stats = drift_metrics_from_obs(self._traj, self.window_start, self.episode_time)
            info["drift"] = stats
            info["orig_reward"] = episode_reward(stats) if np.all(np.isfinite(stats)) else float("nan")
        return vec, float(reward), False, truncated, info

    def close(self):
        self.shutdown()

    # ==================================================================
    # episode start / end
    # ==================================================================
    def _begin_episode(self):
        """Fresh simulator if needed, world reset, unpause, first observation."""
        self._remove_files('reset_ack.txt')
        self._ensure_sim()
        self._reset_world()
        if not self._conn.unpause_sim():
            raise RuntimeError("could not unpause the simulation")
        return self._first_observation()

    def _finish_episode(self):
        """Free the blocked sim thread (step gate), then pause."""
        self._client.release()
        self._conn.pause_sim()
        self._n_since_start += 1

    def _first_observation(self, timeout=5.0):
        """After unpause the plugin may still report 'paused' (status 4) for a moment."""
        if self._client is None:
            self._client = StepClient(self.robot_name, self.n, partition=f"mubot_{self.wk_id}")
        self._client.forget_iteration()
        t0 = time.monotonic()
        while True:
            try:
                return self._client.observe()
            except StepError as e:
                if e.status != 4 or time.monotonic() - t0 > timeout:
                    raise
                time.sleep(0.05)

    # ==================================================================
    # simulator process
    # ==================================================================
    def _child_env(self):
        env = os.environ.copy()
        env.update({
            'GZ_PARTITION': f"mubot_{self.wk_id}",
            'ROS_DOMAIN_ID': str(self.domain_id),
            'ROS_LOG_DIR': self.ros_log_dir,
            'MUBOT_DATABASE_DIR': self.database_dir,
            'PYTHONUNBUFFERED': '1',            # so sim.log fills in real time
        })
        return env

    def _start_sim(self):
        self._sweep_stale()
        cmd = ['ros2', 'launch', 'mubot_sim', 'mubot.launch.py',
               f'robot_name:={self.robot_name}',
               f'urdf_path:={self.worker_folder}mubot.urdf',
               f'database_dir:={self.database_dir}',
               f'headless:={"true" if self.headless else "false"}',
               'run:=false']
        self._log = open(self.worker_folder + 'sim.log', 'w')
        self._proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=self._child_env(),
            start_new_session=True)
        with open(self.worker_folder + 'sim.pgid', 'w') as f:
            f.write(str(self._proc.pid))

    def _wait_ready(self):
        """Both plugins loaded their CSVs (the model is spawned and configured)."""
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            text = self._read_log()
            for m in FAIL_MARKERS:
                if m in text:
                    raise RuntimeError(f"sim startup failed ('{m}'), see {self.worker_folder}sim.log")
            if all(m in text for m in READY_MARKERS):
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"ros2 launch exited early (code {self._proc.returncode}), "
                                   f"see {self.worker_folder}sim.log")
            time.sleep(0.1)
        raise TimeoutError(f"sim not ready after {self.startup_timeout:.0f}s, "
                           f"see {self.worker_folder}sim.log")

    def _ensure_sim(self):
        """Start the simulator only if it is not already running."""
        if (self._proc is not None and self._proc.poll() is None
                and self._conn is not None and self._n_since_start < self.restart_every):
            return
        self._stop_sim()
        self._start_sim()
        self._wait_ready()
        self._conn = GazeboConnection(self.node, self.world_name,
                                      timeout=5.0, connect_timeout=30.0)
        self._n_since_start = 0

    def _reset_world(self):
        pos = self._log_size()
        if not self._conn.reset_world():
            raise RuntimeError("world reset request failed")
        ack = self.worker_folder + 'reset_ack.txt'
        t0 = time.monotonic()
        while not os.path.exists(ack):
            if 'plugin disabled' in self._log_since(pos):
                raise RuntimeError("plugin failed during reset, see sim.log")
            if self._proc.poll() is not None:
                raise RuntimeError("sim died during reset, see sim.log")
            if time.monotonic() - t0 > self.reset_timeout:
                raise TimeoutError(f"no reset ack after {self.reset_timeout:.0f}s")
            time.sleep(0.02)

    def _stop_sim(self):
        """Kill the whole simulator process group. Safe to call repeatedly."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        proc, self._proc = self._proc, None
        if proc is not None:
            pgid = proc.pid
            for sig, wait in ((signal.SIGINT, 8.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._cleanup_ros_logs()
        self._remove_files('sim.pgid')
        if self._log is not None:
            self._log.close()
            self._log = None

    def shutdown(self):
        """Stop everything this env owns. Safe to call repeatedly (also runs at exit)."""
        self._stop_sim()
        if getattr(self, 'node', None) is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
            if rclpy.ok():
                rclpy.shutdown()

    def _sweep_stale(self):
        """Kill a simulator left over from a crashed earlier run (pid in sim.pgid)."""
        p = self.worker_folder + 'sim.pgid'
        try:
            pgid = int(open(p).read())
            if b'mubot.launch.py' in open(f'/proc/{pgid}/cmdline', 'rb').read():
                os.killpg(pgid, signal.SIGKILL)
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            pass
        self._remove_files('sim.pgid')

    # ==================================================================
    # observation
    # ==================================================================
    def _track_yaw(self, obs):
        """CoG yaw relative to the episode start, unwrapped over time."""
        self._yaw_unw += _wrap(float(obs.cog_yaw) - self._yaw_prev)
        self._yaw_prev = float(obs.cog_yaw)
        return self._yaw_unw - self._yaw0

    def _vec(self, obs, yaw_rel):
        hyaw = float(obs.head_pose[2])
        c, s = np.cos(hyaw), np.sin(hyaw)
        hv = obs.head_vel
        vxb = c * hv[0] + s * hv[1]                       # head velocity in the head frame
        vyb = -s * hv[0] + c * hv[1]
        cv = np.asarray(obs.cog_vel, dtype=float)
        v_f = float(np.dot(cv, self._f0))
        v_l = float(np.dot(cv, self._l0))
        lat = float(np.dot(np.asarray(obs.cog_xy) - self._cog0, self._l0))
        parts = [np.asarray(obs.joint_pos) / (np.pi / 2.0),
                 np.asarray(obs.joint_vel) / 10.0,
                 [np.sin(yaw_rel), np.cos(yaw_rel)],
                 [vxb, vyb, hv[2]],
                 [v_f, v_l, lat / 0.1],
                 self._prev_a]
        if self.use_clock:
            ph = 2.0 * np.pi * self.clock_freq * float(obs.sim_time)
            parts.append([np.sin(ph), np.cos(ph)])
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    # ==================================================================
    # helpers
    # ==================================================================
    def _write_clamp_file(self):
        """Hydro reads the last value of control_signal.csv as the clamp frequency."""
        vals = [1.0] * (2 * self.n - 1) + [CLAMP_FREQ]
        with open(self.worker_folder + 'control_signal.csv', 'w') as f:
            f.write(','.join(f'{v:.8e}' for v in vals) + '\n')

    def _read_log(self):
        try:
            with open(self.worker_folder + 'sim.log', 'r', errors='replace') as f:
                return f.read()
        except FileNotFoundError:
            return ''

    def _log_size(self):
        try:
            return os.path.getsize(self.worker_folder + 'sim.log')
        except FileNotFoundError:
            return 0

    def _log_since(self, pos):
        try:
            with open(self.worker_folder + 'sim.log', 'rb') as f:
                f.seek(pos)
                return f.read().decode(errors='replace')
        except FileNotFoundError:
            return ''

    def _remove_files(self, *names):
        for n in names:
            try:
                os.remove(self.worker_folder + n)
            except FileNotFoundError:
                pass

    def _cleanup_ros_logs(self):
        """ROS/launch logs of this worker live in tmpfs; drop them whenever the sim is stopped."""
        shutil.rmtree(self.ros_log_dir, ignore_errors=True)
        os.makedirs(self.ros_log_dir, exist_ok=True)
