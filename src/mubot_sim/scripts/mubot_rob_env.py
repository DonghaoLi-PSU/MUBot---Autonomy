# -*- coding: utf-8 -*-
"""
MubotEnv for ROS 2 (Jazzy) + Gazebo Harmonic (gz-sim8).

A persistent simulator that is stepped in lock-step from Python. The simulator is
started once and reused; each rollout resets the world instead of relaunching it.

Rollout life cycle
  reset(n_joints)   start the simulator if needed (ros2 launch mubot_arg.launch.py,
                    started paused), reset the world and wait for reset_ack.txt, unpause,
                    return the first observation
  advance(volts, n) apply `volts` for n physics steps through the gz-transport service
                    /model/<robot_name>/step; the sim advances exactly n steps and the
                    reply is the observation after the last one (one call = one control step)
  finish()          release the plugin's step gate, then pause the world
  abort()           kill the simulator process group; the next reset() starts a clean one

Observation (Obs, see mubot_step_client.py)
  sim_time, iteration, joint positions / velocities, head pose and velocity, and the
  mass-weighted CoG position / velocity / yaw plus the tail position.

Reward
  drift_metrics_from_obs() computes [cogvel_x, abs_lateral, rms_lat, heading_drift]
  from a list of observations over the evaluation window eval_window() returns
  (4.0 s to the last full control period before 6.0 s). reward() turns them into a
  scalar. The same metrics are still written by the Hydro plugin to
  record/drift_metrics.csv; read_metrics() reads that file (kept for parity checks).

Notes
  - DT must equal max_step_size in mubot.world; CTRL_STEPS is the number of physics
    steps per control step (e.g. 40 steps * 0.25 ms = 10 ms).
  - Each worker has its own gz partition (GZ_PARTITION=mubot_<wk_id>) and ROS domain, and
    the worker folder must be Database/<wk_id>/ (plugins resolve their CSV files from it).
  - The simulator runs in its own process group (killable as a unit); its output goes to
    Database/<wk_id>/sim.log, and the process group id to sim.pgid for stale-process cleanup.
  - Needs the launch file to accept `run:=false` and to start the world-control bridge.

Original (ROS 1) by Donghao Li.
"""

import atexit
import os
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node

from gazebo_connection import GazeboConnection

READY_MARKERS = ("[Actuator] CSV and URDF matched",
                 "[Hydro] CSV and URDF matched")
FAIL_MARKERS = ("plugin disabled",
                "Failed to load system plugin",
                "Could not find shared library")

from mubot_step_client import StepClient, StepError

DT = 0.00025          # must equal max_step_size in mubot.world
CTRL_STEPS = 1          # physics steps per service call
MAX_SIM_TIME = 6.5    # give up 


def _default_domain(wk_id):
    s = str(wk_id)
    return int(s) % 100 if s.isdigit() else 0

def sinusoid_voltages(control, first_iter, n_steps):
    """Same waveform the plugin used in open-loop mode, one row per physics step."""
    raw = np.asarray(control, dtype=float)
    n = raw.size // 2
    amp = np.empty(n); phase = np.zeros(n)
    amp[0] = abs(raw[0])
    for i in range(1, n):
        amp[i] = abs(raw[2 * i - 1]); phase[i] = raw[2 * i]
    freq = abs(raw[2 * n - 1])
    if amp.sum() > 10000.0:
        amp *= 10000.0 / amp.sum()
    t = (first_iter + np.arange(n_steps)) * DT      # sim time of each step
    v = amp[None, :] * np.sin(2 * np.pi * (freq * t[:, None] + phase[None, :]))
    return np.clip(v, -15.0, 15.0)


def drift_metrics_from_obs(traj, t_start=4.0, t_end=6.1):
    """[cogvel_x, abs_lateral, rms_lat, heading_drift] from a list of Obs."""
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

def eval_window(freq, t_rec=4.0, t_term=6.1):
    n = int(np.floor((t_term - t_rec) * freq))
    return t_rec, t_rec + n / freq


class MubotEnv:

    def __init__(self, worker_folder, max_time, wk_id=1, headless=True,
                 world_name='default', domain_id=None,
                 startup_timeout=60.0, rollout_timeout=10.0):
        print("\33[92mInitializing MubotEnv\033[0m")
        self._conn = None
        self._n_since_start = 0
        self.restart_every = 300      # safety: restart the server every N rollouts
        self.reset_timeout = 10.0
        self.worker_folder = os.path.abspath(worker_folder) + "/"   # Database/<wk_id>/
        self.database_dir = os.path.dirname(os.path.abspath(worker_folder))
        self.max_sim_time = max_time
        self.wk_id = wk_id
        self.robot_name = f"mubot_{wk_id}"
        self.headless = headless
        self.world_name = world_name
        self.startup_timeout = startup_timeout
        self.rollout_timeout = rollout_timeout

        if os.path.basename(os.path.abspath(worker_folder)) != str(wk_id):
            raise ValueError(
                f"worker folder '{worker_folder}' must be named '{wk_id}' "
                f"(plugins resolve Database/<robot_name minus 'mubot_'>)")

        self.domain_id = _default_domain(wk_id) if domain_id is None else int(domain_id)
        self.ros_log_dir = f"/dev/shm/roslog_worker_{wk_id}"
        os.makedirs(self.ros_log_dir, exist_ok=True)

        os.environ['ROS_DOMAIN_ID'] = str(self.domain_id)
        if not rclpy.ok():
            rclpy.init(domain_id=self.domain_id)
        self.node = Node(f"mubot_env_{wk_id}")
        self._client = None 
        self._proc = None
        self._log = None
        atexit.register(self.shutdown)
        print("\33[92mMubotEnv initialized\033[0m")
        self._sweep_stale()

    def _first_observation(self, n_joints=0,timeout=5.0):
        """After unpause the plugin may still report 'paused' for a moment."""
        if n_joints:
            self._n_joints = n_joints
        if self._client is None:
            self._client = StepClient(self.robot_name, 0 + self._n_joints, partition=f"mubot_{self.wk_id}")
        self._client.forget_iteration()
        t0 = time.monotonic()
        while True:
            try:
                return self._client.observe()
            except StepError as e:
                if e.status != 4 or time.monotonic() - t0 > timeout:
                    raise
                time.sleep(0.05)

    def _sweep_stale(self):
        p = self.worker_folder + 'sim.pgid'
        try:
            pgid = int(open(p).read())
            if b'mubot_arg.launch.py' in open(f'/proc/{pgid}/cmdline', 'rb').read():
                os.killpg(pgid, signal.SIGKILL)
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            pass
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    # ------------------------------------------------------------------
    # simulator process
    # ------------------------------------------------------------------
    def _child_env(self):
        env = os.environ.copy()
        env.update({
            'GZ_PARTITION': f"mubot_{self.wk_id}",
            'ROS_DOMAIN_ID': str(self.domain_id),
            'ROS_LOG_DIR': self.ros_log_dir,
            'MUBOT_DATABASE_DIR': self.database_dir,
            'PYTHONUNBUFFERED': '1',          # so sim.log fills in real time
        })
        return env

    def _start_sim(self):
        self._sweep_stale()
        cmd = ['ros2', 'launch', 'mubot_sim', 'mubot_arg.launch.py',
               f'robot_name:={self.robot_name}',
               f'urdf_path:={self.worker_folder}mubot.urdf',
               f'database_dir:={self.database_dir}',
               f'headless:={"true" if self.headless else "false"}',
               'run:=false']                  # start paused; we unpause explicitly
        self._log = open(self.worker_folder + 'sim.log', 'w')
        self._proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=self._child_env(),
            start_new_session=True)           # own process group -> killable as a unit
        with open(self.worker_folder + 'sim.pgid', 'w') as f:
            f.write(str(self._proc.pid))


    def _read_log(self):
        try:
            with open(self.worker_folder + 'sim.log', 'r', errors='replace') as f:
                return f.read()
        except FileNotFoundError:
            return ''

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
                raise RuntimeError(
                    f"ros2 launch exited early (code {self._proc.returncode}), "
                    f"see {self.worker_folder}sim.log")
            time.sleep(0.1)
        raise TimeoutError(f"sim not ready after {self.startup_timeout:.0f}s, see {self.worker_folder}sim.log")

    def _wait_terminate(self):
        "terminate.txt may not exist"
        term = self.worker_folder + 'terminate.txt'
        t0 = time.monotonic()
        while not os.path.exists(term):
            if self._proc.poll() is not None:
                raise RuntimeError("sim died during rollout, see sim.log")
            if time.monotonic() - t0 > self.rollout_timeout:
                raise TimeoutError(f"no terminate.txt after {self.rollout_timeout:.0f}s")
            time.sleep(0.02)
        if not os.path.exists(self.worker_folder + '/record/drift_metrics.csv'):
            raise RuntimeError("terminate.txt exists but drift_metrics.csv is missing")

    def _stop_sim(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        proc, self._proc = self._proc, None
        if proc is not None:
            pgid = proc.pid                   # start_new_session => pgid == pid
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
        try:
            os.remove(self.worker_folder + 'sim.pgid')
        except FileNotFoundError:
            pass
        if self._log is not None:
            self._log.close()
            self._log = None


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
    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def step(self, control) -> bool:
        """Run one rollout through the step service. Returns True if drift_metrics.csv is ready."""
        self._remove_files( 'reset_ack.txt')
        # use the same 9-digit values the CSV holds, so Python and the plugin agree
        control = np.array([float('%.8e' % x) for x in np.asarray(control, dtype=float)])
        self._n_joints = control.size // 2
        ok = False
        try:
            self._ensure_sim()
            self._reset_world()
            if not self._conn.unpause_sim():
                raise RuntimeError("could not unpause the simulation")
            obs = self._first_observation()
            it, done = obs.iteration, 0
            max_steps = int(MAX_SIM_TIME / DT)
            term = self.worker_folder + 'terminate.txt'
            # while not os.path.exists(term):
            while done < max_steps:
                if done >= max_steps:
                    raise TimeoutError("no terminate.txt within the sim-time limit")
                n = min(CTRL_STEPS, max_steps - done)
                self._client.step(sinusoid_voltages(control, it + done, n), n)
                done += n
            self._client.release()
            self._conn.pause_sim()          # also stops the plugin's idle timeout
            if not os.path.exists(self.worker_folder + '/record/drift_metrics.csv'):
                raise RuntimeError("terminate.txt exists but drift_metrics.csv is missing")
            self._n_since_start += 1
            ok = True
        except (RuntimeError, TimeoutError) as e:     # StepError is a RuntimeError
            print(f"\33[91m[MubotEnv {self.wk_id}] rollout failed: {e}\33[0m")
            self._stop_sim()
        return ok

    def reset(self, n_joints):
        """Start a rollout: clean files, reset world, unpause. Returns the first obs."""
        self._remove_files('reset_ack.txt')
        self._n_joints = n_joints
        self._ensure_sim()
        self._reset_world()
        if not self._conn.unpause_sim():
            raise RuntimeError("could not unpause the simulation")
        return self._first_observation()

    def advance(self, volts, n_steps=CTRL_STEPS):
        """Apply volts for n_steps physics steps, return the obs after the last one.
        volts: (n_joints,) held for n_steps, or (n_steps, n_joints) per-step."""
        return self._client.step(volts, n_steps)

    def finish(self):
        """End of rollout: free the blocked sim thread, then pause."""
        self._client.release()
        self._conn.pause_sim()
        self._n_since_start += 1          

    def abort(self):
        self._stop_sim()                  # next reset() starts a clean sim

    def reward(self, metrics=None):
        m = self.read_metrics() if metrics is None else metrics
        return float(-m[0] - np.clip(m[1] - 0.1, 0.0, None)
                     - np.clip(m[3] - 0.1, 0.0, None) * 0.2)

    def cleanup_outputs(self):
        self._remove_files('control_signal.csv')

    def close(self):
        """Per-worker cleanup, same role as the ROS 1 version."""
        self.cleanup_outputs()

    def shutdown(self):
        """Stop everything this env owns. Safe to call repeatedly."""
        self._stop_sim()
        if getattr(self, 'node', None) is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
            if rclpy.ok():
                rclpy.shutdown()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _remove_files(self, *names):
        for n in names:
            try:
                os.remove(self.worker_folder + n)
            except FileNotFoundError:
                pass

    def _cleanup_ros_logs(self):
        """ROS/launch logs of this worker live in tmpfs; drop them after every rollout."""
        shutil.rmtree(self.ros_log_dir, ignore_errors=True)
        os.makedirs(self.ros_log_dir, exist_ok=True)

