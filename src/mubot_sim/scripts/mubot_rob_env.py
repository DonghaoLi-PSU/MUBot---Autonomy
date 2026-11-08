# -*- coding: utf-8 -*-
"""
MubotEnv for ROS 2 (Jazzy) + Gazebo Harmonic.

One rollout = one fresh simulator process:
  1. ros2 launch mubot_sim mubot_arg.launch.py   (started paused, headless)
  2. wait until both plugins report "CSV and URDF matching" in sim.log
  3. unpause through /world/<name>/control (ros_gz_bridge)
  4. wait for terminate.txt written by the Hydro plugin
  5. pause, kill the whole process group, clean temp files

Requires the launch file to accept `run:=false` and to start the control bridge
(see the notes that came with this file).

Original (ROS 1) by donghao Li.
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


def _default_domain(wk_id):
    s = str(wk_id)
    return int(s) % 100 if s.isdigit() else 0


class MubotEnv:

    def __init__(self, worker_folder, max_time, wk_id=1, headless=True,
                 world_name='default', domain_id=None,
                 startup_timeout=60.0, rollout_timeout=10.0):
        print("\33[92mInitializing MubotEnv\033[0m")
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

        self._proc = None
        self._log = None
        atexit.register(self.shutdown)
        print("\33[92mMubotEnv initialized\033[0m")

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
            try:                              # sweep stragglers (gz server can outlive launch)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._log is not None:
            self._log.close()
            self._log = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def step(self) -> bool:
        """Run one rollout. Returns True if drift_metrics.csv is ready."""
        self._remove_files('terminate.txt', '/record/drift_metrics.csv')   # never trust stale files
        conn = None
        ok = False
        try:
            self._start_sim()
            self._wait_ready()
            conn = GazeboConnection(self.node, self.world_name,
                                    timeout=5.0, connect_timeout=30.0)
            if not conn.unpause_sim():
                raise RuntimeError("could not unpause the simulation")
            self._wait_terminate()
            conn.pause_sim()
            ok = True
        except (RuntimeError, TimeoutError) as e:
            print(f"\33[91m[MubotEnv {self.wk_id}] rollout failed: {e}\33[0m")
        finally:
            if conn is not None:
                conn.close()
            self._stop_sim()
            self._cleanup_ros_logs()
        time.sleep(0.2)
        return ok

    def read_metrics(self):
        """[cogvel_x, abs_lateral, rms_lat, heading_drift]"""
        df = pd.read_csv(self.worker_folder + "/record/drift_metrics.csv")
        return df.to_numpy(dtype=float)[0]

    def reward(self, metrics=None):
        m = self.read_metrics() if metrics is None else metrics
        return float(-m[0] - np.clip(m[1] - 0.1, 0.0, None)
                     - np.clip(m[3] - 0.1, 0.0, None) * 0.2)

    def cleanup_outputs(self):
        self._remove_files('terminate.txt', '/record/drift_metrics.csv', 'control_signal.csv')

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