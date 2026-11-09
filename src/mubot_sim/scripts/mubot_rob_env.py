# -*- coding: utf-8 -*-
"""
MubotEnv for ROS 2 (Jazzy) + Gazebo Harmonic.

One simulator for each thread only.
Simulators get reset for each rollout.

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
        self._conn = None
        self._n_since_start = 0
        self.restart_every = 300      
        self.reset_timeout = 10.0
        self.worker_folder = os.path.abspath(worker_folder) + "/"   
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
        self._sweep_stale()


    def _sweep_stale(self):
        p = self.worker_folder + 'sim.pgid'
        try:
            pgid = int(open(p).read())
            if b'mubot.launch.py' in open(f'/proc/{pgid}/cmdline', 'rb').read():
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
            'PYTHONUNBUFFERED': '1',
        })
        return env

    def _start_sim(self):
        self._sweep_stale()
        cmd = ['ros2', 'launch', 'mubot_sim', 'mubot.launch.py',
               f'robot_name:={self.robot_name}',
               f'urdf_path:={self.worker_folder}mubot.urdf',
               f'database_dir:={self.database_dir}',
               f'headless:={"true" if self.headless else "false"}',
               'run:=false']                  # start paused; we unpause explicitly
        self._log = open(self.worker_folder + 'sim.log', 'w')
        self._proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=self._child_env(),
            start_new_session=True)           
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

    def step(self) -> bool:
        """Run one rollout on the persistent sim. Returns True if drift_metrics.csv is ready."""
        self._remove_files('terminate.txt', 'reset_ack.txt', '/record/drift_metrics.csv','init_state.csv')
        ok = False
        try:
            self._ensure_sim()          
            self._reset_world()        
            if not self._conn.unpause_sim():
                raise RuntimeError("could not unpause the simulation")
            self._wait_terminate()
            self._conn.pause_sim()
            self._n_since_start += 1
            ok = True
        except (RuntimeError, TimeoutError) as e:
            print(f"\33[91m[MubotEnv {self.wk_id}] rollout failed: {e}\33[0m")
            self._stop_sim()
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