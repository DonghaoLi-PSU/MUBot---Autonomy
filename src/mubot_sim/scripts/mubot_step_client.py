# -*- coding: utf-8 -*-
"""
Client for the MUBot lock-step service  /model/<robot_name>/step  (gz.msgs.Double_V -> Double_V).

    client = StepClient("mubot_1", n_joints=7, partition="mubot_1")
    obs = client.step([5] * 7, n_steps=40)                # hold one voltage vector for 40 steps
    obs = client.step(volts_2d, n_steps=40)               # volts_2d shape (40, 7): per-step sequence
    obs = client.observe()                                # n_steps = 0, no advance
    client.release()                                      # end of rollout, before pause / reset

The sim only advances while a request is being served, so the caller decides exactly
how many physics steps happen. `obs.iteration` advances by exactly n_steps per call,
which _call() verifies.

Reply layout (length 3 + 2N + 13):
    status, sim_time, iteration,
    joint_pos[N], joint_vel[N],
    head x, y, yaw, vx, vy, wz,
    cog x, cog y, cogvel x, cogvel y, cog yaw , tail x, tail y

Needs the gz python bindings (python3-gz-transport13, python3-gz-msgs10 on Harmonic).
GZ_PARTITION must match the simulator's and must be set before the Node is created.
"""

import os
from dataclasses import dataclass

import numpy as np

N_EXTRA = 13        # 6 head values + 7 CoG/tail values after the joint states


@dataclass
class Obs:
    sim_time: float
    iteration: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    head_pose: np.ndarray      # x, y, yaw
    head_vel: np.ndarray       # vx, vy, wz
    cog_xy: np.ndarray         # mass-weighted centre of the link origins (x, y)
    cog_vel: np.ndarray        # mass-weighted link-origin velocity (vx, vy)
    cog_yaw: float             # mass-weighted mean link yaw
    tail_xy: np.ndarray        # last link origin (x, y)


class StepError(RuntimeError):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status        # service status code, None if the call itself failed


class StepClient:
    def __init__(self, robot_name, n_joints, partition=None, timeout_ms=10000):
        if partition is not None:
            os.environ['GZ_PARTITION'] = str(partition)
        from gz.transport13 import Node
        from gz.msgs10.double_v_pb2 import Double_V
        self._Double_V = Double_V
        self._node = Node()
        self.service = f"/model/{robot_name}/step"
        self.n = int(n_joints)
        self.timeout_ms = int(timeout_ms)
        self._last_iter = None

    # ------------------------------------------------------------------
    def observe(self):
        return self._call([0.0])

    def release(self):
        """End of a rollout: stop the plugin waiting for the next request."""
        req = self._Double_V()
        req.data.extend([-1.0])
        ok, rep = self._node.request(self.service, req, self._Double_V, self._Double_V, self.timeout_ms)
        if not ok or not list(rep.data) or list(rep.data)[0] != 0.0:
            raise StepError(f"{self.service}: release failed")
        self._last_iter = None

    def step(self, volts, n_steps):
        volts = np.asarray(volts, dtype=float)
        n_steps = int(n_steps)
        if volts.ndim == 1:
            if volts.size != self.n:
                raise ValueError(f"expected {self.n} voltages, got {volts.size}")
        elif volts.shape != (n_steps, self.n):
            raise ValueError(f"expected shape ({n_steps}, {self.n}), got {volts.shape}")
        return self._call([float(n_steps), *volts.ravel().tolist()])

    def forget_iteration(self):
        """Call after a world reset ."""
        self._last_iter = None

    # ------------------------------------------------------------------
    def _call(self, data):
        req = self._Double_V()
        req.data.extend(data)
        n_steps = int(data[0])
        ok, rep = self._node.request(self.service, req, self._Double_V, self._Double_V, self.timeout_ms)
        if not ok:
            raise StepError(f"{self.service}: no reply in {self.timeout_ms} ms "
                            f"(sim paused, dead, or gave up waiting) -- treat the sim as lost")
        d = list(rep.data)
        if not d or d[0] != 0.0:
            raise StepError(f"{self.service}: status {d[0] if d else 'empty'} "
                            f"(1 invalid request, 2 busy, 3 aborted by reset, "
                            f"4 sim is paused -- unpause it first)",
                            status=int(d[0]) if d else None)
        n = self.n
        if len(d) != 3 + 2 * n + N_EXTRA:
            raise StepError(f"unexpected reply length {len(d)}, expected {3 + 2 * n + N_EXTRA} "
                            f"(rebuild the plugin / check n_joints)")
        b = 3 + 2 * n
        obs = Obs(sim_time=d[1], iteration=int(round(d[2])),
                  joint_pos=np.array(d[3:3 + n]), joint_vel=np.array(d[3 + n:b]),
                  head_pose=np.array(d[b:b + 3]),
                  head_vel=np.array(d[b + 3:b + 6]),
                  cog_xy=np.array(d[b + 6:b + 8]),
                  cog_vel=np.array(d[b + 8:b + 10]),
                  cog_yaw=float(d[b + 10]),
                  tail_xy=np.array(d[b + 11:b + 13]))
        # lock-step check: the sim must have advanced exactly n_steps since the last reply
        if self._last_iter is not None and obs.iteration - self._last_iter != n_steps:
            raise StepError(f"lock-step violated: iteration advanced "
                            f"{obs.iteration - self._last_iter}, expected {n_steps}")
        self._last_iter = obs.iteration
        return obs