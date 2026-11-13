#!/usr/bin/env python3
"""
RL training for MUBot (ROS 2 Jazzy + Gazebo Harmonic).
Uncomplete. Lock-step control inplemented. Not connect to PPO yet.

Run from a shell where ROS and the workspace are sourced:
    python start_empgpe.py --wk-id default
    python start_empgpe.py --wk-id 3 --mode continue
    python start_empgpe.py --wk-id 3 --mode rerun --morph-id 17
"""

import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from EM_pgpe import EMPGPE
from mubot_rob_env import (MubotEnv, sinusoid_voltages, DT, CTRL_STEPS, drift_metrics_from_obs, eval_window)

FAIL_REWARD = -10.0     # reward given to a rollout that still fails after retries
MAX_RETRY = 2           # extra attempts per rollout
MAX_SIM_TIME = 6.5    # give up if terminate.txt never appears


class MubotPGPE:
    def __init__(self, file_path, max_sim_time, wk_id, morph_ID=None, headless=True):
        self.file_path = os.path.abspath(file_path) + "/"
        self.wk_id = wk_id
        self.morph_ID = morph_ID
        self.morph_path = None
        if morph_ID is not None:
            self.morph_path = Path(self.file_path).parent.parent / 'Population_data' / str(morph_ID)
        self.env = MubotEnv(file_path, max_sim_time, wk_id, headless=headless)

    # ------------------------------------------------------------------
    @staticmethod
    def _schedule(joint_num, continuing=False):
        """return (n_rollout, n_episodes) depends on the size of mubot."""
        if joint_num <= 7:
            return 40, 10 * 2
        if joint_num <= 9:
            return (60, 20 * 2) if continuing else (6, 17 * 2)
        return (80, 20 * 2) if continuing else (80, 20 * 2)

    @staticmethod
    def _announce(x, y, n_episodes, n_rollout):
        t = datetime.now().strftime('%m-%d %H:%M:%S')
        print(f"\033[32m{t}  ######## \033[31m  E :{x+1}/{n_episodes} "
              f"\033[93m  R :{y+1}/{n_rollout}\033[0m")

    def _rollout(self, control):
        """Write control_signal.csv, run one simulation, return (metrics[4], reward)."""
        self._csv_write('control_signal.csv', control, mode='w')
        for attempt in range(MAX_RETRY + 1):
            if self.env.step(control):
                stats = self.env.read_metrics()
                reward = self.env.reward(stats)
                self.env.cleanup_outputs()
                return stats, reward
            print(f"\033[91m rollout failed (attempt {attempt+1}/{MAX_RETRY+1})\033[0m")
        self.env.cleanup_outputs()
        return np.full(4, np.nan), FAIL_REWARD
    
    def _step_rollout(self, control, policy=None):
        """policy(obs) -> volts (n_joints,), called once per control step.
        policy=None: open-loop sinusoid from `control` (EM-PGPE parity)."""
        self._csv_write('control_signal.csv', control, mode='w')
        control = np.array([float('%.8e' % x) for x in np.asarray(control, dtype=float)])
        n_joints = control.size // 2
        max_steps = int(MAX_SIM_TIME / DT)
        for attempt in range(MAX_RETRY + 1):
            try:
                obs = self.env.reset(n_joints)
                it0, steps, traj = obs.iteration, 0, [obs]
                while steps < max_steps:
                    n = min(CTRL_STEPS, max_steps - steps)
                    volts = (sinusoid_voltages(control, it0 + steps, n)
                            if policy is None else policy(obs))
                    obs = self.env.advance(volts, n)      # obs after this control step
                    traj.append(obs)
                    steps += n
                self.env.finish()
                self.last_traj = traj
                t0, t1 = eval_window(abs(control[-1]))          # control = the rounded array in this function
                stats = drift_metrics_from_obs(traj, t0, t1)
                if not np.all(np.isfinite(stats)):
                    raise RuntimeError("no observations inside the evaluation window")
                reward = self.env.reward(stats)                 
                self.env.cleanup_outputs()
                return stats, reward
            except (RuntimeError, TimeoutError) as e:     # StepError is a RuntimeError
                print(f"\033[91m rollout failed (attempt {attempt+1}/{MAX_RETRY+1}): {e}\033[0m")
                self.env.abort()
        self.env.cleanup_outputs()
        return np.full(4, np.nan), FAIL_REWARD

    def _move_gait(self):
        if self.morph_ID is not None and os.path.exists(self.file_path + 'gait_record.csv'):
            os.makedirs(self.morph_path, exist_ok=True)
            shutil.move(self.file_path + 'gait_record.csv', self.morph_path / 'gait_.csv')

    # ------------------------------------------------------------------
    def run_pgpe(self, wk_id, joint_num):
        # clean stale files
        for f in ('history_cpg.csv', 'history_rollout.csv', 'history_reward.csv'): 
            if os.path.exists(self.file_path + f):
                os.remove(self.file_path + f)
        self.wk_id = wk_id
        print(f"wk_id {wk_id} and joint number {joint_num}")
        n_rollout, n_episodes = self._schedule(joint_num)
        n_select = int(n_rollout / 2)+1
        n_param = joint_num * 2           # amp, (amp+phase)*(n-1), freq
        record_mu = np.zeros((n_episodes + 1, n_param))
        record_sigma = np.zeros((n_episodes + 1, n_param))
        record_reward = np.zeros((n_episodes + 1, n_rollout))
        record_cpg = np.zeros(((n_episodes + 1) * n_rollout, n_param))
        record_sta = np.zeros(((n_episodes + 1) * n_rollout, 4))

        # initial mu / sigma
        record_mu[0, 0] = random.uniform(10, 15);   record_sigma[0, 0] = 5
        for i in range(joint_num - 1):
            record_mu[0, 2*i+1] = random.uniform(10, 15);  record_sigma[0, 2*i+1] = 5     # amplitude
            record_mu[0, 2*i+2] = random.uniform(0, 1);    record_sigma[0, 2*i+2] = 0.3   # phase
        record_mu[0, -1] = random.uniform(0, 10);   record_sigma[0, -1] = 2.5             # frequency

        empgpe = EMPGPE(n_param, n_rollout, joint_num, n_select)
        self._csv_write('history_cpg.csv', np.hstack((record_mu[0, :], record_sigma[0, :])), mode='w')
        for x in range(n_episodes):
            sl = slice(x * n_rollout, (x + 1) * n_rollout)
            record_cpg[sl, :] = empgpe.sample_params(record_mu[x, :], record_sigma[x, :])
            record_cpg[sl, -1] = np.clip(record_cpg[sl, -1], 1.0, 15)
            for y in range(n_rollout):
                self._announce(x, y, n_episodes, n_rollout)
                idx = x * n_rollout + y
                temp_input = record_cpg[idx, :]
                temp_input = np.array([10,10,-0.1,10,-.2,10,-.3,10,-.4,10,-.5,10,-.6,2])
                # record_sta[idx, :], record_reward[x, y] = self._rollout(temp_input)
                record_sta[idx, :], record_reward[x, y] = self._step_rollout(temp_input)
                mo_y = 'w' if (x == 0 and y == 0) else 'a'
                self._csv_write('history_rollout.csv', np.hstack((temp_input, record_reward[x, y])), mode=mo_y)
                self._csv_write('/record/drift_history.csv', record_sta[idx, :], mode=mo_y)

            record_mu[x+1, :], record_sigma[x+1, :], _ = empgpe.learn(
                record_cpg[sl, :], record_reward[x, :], record_mu[x, :], record_sigma[x, :])
            self._csv_write('history_cpg.csv', np.hstack((record_mu[x+1, :], record_sigma[x+1, :])))
            self._csv_write('history_reward.csv', record_reward[x, :], mode='w' if x == 0 else 'a')
            mean_reward = np.mean(record_reward[x, :]); std_reward = np.std(record_reward[x, :])
            if (std_reward <= mean_reward * 0.1) and (x >= n_episodes * 0.5):
                print("Early converged at episode: " + str(x + 1))
                break
        final_max = np.argmax(record_reward[x, :])
        final_reward = record_reward[x, final_max]
        final_cpg = record_cpg[x * n_rollout + final_max, :]
        self._move_gait()
        return final_reward, final_cpg

    def one_pepg(self, wk_id, morph_ID, joint_num):
        """Re-run one stored controller (Database/<id>/controller.csv)."""
        self.wk_id = wk_id
        print("##### Rerun for morph ID: " + str(morph_ID) + " #####")
        control = pd.read_csv(self.file_path + 'controller.csv', header=None, index_col=None).to_numpy().flatten()
        _, reward = self._rollout(control)
        print(f" Reward:  {reward:.6e}")
        return reward

    def _csv_write(self, filename, data, mode='a'):
        """Write CSV and guarantee the data is on disk before returning,
        so the Gazebo plugin never reads a partially written file."""
        df_data = pd.DataFrame(np.asarray(data).reshape(1, -1))
        with open(self.file_path + filename, mode, newline='') as f:
            df_data.to_csv(f, float_format='%.8e', header=None, index=None)
            f.flush()
            os.fsync(f.fileno())

            


def _count_joints(folder: Path):
    with open(folder / 'joint_param.csv') as f:
        return sum(1 for line in f if line.strip())


def main():
    here = Path(__file__).resolve().parent
    default_db = os.environ.get('MUBOT_DATABASE_DIR') or str(here.parent / 'Database')

    ap = argparse.ArgumentParser(description="MUBot EM-PGPE training")
    ap.add_argument('--wk-id', default='1', help="worker / model id = Database sub-folder name")
    ap.add_argument('--database-dir', default=default_db)
    ap.add_argument('--act-num', type=int, default=None, help="number of joints (default: rows of joint_param.csv)")
    ap.add_argument('--max-sim-time', type=float, default=6.1)
    ap.add_argument('--mode', choices=['run', 'continue', 'rerun'], default='run')
    ap.add_argument('--morph-id', default=None)
    ap.add_argument('--gui', action='store_true', help="show the Gazebo GUI (default: headless)")
    args = ap.parse_args()

    folder = Path(args.database_dir) / str(args.wk_id)
    if not folder.is_dir():
        raise SystemExit(f"worker folder not found: {folder}")
    act_num = args.act_num or _count_joints(folder)
    print(f"worker folder: {folder}   actuators: {act_num}   mode: {args.mode}")

    pgpe = MubotPGPE(folder, args.max_sim_time, args.wk_id, args.morph_id, headless=not args.gui)
    try:
        if args.mode == 'run':
            reward, cpg = pgpe.run_pgpe(args.wk_id, act_num)
        # working on continue and one step in future
        # elif args.mode == 'continue':
        #     reward, cpg = pgpe.continue_pgpe(args.wk_id, act_num)
        # else:
        #     reward, cpg = pgpe.one_pepg(args.wk_id, args.morph_id, act_num), None
        print(f"\nfinal reward: {reward:.6e}")
        if cpg is not None:
            print("best control:", np.array2string(cpg, precision=4))
    finally:
        pgpe.env.shutdown()     # kills any sim still running, also on Ctrl-C


if __name__ == '__main__':
    main()