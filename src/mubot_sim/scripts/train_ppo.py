# -*- coding: utf-8 -*-
"""
PPO training for MUBot (replaces start_ppo.py / EM-PGPE).

    pip install gymnasium stable-baselines3 tensorboard
    python3 train_ppo.py --smoke                       # 1 random episode, checks the whole pipeline
    python3 train_ppo.py --timesteps 1000000           # train
    python3 train_ppo.py --resume --timesteps 500000   # continue from Database/<id>/ppo/
    python3 train_ppo.py --eval                        # one deterministic episode with the saved policy

Output in Database/<wk_id>/ppo/: ppo_mubot.zip, vecnormalize.pkl, checkpoints/, tb/
One PPO update uses n_steps control steps (default 1200 = 2 episodes of 600 steps).
"""

import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mubot_gym_env import MubotGymEnv


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wk-id', type=int, default=1)
    ap.add_argument('--database-dir', default='Database')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--timesteps', type=int, default=1_000_000)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--n-steps', type=int, default=1200)
    ap.add_argument('--batch-size', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w-act', type=float, default=0.0, help='penalty on action changes')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()

class LogCallback(BaseCallback):
    """Logs the old-style episode reward (comparable with EM-PGPE) and simulator failures."""
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "orig_reward" in info and np.isfinite(info["orig_reward"]):
                self.logger.record_mean("mubot/orig_reward", info["orig_reward"])
                d = info.get("drift")
                if d is not None:
                    self.logger.record_mean("mubot/cogvel_x", d[0])
                    self.logger.record_mean("mubot/abs_lat", d[1])
                    self.logger.record_mean("mubot/heading_drift", d[3])
            if info.get("sim_failed"):
                self.logger.record_mean("mubot/sim_failed", 1.0)
        return True

def make_env(args, folder):
    from stable_baselines3.common.monitor import Monitor
    env = MubotGymEnv(folder, wk_id=args.wk_id, headless=not args.gui, w_act=args.w_act)
    return Monitor(env, info_keywords=("orig_reward", "sim_failed"))


def smoke(args, folder):
    """Random-policy episode without SB3: checks reset/step/finish and prints timing."""
    import time
    env = MubotGymEnv(folder, wk_id=args.wk_id, headless=not args.gui)
    try:
        obs, _ = env.reset()
        print(f"obs dim {obs.shape[0]}, action dim {env.action_space.shape[0]}")
        t0, ret, k, info = time.time(), 0.0, 0, {}
        while True:
            obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
            ret += r
            k += 1
            if not np.all(np.isfinite(obs)):
                print("NaN/inf in observation at step", k)
                break
            if term or trunc:
                break
        dt = time.time() - t0
        print(f"{k} steps, return {ret:.3f}, {dt:.1f} s ({1e3*dt/max(k,1):.1f} ms/step)")
        print("drift", info.get("drift"), "orig_reward", info.get("orig_reward"),
              "sim_failed", info.get("sim_failed", False))
    finally:
        env.close()


def main():
    args = parse()

    here = Path(__file__).resolve().parent    
    folder = here.parent / args.database_dir / str(args.wk_id)
    out = folder / 'ppo'
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        smoke(args, folder)
        return

    venv = DummyVecEnv([lambda: make_env(args, folder)])
    model_path, norm_path = out / 'ppo_mubot', out / 'vecnormalize.pkl'

    if args.resume or args.eval:
        venv = VecNormalize.load(str(norm_path), venv)
        model = PPO.load(str(model_path), env=venv, device='cpu')
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO('MlpPolicy', venv, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=10, learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, seed=args.seed, device='cpu',
                    policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.Tanh, log_std_init=-1.0),
                    tensorboard_log=str(out / 'tb') if _has_tb() else None, verbose=1)

    def save():
        model.save(str(model_path))
        venv.save(str(norm_path))
        print(f"saved {model_path}.zip and {norm_path}")

    if args.eval:
        venv.training = False
        venv.norm_reward = False
        obs = venv.reset()
        done, ret = False, 0.0
        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, dones, infos = venv.step(action)
                ret += float(r[0])
                done = bool(dones[0])
            print("return", ret, "orig_reward", infos[0].get("orig_reward"), "drift", infos[0].get("drift"))
        finally:
            venv.close()
        return

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    ckpt = CheckpointCallback(save_freq=max(50_000 // 1, 1), save_path=str(out / 'checkpoints'),
                              name_prefix='ppo_mubot', save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=[ckpt, LogCallback()],
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("interrupted, saving")
    finally:
        save()
        venv.close()


def _has_tb():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == '__main__':
    main()
