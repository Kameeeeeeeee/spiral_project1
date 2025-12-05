# train_spiral_ppo.py

from __future__ import annotations

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from spiral_env import SpiralTentacle2TEnv


def make_env(**env_kwargs):
    def _thunk():
        env = SpiralTentacle2TEnv(**env_kwargs)
        env = Monitor(env)
        return env
    return _thunk


if __name__ == "__main__":
    env_kwargs = dict(
        render_mode=None,
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=400,
    )

    n_envs = 8
    env = make_vec_env(make_env(**env_kwargs), n_envs=n_envs)

    eval_env = SpiralTentacle2TEnv(**env_kwargs)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128],
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.02,
        vf_coef=0.7,
        max_grad_norm=0.7,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_spiral/",
        verbose=1,
    )

    os.makedirs("models", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_spiral_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./models/",
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    total_timesteps = 1_000_000

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    model_path = "ppo_spiral_cable_2t"
    model.save(model_path)
    print(f"Saved final model to {model_path}")

    env.close()
    eval_env.close()
