from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from spiral_env import SpiralTentacle2TEnv


if __name__ == "__main__":
    # среда: та же геометрия, что и в статье
    env_kwargs = dict(
        render_mode=None,
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=400,
    )

    num_envs = 4

    env = make_vec_env(
        SpiralTentacle2TEnv,
        n_envs=num_envs,
        env_kwargs=env_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=2e-4,
        ent_coef=0.02,   # достаточно высокая энтропия - больше исследования
        clip_range=0.2,
        tensorboard_log="./tensorboard_spiral_grasp/",
    )

    # стейдж 1: учим только захват
    total_timesteps = 350_000
    model.learn(total_timesteps=total_timesteps)

    model_path = "ppo_spiral_grasp_stage1"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    env.close()
