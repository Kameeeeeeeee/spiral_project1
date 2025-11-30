# train_spiral_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from spiral_env import SpiralTentacle2TEnv


if __name__ == "__main__":
    # параметры щупальца
    env_kwargs = dict(
        render_mode=None,
        num_links=10,
        link_length=0.05,
        link_radius=0.01,
        max_episode_steps=300,
    )

    # количество параллельных окружений
    num_envs = 4

    # создаем векторизованное окружение
    # сюда передаем КЛАСС окружения и env_kwargs
    env = make_vec_env(
        SpiralTentacle2TEnv,
        n_envs=num_envs,
        env_kwargs=env_kwargs,
    )

    # создаем модель PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard_spiral/",
    )

    # обучение
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # сохранение модели
    model_path = "ppo_spiral_2tendons"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    env.close()
