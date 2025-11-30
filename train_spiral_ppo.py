# train_spiral_ppo.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from spiral_env import SpiralTentacle2TEnv


if __name__ == "__main__":
    env_kwargs = dict(
        render_mode=None,
        num_links=12,        # можно чуть больше звеньев для гибкости
        link_length=0.04,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=300,
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
        learning_rate=3e-4,
        ent_coef=0.01,             # поощрение исследования
        tensorboard_log="./tensorboard_spiral/",
    )

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    model_path = "ppo_spiral_2tendons_tapered"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    env.close()
