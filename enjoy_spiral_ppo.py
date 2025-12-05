# enjoy_spiral_ppo.py

from __future__ import annotations

import time

from stable_baselines3 import PPO

from spiral_env import SpiralTentacle2TEnv


def main():
    env = SpiralTentacle2TEnv(
        render_mode="human",
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=400,
    )

    model_path = "ppo_spiral_cable_2t"
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    truncated = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            print("episode done, info:", info)
            obs, _ = env.reset()
            done = False
            truncated = False

        time.sleep(0.01)


if __name__ == "__main__":
    main()
