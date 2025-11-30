# enjoy_spiral_ppo.py

import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from spiral_env import SpiralTentacle2TEnv


def main():
    # те же параметры, что и при обучении
    env = SpiralTentacle2TEnv(
        render_mode=None,
        num_links=10,
        link_length=0.05,
        link_radius=0.01,
        max_episode_steps=300,
    )

    # загружаем обученную модель
    model = PPO.load("ppo_spiral_2tendons")

    # создаем viewer на том же model/data, что использует env
    model_mj = env.model
    data_mj = env.data

    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        obs, _ = env.reset()
        t0 = time.time()

        while viewer.is_running():
            # предсказываем действие по текущему наблюдению
            action, _ = model.predict(obs, deterministic=True)

            # делаем шаг окружения (там внутри уже вызывается mj_step)
            obs, reward, terminated, truncated, info = env.step(action)

            # рисуем текущее состояние
            viewer.sync()

            # если эпизод закончился - начинаем новый
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
