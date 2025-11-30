# debug_viewer.py

import time
import numpy as np
import mujoco
import mujoco.viewer

from spiral_env import SpiralTentacle2TEnv


def main():
    # создаем среду
    env = SpiralTentacle2TEnv(
        render_mode=None,
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=10_000,
    )

    model = env.model
    data = env.data

    # начальное состояние
    obs, _ = env.reset()

    print("Открываю MuJoCo viewer для дебага щупальца.")
    print("Щупальце будет автоматически шевелиться по синусам.")
    print("Закрыть окно - просто закрой окно viewer.")

    t0 = time.time()

    # пассивный viewer - мы сами двигаем симуляцию
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0

            # простое тестовое действие - синусные колебания тросов
            # левый и правый трос двигаются с небольшой амплитудой
            left = 0.5 * np.sin(0.5 * t)
            right = 0.5 * np.sin(0.5 * t + np.pi / 2)

            action = np.array([left, right], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
                t0 = time.time()

            # обновляем картинку
            viewer.sync()

            # маленькая пауза чтобы не грузить CPU
            time.sleep(0.01)


if __name__ == "__main__":
    main()
