# debug_viewer.py

import time
import numpy as np

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

    obs, _ = env.reset()

    while True:
        # можно поставить нули, чтобы щупальца просто лежала
        # action = np.zeros(env.action_space.shape, dtype=np.float32)
        action = env.action_space.sample().astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("episode ended, info:", info)
            obs, _ = env.reset()

        time.sleep(0.01)


if __name__ == "__main__":
    main()
