from __future__ import annotations

import time

from spiral_env import SpiralEnv, EnvCfg
from vision_defaults import IMAGE_SIZE, CAMERA_MODE, CAMERA_DISTANCE_SCALE, CAMERA_LOOKAT_OFFSET

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# Edit these constants directly, no CLI. 
SEED = 0
EPISODES = 50

MODEL_PATH = "runs_spiral/ppo_spiral_vision_final.zip" 
MODEL_PATH = "runs_spiral/checkpoints/ppo_spiral_vision_5857344_steps.zip"
MODEL_PATH = "runs_spiral/best/best_model.zip"


if __name__ == "__main__":
    env = DummyVecEnv(
        [
            lambda: SpiralEnv(
                render_mode="human",
                cfg=EnvCfg(
                    seed=SEED,
                    domain_randomization=False,
                    obs_mode="vision",
                    state_include_ball=False,
                    image_size=IMAGE_SIZE,
                    image_grayscale=True,
                    vision_randomization=False,
                    camera_mode=CAMERA_MODE,
                    camera_distance_scale=CAMERA_DISTANCE_SCALE,
                    camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
                ),
            )
        ]
    )

    model = PPO.load(MODEL_PATH, env=env)

    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        ep_r = 0.0
        t0 = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_r += float(rewards[0])
            done = bool(dones[0])

        dt = time.time() - t0
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        print(f"Episode {ep+1}: return={ep_r:.3f}, walltime={dt:.2f}s, info={info0}")

    env.close()
