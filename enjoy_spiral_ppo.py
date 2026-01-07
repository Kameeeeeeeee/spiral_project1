from __future__ import annotations

import time

from spiral_env import SpiralEnv, EnvCfg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Edit these constants directly, no CLI.
SEED = 0
EPISODES = 10

MODEL_PATH = "runs_spiral/ppo_spiral_final.zip"
VECNORM_PATH = "runs_spiral/vecnormalize.pkl"


if __name__ == "__main__":
    env = DummyVecEnv([lambda: SpiralEnv(render_mode="human", cfg=EnvCfg(seed=SEED, domain_randomization=False))])

    try:
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False
    except Exception:
        print("VecNormalize stats not found, running without normalization.")

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
