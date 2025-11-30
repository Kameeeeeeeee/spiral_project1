# enjoy_spiral_ppo.py

from stable_baselines3 import PPO

from spiral_env import SpiralTentacle2TEnv


def main():
    env = SpiralTentacle2TEnv(
        render_mode="human",
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=1300,
    )

    model = PPO.load("ppo_spiral_flat_tapered")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
