# enjoy_spiral_ppo.py

from stable_baselines3 import PPO

from spiral_env import SpiralTentacle2TEnv


def main():
    env = SpiralTentacle2TEnv(
        render_mode="human",
        num_links=10,
        link_length=0.05,
        link_radius=0.01,
        max_episode_steps=1300,
    )

    model = PPO.load("ppo_spiral_2tendons")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
