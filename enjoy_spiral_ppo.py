from __future__ import annotations

from stable_baselines3 import PPO

from spiral_env import SpiralTentacle2TEnv


def main():
    env = SpiralTentacle2TEnv(
        render_mode="human",
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=2000,
    )

    model = PPO.load("ppo_spiral_grasp_stage1")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(
                f"episode done | success={info.get('success')} "
                f"dist_tip_ball={info.get('dist_tip_ball'):.3f} "
                f"disp={info.get('ball_disp'):.3f} "
                f"wrap={info.get('wrap_term'):.3f} "
                f"in_contact={info.get('in_contact')} "
            )
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
