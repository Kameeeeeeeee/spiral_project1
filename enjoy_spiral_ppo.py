# enjoy_spiral_ppo.py
from __future__ import annotations

import time
from typing import Any, Dict

from stable_baselines3 import PPO

from spiral_env import DomainRandRanges, SpiralPickPlaceEnv


def main():
    dr = DomainRandRanges()
    env_kwargs: Dict[str, Any] = dict(
        render_mode="human",
        n_segments=24,
        total_length=0.45,
        tip_width=0.0075,
        tip_thickness=0.0024,
        lift_z=0.010,
        tendon_offset_frac=0.55,
        k_tip=3.0,
        damping_mul=1.2,
        frictionloss_tip=0.012,
        armature_mul=1.2,
        m_tip=2e-6,
        motor_gear=2600.0,
        timestep=0.001,
        cube_half=0.02,
        cube_density=800.0,
        cube_friction="1.1 0.02 0.0001",
        max_episode_steps=900,
        substeps=10,
        arena_half=0.45,
        cube_spawn_xy_jitter=0.06,
        goal_a_xy=(0.18, 0.12),
        goal_b_xy=(0.18, -0.12),
        goal_z=0.02,
        goal_radius=0.05,
        success_hold_steps=30,
        grasp_min_contacts=2,
        grasp_v_cube_max=0.25,
        grasp_rel_v_tip_max=0.25,
        grasp_max_z=0.10,
        l_max=0.18,
        ell_rest_bias=0.0,
        kp=200000.0,
        kd=6000.0,
        dr_ranges=dr,
        dr_enable=False,
        w_reach=1.6,
        w_grasp=1.4,
        w_transport=0.0,
        w_place=0.0,
        ctrl_energy_cost=0.0015,
        action_smooth_cost=0.01,
        grasp_v_scale=0.40,
        grasp_t_scale=0.55,
        grasp_filter_tau=0.12,
        vspool_penalty_coef=0.05,
        tension_penalty_coef=0.10,
        v_fly_threshold=1.0,
        z_fly_threshold=0.18,
        fly_penalty=8.0,
        r_contact=0.09,
        alpha_success=0.50,
        v_soft=0.22,
        w_d_reach=1.0,
        w_v_reach=0.0,
        w_prog_reach=0.05,
        prog_clip=0.01,
        w_cb=1.2,
        w_c=0.10,
        w_vc=0.45,
        w_T=0.25,
        w_L=0.12,
    )

    env = SpiralPickPlaceEnv(**env_kwargs)
    model = PPO.load("ppo_spiral_pull", device="cpu")

    obs, info = env.reset()
    ep_r = 0.0
    ep = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        ep_r += float(r)

        if env.render_mode == "human":
            time.sleep(0.001)

        if (env._step % 30) == 0:
            t_sum = float(info.get("t_left", 0.0) + info.get("t_right", 0.0))
            print(
                "step",
                env._step,
                "phase",
                info.get("phase"),
                "tip->cube",
                f'{info.get("dist_tip_to_cube", 0.0):.3f}',
                "cube->base",
                f'{info.get("dist_cube_to_base", 0.0):.3f}',
                "contacts",
                info.get("num_contacts"),
                "v_cube",
                f'{info.get("cube_speed", 0.0):.3f}',
                "v_spool",
                f'{info.get("v_spool", 0.0):.3f}',
                "T_sum",
                f"{t_sum:.1f}",
                "dL",
                f'{info.get("dL", 0.0):.4f}',
                "success",
                info.get("success"),
            )

        if terminated or truncated:
            print(
                "episode",
                ep,
                "return",
                f"{ep_r:.2f}",
                "terminated",
                terminated,
                "truncated",
                truncated,
                "success",
                info.get("success"),
            )
            ep += 1
            ep_r = 0.0
            obs, info = env.reset()


if __name__ == "__main__":
    main()
