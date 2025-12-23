# train_spiral_ppo.py
from __future__ import annotations

from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from spiral_env import DomainRandRanges, SpiralPickPlaceEnv


def main():
    dr = DomainRandRanges()
    env_kwargs: Dict[str, Any] = dict(
        render_mode=None,
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
        dr_enable=True,
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
        # FIX (1): ensure directional tip-velocity incentive is off (env keeps API but does not apply it)
        w_v_reach=0.0,
        w_prog_reach=0.05,
        prog_clip=0.01,
        w_cb=1.2,
        w_c=0.10,
        w_vc=0.45,
        w_T=0.25,
        w_L=0.12,
    )

    num_envs = 6
    env = make_vec_env(SpiralPickPlaceEnv, n_envs=num_envs, env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.8,
        tensorboard_log="./tensorboard_spiral_pull/",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    total_timesteps = 240_000
    model.learn(total_timesteps=total_timesteps)

    model_path = "ppo_spiral_pull"
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
