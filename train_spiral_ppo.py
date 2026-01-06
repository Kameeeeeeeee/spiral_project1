from __future__ import annotations

import torch

from spiral_env import SpiralEnv, EnvCfg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


# Edit these constants directly, no CLI.
SEED = 0
N_ENVS = 4
TOTAL_TIMESTEPS = 500_000

LOGDIR = "runs_spiral"


def _make(rank: int):
    def _thunk():
        cfg = EnvCfg(seed=SEED + 1000 * rank, domain_randomization=True)
        return Monitor(SpiralEnv(render_mode=None, cfg=cfg))
    return _thunk


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if N_ENVS <= 1:
        env = DummyVecEnv([_make(0)])
    else:
        env = SubprocVecEnv([_make(i) for i in range(N_ENVS)], start_method="spawn")

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    eval_env = DummyVecEnv([lambda: Monitor(SpiralEnv(render_mode=None, cfg=EnvCfg(seed=SEED + 9999, domain_randomization=True)))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.2,
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        use_sde=True,
        sde_sample_freq=4,
        verbose=1,
        tensorboard_log=f"{LOGDIR}/tb",
        seed=SEED,
    )
    print("SB3 device:", model.device)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // max(1, N_ENVS)),
        save_path=f"{LOGDIR}/checkpoints",
        name_prefix="ppo_spiral",
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{LOGDIR}/best",
        log_path=f"{LOGDIR}/eval",
        eval_freq=max(1, 50_000 // max(1, N_ENVS)),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[ckpt_cb, eval_cb])

    model.save(f"{LOGDIR}/ppo_spiral_final")
    env.save(f"{LOGDIR}/vecnormalize.pkl")
    print("Saved:", f"{LOGDIR}/ppo_spiral_final.zip", f"{LOGDIR}/vecnormalize.pkl")
