from __future__ import annotations

import torch

from spiral_env import SpiralEnv, EnvCfg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


# Edit these constants directly, no CLI.
SEED = 0
N_ENVS = 4
TOTAL_TIMESTEPS = 5_000_000

LOGDIR = "runs_spiral"

DR_RAMP_START = 300_000
DR_RAMP_END = 1_500_000
DR_UPDATE_FREQ = 1_000
DR_MIN_STRENGTH = 0.0


def _make(rank: int):
    def _thunk():
        cfg = EnvCfg(seed=SEED + 1000 * rank, domain_randomization=True)
        return Monitor(SpiralEnv(render_mode=None, cfg=cfg))
    return _thunk


class DRStrengthCallback(BaseCallback):
    def __init__(self, ramp_start: int, ramp_end: int, update_freq: int = 1000, min_strength: float = 0.0) -> None:
        super().__init__(verbose=0)
        self.ramp_start = int(ramp_start)
        self.ramp_end = int(max(ramp_end, ramp_start + 1))
        self.update_freq = int(max(1, update_freq))
        self.min_strength = float(max(0.0, min(1.0, min_strength)))
        self._last_update = -1

    def _progress(self, num_timesteps: int) -> float:
        if num_timesteps <= self.ramp_start:
            progress = 0.0
        elif num_timesteps >= self.ramp_end:
            progress = 1.0
        else:
            progress = (num_timesteps - self.ramp_start) / float(self.ramp_end - self.ramp_start)
        return max(self.min_strength, progress)

    def _update_strength(self) -> None:
        strength = self._progress(self.num_timesteps)
        self.training_env.env_method("set_dr_strength", strength)

    def _on_training_start(self) -> None:
        self._last_update = self.num_timesteps - self.update_freq
        self._update_strength()

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_update) >= self.update_freq:
            self._last_update = self.num_timesteps
            self._update_strength()
        return True


class SyncEvalCallback(EvalCallback):
    def __init__(self, eval_env, train_env, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms
        return super()._on_step()


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

    eval_env = DummyVecEnv([lambda: Monitor(SpiralEnv(render_mode=None, cfg=EnvCfg(seed=SEED + 9999, domain_randomization=False)))])
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
        batch_size=1024,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.03,
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

    eval_cb = SyncEvalCallback(
        eval_env,
        env,
        best_model_save_path=f"{LOGDIR}/best",
        log_path=f"{LOGDIR}/eval",
        eval_freq=max(1, 50_000 // max(1, N_ENVS)),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    dr_cb = DRStrengthCallback(
        DR_RAMP_START,
        DR_RAMP_END,
        update_freq=DR_UPDATE_FREQ,
        min_strength=DR_MIN_STRENGTH,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[ckpt_cb, eval_cb, dr_cb])

    model.save(f"{LOGDIR}/ppo_spiral_final")
    env.save(f"{LOGDIR}/vecnormalize.pkl")
    print("Saved:", f"{LOGDIR}/ppo_spiral_final.zip", f"{LOGDIR}/vecnormalize.pkl")
