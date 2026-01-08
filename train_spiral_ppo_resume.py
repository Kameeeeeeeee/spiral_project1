from __future__ import annotations

import re
from pathlib import Path

import torch

from spiral_env import SpiralEnv, EnvCfg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

SEED = 0
N_ENVS = 4

EXTRA_TIMESTEPS = 5_000_000

LOGDIR = Path("runs_spiral")

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


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    best_path = None
    best_steps = -1
    pat = re.compile(r"ppo_spiral_(\d+)_steps\.zip$")
    for p in ckpt_dir.glob("ppo_spiral_*_steps.zip"):
        m = pat.search(p.name)
        if not m:
            continue
        steps = int(m.group(1))
        if steps > best_steps:
            best_steps = steps
            best_path = p
    return best_path


def _pick_resume_paths() -> tuple[Path, Path]:
    ckpt_dir = LOGDIR / "checkpoints"
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    vn_root = LOGDIR / "vecnormalize.pkl"
    if latest_ckpt is not None:
        vn = ckpt_dir / "vecnormalize.pkl"
        if vn.exists():
            return latest_ckpt, vn
        if vn_root.exists():
            return latest_ckpt, vn_root
        return latest_ckpt, vn

    best_model = LOGDIR / "best" / "best_model.zip"
    if best_model.exists():
        return best_model, vn_root

    final_model = LOGDIR / "ppo_spiral_final.zip"
    return final_model, vn_root


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    resume_model_path, resume_vn_path = _pick_resume_paths()
    if not resume_model_path.exists():
        raise FileNotFoundError(f"Resume model not found: {resume_model_path.resolve()}")
    if not resume_vn_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {resume_vn_path.resolve()}")

    print("Resuming from model:", resume_model_path)
    print("Resuming VecNormalize:", resume_vn_path)

    if N_ENVS <= 1:
        venv = DummyVecEnv([_make(0)])
    else:
        venv = SubprocVecEnv([_make(i) for i in range(N_ENVS)], start_method="spawn")

    env = VecNormalize.load(str(resume_vn_path), venv)
    env.training = True
    env.norm_reward = True

    eval_env = DummyVecEnv([lambda: Monitor(SpiralEnv(render_mode=None, cfg=EnvCfg(seed=SEED + 9999, domain_randomization=False)))])
    eval_env = VecNormalize.load(str(resume_vn_path), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(
        str(resume_model_path),
        env=env,
        device="cpu",
        tensorboard_log=str(LOGDIR / "tb"),
        seed=SEED,
        print_system_info=True,
    )
    print("SB3 device:", model.device)
    print("Loaded num_timesteps:", model.num_timesteps)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // max(1, N_ENVS)),
        save_path=str(LOGDIR / "checkpoints"),
        name_prefix="ppo_spiral",
        save_vecnormalize=True,
    )

    eval_cb = SyncEvalCallback(
        eval_env,
        env,
        best_model_save_path=str(LOGDIR / "best"),
        log_path=str(LOGDIR / "eval"),
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

    model.learn(total_timesteps=EXTRA_TIMESTEPS, callback=[ckpt_cb, eval_cb, dr_cb], reset_num_timesteps=False)

    out_model = LOGDIR / "ppo_spiral_continued"
    model.save(str(out_model))
    env.save(str(LOGDIR / "vecnormalize.pkl"))
    print("Saved:", f"{out_model}.zip", str(LOGDIR / "vecnormalize.pkl"))
