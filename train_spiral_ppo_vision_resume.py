from __future__ import annotations

import re
from pathlib import Path

import torch

from spiral_env import SpiralEnv, EnvCfg
from vision_defaults import (
    IMAGE_SIZE,
    CAMERA_MODE,
    CAMERA_DISTANCE_SCALE,
    CAMERA_LOOKAT_OFFSET,
    SPAWN_CURRICULUM_EPISODES,
    SPAWN_RADIUS_SCALE_START,
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


# Edit these constants directly, no CLI.
SEED = 0
N_ENVS = 4
EXTRA_TIMESTEPS = 5_000_000

LOGDIR = Path("runs_spiral")
MODEL_PREFIX = "ppo_spiral_vision"

MODEL_PATH = "runs_spiral\ppo_spiral_vision_continued100k.zip"

MODEL_PATH = "runs_spiral/ppo_spiral_vision_final.zip"

MODEL_PATH = "runs_spiral/checkpoints/ppo_spiral_vision_11600000_steps.zip"
#MODEL_PATH = "runs_spiral/best/best_model.zip"


USE_DR = False
DR_RAMP_START = 300_000
DR_RAMP_END = 1_500_000
DR_UPDATE_FREQ = 1_000
DR_MIN_STRENGTH = 0.0


def _make(rank: int):
    def _thunk():
        cfg = EnvCfg(
            seed=SEED + 1000 * rank,
            domain_randomization=USE_DR,
            obs_mode="vision",
            state_include_ball=False,
            image_size=IMAGE_SIZE,
            image_grayscale=True,
            vision_randomization=True,
            camera_mode=CAMERA_MODE,
            camera_distance_scale=CAMERA_DISTANCE_SCALE,
            camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
            spawn_curriculum_episodes=SPAWN_CURRICULUM_EPISODES,
            spawn_radius_scale_start=SPAWN_RADIUS_SCALE_START,
        )
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


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    best_path = None
    best_steps = -1
    pat = re.compile(rf"{re.escape(MODEL_PREFIX)}_(\d+)_steps\.zip$")
    for p in ckpt_dir.glob(f"{MODEL_PREFIX}_*_steps.zip"):
        m = pat.search(p.name)
        if not m:
            continue
        steps = int(m.group(1))
        if steps > best_steps:
            best_steps = steps
            best_path = p
    return best_path


def _pick_resume_path() -> Path:
    ckpt_dir = LOGDIR / "checkpoints"
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        return latest_ckpt

    best_model = LOGDIR / "best" / "best_model.zip"
    if best_model.exists():
        return best_model

    final_model = LOGDIR / f"{MODEL_PREFIX}_final.zip"
    return final_model


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    resume_model_path = MODEL_PATH
    print("Resuming from:", resume_model_path)
    if N_ENVS <= 1:
        env = DummyVecEnv([_make(0)])
    else:
        env = SubprocVecEnv([_make(i) for i in range(N_ENVS)], start_method="spawn")

    eval_env = DummyVecEnv(
        [
            lambda: Monitor(
                SpiralEnv(
                    render_mode=None,
                    cfg=EnvCfg(
                        seed=SEED + 9999,
                        domain_randomization=False,
                        obs_mode="vision",
                        state_include_ball=False,
                        image_size=IMAGE_SIZE,
                        image_grayscale=True,
                        vision_randomization=False,
                        camera_mode=CAMERA_MODE,
                        camera_distance_scale=CAMERA_DISTANCE_SCALE,
                        camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
                        spawn_curriculum_episodes=SPAWN_CURRICULUM_EPISODES,
                        spawn_radius_scale_start=SPAWN_RADIUS_SCALE_START,
                    ),
                )
            )
        ]
    )

    model = PPO.load(
        str(resume_model_path),
        env=env,
        device="cpu",
        tensorboard_log=str(LOGDIR / "tb"),
        seed=SEED,
    )
    print("SB3 device:", model.device)
    print("Loaded num_timesteps:", model.num_timesteps)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // max(1, N_ENVS)),
        save_path=str(LOGDIR / "checkpoints"),
        name_prefix=MODEL_PREFIX,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(LOGDIR / "best"),
        log_path=str(LOGDIR / "eval"),
        eval_freq=max(1, 50_000 // max(1, N_ENVS)),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    callbacks = [ckpt_cb, eval_cb]
    if USE_DR:
        dr_cb = DRStrengthCallback(
            DR_RAMP_START,
            DR_RAMP_END,
            update_freq=DR_UPDATE_FREQ,
            min_strength=DR_MIN_STRENGTH,
        )
        callbacks.append(dr_cb)

    model.learn(total_timesteps=EXTRA_TIMESTEPS, callback=callbacks, reset_num_timesteps=False)

    out_model = LOGDIR / f"{MODEL_PREFIX}_continued"
    model.save(str(out_model))
    print("Saved:", f"{out_model}.zip")
