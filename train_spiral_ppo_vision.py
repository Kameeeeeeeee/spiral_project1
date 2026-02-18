from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from spiral_env import SpiralEnv, EnvCfg
from vision_defaults import (
    IMAGE_SIZE,
    CAMERA_MODE,
    CAMERA_DISTANCE_SCALE,
    CAMERA_LOOKAT_OFFSET,
    SPAWN_CURRICULUM_EPISODES,
    SPAWN_RADIUS_SCALE_START,
)
from vision_models import VisionModelCfg, SpiralVisionBackbone

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Edit these constants directly, no CLI.
SEED = 0
N_ENVS = 12
TOTAL_TIMESTEPS = 10_000_000

LOGDIR = "runs_spiral"
MODEL_PREFIX = "ppo_spiral_vision"

BACKBONE_PATH = "runs_spiral/vision_backbone.pt"
FREEZE_BACKBONE = False

DR_RAMP_START = 300_000
DR_RAMP_END = 1_500_000
DR_UPDATE_FREQ = 1_000
DR_MIN_STRENGTH = 0.0


def _make(rank: int):
    def _thunk():
        cfg = EnvCfg(
            seed=SEED + 1000 * rank,
            domain_randomization=True,
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


class SyncEvalCallback(EvalCallback):
    def __init__(self, eval_env, train_env, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        return super()._on_step()


class SpiralVisionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        image_features_dim: int = 128,
        state_features_dim: int = 64,
        backbone_path: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        image_space = observation_space["image"]
        state_space = observation_space["state"]
        if len(image_space.shape) != 3:
            raise ValueError(f"Expected image shape (H, W, C), got {image_space.shape}.")
        img_h, img_w, img_channels = image_space.shape
        if img_h != img_w:
            raise ValueError("Expected square images.")

        self._image_cfg = VisionModelCfg(
            image_size=int(img_h),
            in_channels=int(img_channels),
            features_dim=int(image_features_dim),
        )
        super().__init__(observation_space, features_dim=image_features_dim + state_features_dim)

        self.image_net = SpiralVisionBackbone(self._image_cfg)
        self.state_net = nn.Sequential(
            nn.Linear(int(state_space.shape[0]), int(state_features_dim)),
            nn.ReLU(),
        )

        if backbone_path:
            path = Path(backbone_path)
            if path.is_file():
                payload = torch.load(path, map_location="cpu")
                state_dict = payload.get("state_dict", payload)
                self.image_net.load_state_dict(state_dict, strict=False)

        if freeze_backbone:
            for param in self.image_net.parameters():
                param.requires_grad = False

    def forward(self, obs):
        img = obs["image"].float().div(255.0).permute(0, 3, 1, 2).contiguous()
        state = obs["state"].float()
        img_features = self.image_net(img)
        state_features = self.state_net(state)
        return torch.cat([img_features, state_features], dim=1)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    sample_obs_space = env.observation_space
    state_dim = int(sample_obs_space["state"].shape[0])
    action_hist_len = state_dim // 2
    print(f"Obs config: action_history_len(k)={action_hist_len}, state_dim={state_dim}")

    policy_kwargs = dict(
        features_extractor_class=SpiralVisionExtractor,
        features_extractor_kwargs=dict(
            image_features_dim=128,
            state_features_dim=64,
            backbone_path=BACKBONE_PATH,
            freeze_backbone=FREEZE_BACKBONE,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=0.0,
        normalize_images=False,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        device=device,
        n_steps=1024,
        batch_size=3072,
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
        sde_sample_freq=1,
        verbose=1,
        tensorboard_log=f"{LOGDIR}/tb",
        seed=SEED,
    )
    print("SB3 device:", model.device)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // max(1, N_ENVS)),
        save_path=f"{LOGDIR}/checkpoints",
        name_prefix=MODEL_PREFIX,
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

    model.save(f"{LOGDIR}/{MODEL_PREFIX}_final")
    print("Saved:", f"{LOGDIR}/{MODEL_PREFIX}_final.zip")
