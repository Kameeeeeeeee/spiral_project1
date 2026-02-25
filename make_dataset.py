from __future__ import annotations

from pathlib import Path

import numpy as np

import deb_v2 as deb
from spiral_env import EnvCfg, SpiralEnv
from vision_defaults import CAMERA_DISTANCE_SCALE, CAMERA_LOOKAT_OFFSET, CAMERA_MODE


# Dataset flags
DATASET_ROOT = Path("./assets/dataset")
NUM_FRAMES = 200
IMAGE_SIZE = 2048
MARKER_FAMILY = "apriltag"  # "aruco" | "apriltag" | "both"

# Motion profile
ACTION_FREQ_HZ = 0.55
ACTION_NOISE_STD = 0.12
ACTION_GAIN = 0.9


def _demo_action(step_idx: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    t = float(step_idx) * float(dt)
    w = 2.0 * np.pi * ACTION_FREQ_HZ
    a0 = 0.85 * np.sin(w * t) + 0.35 * np.sin(1.9 * w * t + 0.7)
    a1 = 0.85 * np.cos(1.1 * w * t + 0.3) + 0.30 * np.sin(2.3 * w * t + 1.1)
    a = ACTION_GAIN * np.array([a0, a1], dtype=np.float32)
    a += rng.normal(0.0, ACTION_NOISE_STD, size=(2,)).astype(np.float32)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    arr = frame.astype(np.float32, copy=False)
    vmax = float(np.max(arr)) if arr.size > 0 else 0.0
    if vmax <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _write_rgb_png(path: Path, frame_rgb: np.ndarray) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for writing PNG dataset frames.") from exc
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _build_env_for_family(marker_family: str, image_size: int) -> SpiralEnv:
    original_build = deb.build_mjcf

    def _patched_build_mjcf(*args, **kwargs):
        if "marker_family" in kwargs:
            return original_build(*args, **kwargs)
        return original_build(marker_family=marker_family)

    deb.build_mjcf = _patched_build_mjcf
    try:
        cfg = EnvCfg(
            seed=0,
            obs_mode="vision",
            image_size=image_size,
            image_grayscale=False,
            vision_randomization=False,
            camera_name="top",
            camera_mode=CAMERA_MODE,
            camera_distance_scale=CAMERA_DISTANCE_SCALE,
            camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
            domain_randomization=False,
            mirror_prob=0.0,
        )
        env = SpiralEnv(render_mode=None, cfg=cfg)
        env.reset()
        return env
    finally:
        deb.build_mjcf = original_build


def _clean_previous_frames(out_dir: Path) -> None:
    for p in out_dir.glob("frame_*.png"):
        p.unlink(missing_ok=True)


def _generate_family_dataset(marker_family: str, num_frames: int, image_size: int) -> None:
    out_dir = DATASET_ROOT / marker_family
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_frames(out_dir)

    env = _build_env_for_family(marker_family=marker_family, image_size=image_size)
    rng = np.random.default_rng(123)

    try:
        dt = float(env.cfg.control_dt)
        for step_idx in range(num_frames):
            action = _demo_action(step_idx, dt, rng)
            env.step(action)
            frame = _to_uint8_rgb(env.render_camera())
            _write_rgb_png(out_dir / f"frame_{step_idx:04d}.png", frame)
    finally:
        env.close()


def main() -> None:
    num_frames = int(max(1, NUM_FRAMES))
    image_size = int(max(64, IMAGE_SIZE))
    marker_family = str(MARKER_FAMILY).strip().lower()

    if marker_family not in {"aruco", "apriltag", "both"}:
        raise ValueError("MARKER_FAMILY must be one of: 'aruco', 'apriltag', 'both'")

    families = ["aruco", "apriltag"] if marker_family == "both" else [marker_family]
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    for family in families:
        print(f"Generating dataset: family={family}, frames={num_frames}, size={image_size}")
        _generate_family_dataset(marker_family=family, num_frames=num_frames, image_size=image_size)

    print("Dataset generation complete.")
    print(f"Root: {DATASET_ROOT.resolve()}")


if __name__ == "__main__":
    main()
