from __future__ import annotations

# README
# 1) Run: python aruco_demo_check.py
# 2) Script simulates a few MuJoCo steps, renders from camera, runs ArUco pipeline.
# 3) It prints detected/missing IDs, state vector shape, and sample segment state.
# 4) If SAVE_DEBUG_IMAGES is True, debug overlays are saved to ./debug/.

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import deb_v2 as deb
from aruco_pipeline import ArucoConfig, ArucoPipeline, make_camera_matrix_from_fovy
from spiral_env import EnvCfg, SpiralEnv
from vision_defaults import CAMERA_DISTANCE_SCALE, CAMERA_LOOKAT_OFFSET, CAMERA_MODE


# Keeping variable for compatibility. Demo now uses SpiralEnv camera path.
PATH_TO_XML = Path("./assets/spiral_scene.xml")
CAMERA_NAME = "top"
RENDER_SIZE = 1024
N_STEPS = 64
SAVE_DEBUG_IMAGES = True
ACTION_FREQ_HZ = 0.55
ACTION_NOISE_STD = 0.12
ACTION_GAIN = 0.9
USE_POSE3D = False


def _draw_debug(frame: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception:
        return frame
    img = frame.copy()
    tracks = info.get("tracks", {})
    for marker_id, tr in tracks.items():
        valid = bool(tr.get("valid", False))
        color = (40, 220, 40) if valid else (50, 50, 220)
        corners = np.asarray(tr.get("corners_px", np.zeros((4, 2))), dtype=np.float32).reshape(4, 2)
        if np.all(np.isfinite(corners)) and np.max(np.abs(corners)) > 1e-6:
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        c = np.asarray(tr.get("center_px", np.zeros(2)), dtype=np.float32).reshape(2)
        cv2.circle(img, (int(c[0]), int(c[1])), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"id={marker_id} v={int(valid)} l={int(tr.get('lost_frames', 0))}",
            (int(c[0]) + 4, int(c[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    arr = frame.astype(np.float32, copy=False)
    vmax = float(np.max(arr)) if arr.size > 0 else 0.0
    if vmax <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _demo_action(step_idx: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate smooth but diverse 2D actions so tentacle visits many poses.
    """
    t = float(step_idx) * float(dt)
    w = 2.0 * np.pi * ACTION_FREQ_HZ
    a0 = 0.85 * np.sin(w * t) + 0.35 * np.sin(1.9 * w * t + 0.7)
    a1 = 0.85 * np.cos(1.1 * w * t + 0.3) + 0.30 * np.sin(2.3 * w * t + 1.1)
    a = ACTION_GAIN * np.array([a0, a1], dtype=np.float32)
    a += rng.normal(0.0, ACTION_NOISE_STD, size=(2,)).astype(np.float32)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def main() -> None:
    xml = deb.build_mjcf()
    model_tmp = mujoco.MjModel.from_xml_string(xml)
    off_h = int(model_tmp.vis.global_.offheight)
    safe_size = int(min(RENDER_SIZE, off_h))

    # Use the same camera setup path as vision pretraining files.
    cfg = EnvCfg(
        seed=0,
        obs_mode="vision",
        image_size=safe_size,
        image_grayscale=False,
        vision_randomization=False,
        camera_name=CAMERA_NAME,
        camera_mode=CAMERA_MODE,
        camera_distance_scale=CAMERA_DISTANCE_SCALE,
        camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
        domain_randomization=False,
        mirror_prob=0.0,
    )
    env = SpiralEnv(render_mode=None, cfg=cfg)
    env.reset()

    ar_cfg = ArucoConfig(
        dict_name="DICT_4X4_50",
        marker_length_m=float(deb.marker_length_m),
        output_mode="pose3d" if USE_POSE3D else "image2d",
        expected_ids=list(range(19)),
        input_color="RGB",
        useRefineDetectedMarkers=False,
        enable_kalman=True,
        camera_matrix=make_camera_matrix_from_fovy(
            width=safe_size,
            height=safe_size,
            fovy_deg=float(deb.CAM_TOP_FOVY),
        )
        if USE_POSE3D
        else None,
        dist_coeffs=np.zeros((5, 1), dtype=np.float32) if USE_POSE3D else None,
    )
    pipeline = ArucoPipeline(ar_cfg)
    rng = np.random.default_rng(123)

    debug_dir = Path("./debug")
    if SAVE_DEBUG_IMAGES:
        debug_dir.mkdir(parents=True, exist_ok=True)

    last_state = None
    last_info = None
    detected_counts: list[int] = []
    for step_idx in range(N_STEPS):
        action = _demo_action(step_idx, cfg.control_dt, rng)
        env.step(action)
        frame = _to_uint8_rgb(env.render_camera())
        state_vec, info = pipeline.step(frame, timestamp=step_idx)
        detected_counts.append(int(info.get("num_detected", 0)))
        last_state = state_vec
        last_info = info
        if SAVE_DEBUG_IMAGES and step_idx % 4 == 0:
            dbg = _draw_debug(frame, info)
            try:
                import cv2  # type: ignore

                bgr = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"aruco_debug_{step_idx:03d}.png"), bgr)
            except Exception:
                pass

    env.close()

    if last_state is None or last_info is None:
        print("No frames processed.")
        return

    print("=== ArUco Demo Summary ===")
    print(f"detected_ids: {last_info['detected_ids']}")
    print(f"missing_ids:  {last_info['missing_ids']}")
    print(f"state_vec shape: {tuple(last_state.shape)}")
    if last_state.size >= 3:
        print(f"segment_01 state head: {last_state[:8].tolist()}")
    print(f"world_missing: {last_info.get('world_missing', True)}")
    print(f"mean_reprojection_error: {last_info.get('mean_reprojection_error')}")
    print(f"last_frame_mean_intensity: {float(np.mean(frame)):.3f}")
    if detected_counts:
        print(f"detected_ids_avg: {float(np.mean(detected_counts)):.2f}")
        print(f"detected_ids_max: {int(np.max(detected_counts))}")


if __name__ == "__main__":
    main()
