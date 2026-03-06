from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

import deb_v2 as deb
from spiral_env import EnvCfg, SpiralEnv
from vision_defaults import CAMERA_DISTANCE_SCALE, CAMERA_LOOKAT_OFFSET, CAMERA_MODE
from aruco_pipeline import make_camera_matrix_from_fovy


# Dataset flags
DATASET_ROOT = Path("./assets/dataset")
NUM_FRAMES = 200
IMAGE_SIZE = 2048
MARKER_FAMILY = "apriltag"  # "aruco" | "apriltag" | "both"

# Motion profile
ACTION_FREQ_HZ = 0.55
ACTION_NOISE_STD = 0.12
ACTION_GAIN = 0.9
ANNOTATIONS_FILE = "segments_gt.jsonl"


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
    (out_dir / ANNOTATIONS_FILE).unlink(missing_ok=True)


def _as_list(x: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


def _build_marker_geom_ids(env: SpiralEnv, marker_family: str) -> list[int]:
    ids: list[int] = []
    prefix = "apriltag" if marker_family == "apriltag" else "aruco"
    for i in range(deb.N_SEGMENTS):
        gid = deb._find_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_marker_{i:02d}")
        if gid < 0:
            raise RuntimeError(f"Missing marker geom: {prefix}_marker_{i:02d}")
        ids.append(int(gid))
    return ids


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v * 0.0
    return v / n


def _get_renderer_gl_camera(env: SpiralEnv) -> dict[str, np.ndarray | float] | None:
    renderer = getattr(env, "_renderer", None)
    if renderer is None or not hasattr(renderer, "scene"):
        return None
    scene = renderer.scene
    cams = getattr(scene, "camera", None)
    if cams is None:
        return None
    try:
        cam = cams[0]
    except Exception:
        return None

    pos = np.asarray(getattr(cam, "pos", np.zeros(3)), dtype=np.float64).reshape(-1)
    fwd = np.asarray(getattr(cam, "forward", np.zeros(3)), dtype=np.float64).reshape(-1)
    up = np.asarray(getattr(cam, "up", np.zeros(3)), dtype=np.float64).reshape(-1)
    if pos.size < 3 or fwd.size < 3 or up.size < 3:
        return None
    pos = pos[:3]
    forward = _norm(fwd[:3])
    up = _norm(up[:3])
    right = _norm(np.cross(forward, up))
    up = _norm(np.cross(right, forward))
    if float(np.linalg.norm(right)) < 1e-9 or float(np.linalg.norm(up)) < 1e-9:
        return None

    near = getattr(cam, "frustum_near", None)
    left = getattr(cam, "frustum_left", None)
    right_f = getattr(cam, "frustum_right", None)
    bottom = getattr(cam, "frustum_bottom", None)
    top = getattr(cam, "frustum_top", None)
    if near is None or left is None or right_f is None or bottom is None or top is None:
        return None
    try:
        near_f = float(near)
        left_f = float(left)
        rightf_f = float(right_f)
        bottom_f = float(bottom)
        top_f = float(top)
    except Exception:
        return None

    if abs(rightf_f - left_f) < 1e-9 or abs(top_f - bottom_f) < 1e-9 or near_f <= 0.0:
        return None
    return {
        "pos": pos.astype(np.float64),
        "forward": forward.astype(np.float64),
        "up": up.astype(np.float64),
        "right": right.astype(np.float64),
        "frustum_near": near_f,
        "frustum_left": left_f,
        "frustum_right": rightf_f,
        "frustum_bottom": bottom_f,
        "frustum_top": top_f,
    }


def _camera_world_pose(env: SpiralEnv) -> tuple[np.ndarray, np.ndarray]:
    gl_cam = _get_renderer_gl_camera(env)
    if gl_cam is not None:
        pos = np.asarray(gl_cam["pos"], dtype=np.float64).reshape(3)
        right = np.asarray(gl_cam["right"], dtype=np.float64).reshape(3)
        up = np.asarray(gl_cam["up"], dtype=np.float64).reshape(3)
        forward = np.asarray(gl_cam["forward"], dtype=np.float64).reshape(3)
        R_wc = np.stack([right, up, forward], axis=1).astype(np.float64)
        return pos, R_wc
    camera_id = int(env._camera_id) if hasattr(env, "_camera_id") else deb._find_id(  # type: ignore[attr-defined]
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top"
    )
    if camera_id < 0:
        raise RuntimeError("Failed to resolve camera pose from renderer or model camera.")
    cam_pos = np.asarray(env.data.cam_xpos[camera_id], dtype=np.float64).reshape(3)
    R_wc = np.asarray(env.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    return cam_pos, R_wc


def _camera_intrinsics_from_env(env: SpiralEnv, width: int, height: int) -> np.ndarray:
    gl_cam = _get_renderer_gl_camera(env)
    if gl_cam is not None:
        near = float(gl_cam["frustum_near"])
        left = float(gl_cam["frustum_left"])
        right = float(gl_cam["frustum_right"])
        bottom = float(gl_cam["frustum_bottom"])
        top = float(gl_cam["frustum_top"])
        fx = float(width) * near / max(1e-9, (right - left))
        fy = float(height) * near / max(1e-9, (top - bottom))
        cx = -float(width) * left / max(1e-9, (right - left))
        cy = float(height) * top / max(1e-9, (top - bottom))
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    fovy_deg: float | None = None
    if hasattr(env, "_mj_camera") and getattr(env._mj_camera, "fovy", None) is not None:  # type: ignore[attr-defined]
        try:
            fovy_deg = float(env._mj_camera.fovy)  # type: ignore[attr-defined]
        except Exception:
            fovy_deg = None
    if fovy_deg is None and str(getattr(env, "_camera_mode", "")).lower() == "fixed":
        camera_id = int(env._camera_id) if hasattr(env, "_camera_id") else deb._find_id(  # type: ignore[attr-defined]
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top"
        )
        if camera_id >= 0:
            fovy_deg = float(env.model.cam_fovy[camera_id])
    if fovy_deg is None:
        fovy_deg = float(getattr(deb, "CAM_TOP_FOVY", 60.0))
    return make_camera_matrix_from_fovy(width=width, height=height, fovy_deg=fovy_deg).astype(np.float64)


def _project_world_to_pixel(
    env: SpiralEnv,
    p_world: np.ndarray,
    K_fallback: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float, bool]:
    p_w = np.asarray(p_world, dtype=np.float64).reshape(3)
    gl_cam = _get_renderer_gl_camera(env)
    if gl_cam is not None:
        pos = np.asarray(gl_cam["pos"], dtype=np.float64).reshape(3)
        forward = np.asarray(gl_cam["forward"], dtype=np.float64).reshape(3)
        up = np.asarray(gl_cam["up"], dtype=np.float64).reshape(3)
        right = np.asarray(gl_cam["right"], dtype=np.float64).reshape(3)
        near = float(gl_cam["frustum_near"])
        left = float(gl_cam["frustum_left"])
        right_f = float(gl_cam["frustum_right"])
        bottom = float(gl_cam["frustum_bottom"])
        top = float(gl_cam["frustum_top"])

        rel = p_w - pos
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        z_cam = float(np.dot(rel, forward))
        if z_cam <= 1e-9:
            return np.array([np.nan, np.nan], dtype=np.float64), z_cam, False

        x_near = near * x_cam / z_cam
        y_near = near * y_cam / z_cam
        u = float(width) * (x_near - left) / max(1e-9, (right_f - left))
        v = float(height) * (top - y_near) / max(1e-9, (top - bottom))
        in_img = (0.0 <= u < float(width)) and (0.0 <= v < float(height))
        return np.array([u, v], dtype=np.float64), z_cam, bool(in_img)

    cam_pos, R_wc = _camera_world_pose(env)
    p_cam = R_wc.T @ (p_w - cam_pos)
    z = float(p_cam[2])
    if z <= 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64), z, False
    u = float(K_fallback[0, 0]) * (float(p_cam[0]) / z) + float(K_fallback[0, 2])
    # Image v axis points down; world/camera up is opposite sign.
    v = float(K_fallback[1, 1]) * (float(-p_cam[1]) / z) + float(K_fallback[1, 2])
    in_img = (0.0 <= u < float(width)) and (0.0 <= v < float(height))
    return np.array([u, v], dtype=np.float64), z, bool(in_img)


def _collect_frame_annotation(
    env: SpiralEnv,
    marker_family: str,
    marker_geom_ids: list[int],
    K: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    action_input: np.ndarray,
    info: dict[str, float],
    reward: float,
    terminated: bool,
    truncated: bool,
    image_path: Path,
) -> dict[str, object]:
    h = int(env._image_height)  # type: ignore[attr-defined]
    w = int(env._image_width)  # type: ignore[attr-defined]
    cam_pos, R_wc = _camera_world_pose(env)

    segments: list[dict[str, object]] = []
    for seg_id in range(deb.N_SEGMENTS):
        gid = int(marker_geom_ids[seg_id])
        bid = deb._find_id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"seg_{seg_id:02d}")
        if bid < 0:
            raise RuntimeError(f"Missing segment body id: seg_{seg_id:02d}")

        marker_center_world = np.asarray(env.data.geom_xpos[gid], dtype=np.float64).reshape(3)
        marker_R_world = np.asarray(env.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        marker_size = np.asarray(env.model.geom_size[gid], dtype=np.float64).reshape(3)
        sx, sy, sz = float(marker_size[0]), float(marker_size[1]), float(marker_size[2])
        local_corners = np.array(
            [
                [-sx, sy, sz],
                [sx, sy, sz],
                [sx, -sy, sz],
                [-sx, -sy, sz],
            ],
            dtype=np.float64,
        )
        corners_world = (marker_R_world @ local_corners.T).T + marker_center_world.reshape(1, 3)

        center_px, center_depth, center_in_img = _project_world_to_pixel(
            env=env,
            p_world=marker_center_world,
            K_fallback=K,
            width=w,
            height=h,
        )

        corners_px: list[list[float]] = []
        corner_depths: list[float] = []
        corners_in = 0
        for c_w in corners_world:
            p, z, in_img = _project_world_to_pixel(
                env=env,
                p_world=c_w,
                K_fallback=K,
                width=w,
                height=h,
            )
            corners_px.append(_as_list(p))
            corner_depths.append(float(z))
            if in_img:
                corners_in += 1

        joint_qpos = None
        joint_qvel = None
        if seg_id > 0:
            qadr = int(env.joint_qposadrs[seg_id - 1])
            vadr = int(env.joint_qveladrs[seg_id - 1])
            joint_qpos = float(env.data.qpos[qadr])
            joint_qvel = float(env.data.qvel[vadr])

        segments.append(
            {
                "id": int(seg_id),
                "marker_id": int(seg_id),
                "body_name": f"seg_{seg_id:02d}",
                "marker_geom_name": f"{marker_family}_marker_{seg_id:02d}",
                "joint_qpos_rad": joint_qpos,
                "joint_qvel_rad_s": joint_qvel,
                "body_center_world_m": _as_list(np.asarray(env.data.xpos[bid], dtype=np.float64).reshape(3)),
                "body_rot_world_rowmajor": _as_list(np.asarray(env.data.xmat[bid], dtype=np.float64).reshape(9)),
                "marker_center_world_m": _as_list(marker_center_world),
                "marker_rot_world_rowmajor": _as_list(marker_R_world.reshape(9)),
                "marker_size_half_m": _as_list(marker_size),
                "marker_corners_world_m_tl_tr_br_bl": [_as_list(c) for c in corners_world],
                "marker_center_px": _as_list(center_px),
                "marker_center_depth_m": float(center_depth),
                "marker_center_in_image": bool(center_in_img),
                "marker_corners_px_tl_tr_br_bl": corners_px,
                "marker_corners_depth_m_tl_tr_br_bl": corner_depths,
                "num_corners_in_image": int(corners_in),
                "fully_visible": bool(corners_in == 4),
            }
        )

    ball_pos = np.asarray(env.data.xpos[env.ball_body_id], dtype=np.float64).reshape(3)
    base_pos = np.asarray(env.data.xpos[env.base_body_id], dtype=np.float64).reshape(3)
    tip_pos = np.asarray(env.data.xpos[env.tip_body_id], dtype=np.float64).reshape(3)
    line: dict[str, object] = {
        "frame_idx": int(frame_idx),
        "timestamp_s": float(timestamp_s),
        "image_path": str(image_path).replace("\\", "/"),
        "family": marker_family,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "action_input": _as_list(np.asarray(action_input, dtype=np.float64).reshape(2)),
        "action_applied": [float(info.get("a0", 0.0)), float(info.get("a1", 0.0))],
        "control": {
            "T_left": float(info.get("T_left", 0.0)),
            "T_right": float(info.get("T_right", 0.0)),
            "Tsum": float(info.get("Tsum", 0.0)),
        },
        "task": {
            "d_tip_ball": float(info.get("d_tip_ball", np.nan)),
            "d_ball_base": float(info.get("d_ball_base", np.nan)),
            "wrap_count": int(info.get("wrap_count", 0)),
            "wrap_frac": float(info.get("wrap_frac", 0.0)),
        },
        "camera": {
            "name": "top",
            "mode": str(env.cfg.camera_mode),
            "image_width": int(w),
            "image_height": int(h),
            "K": _as_list(K.reshape(9)),
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "cam_pos_world_m": _as_list(cam_pos),
            "cam_rot_world_rowmajor": _as_list(R_wc.reshape(9)),
            "projection_mode": {"method": "renderer_gl_frustum"},
        },
        "scene": {
            "ball_pos_world_m": _as_list(ball_pos),
            "base_pos_world_m": _as_list(base_pos),
            "tip_pos_world_m": _as_list(tip_pos),
        },
        "segments": segments,
    }
    return line


def _generate_family_dataset(marker_family: str, num_frames: int, image_size: int) -> None:
    out_dir = DATASET_ROOT / marker_family
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_frames(out_dir)

    env = _build_env_for_family(marker_family=marker_family, image_size=image_size)
    rng = np.random.default_rng(123)
    marker_geom_ids = _build_marker_geom_ids(env, marker_family=marker_family)
    K: np.ndarray | None = None

    try:
        dt = float(env.cfg.control_dt)
        ann_path = out_dir / ANNOTATIONS_FILE
        with ann_path.open("w", encoding="utf-8") as ann_file:
            for step_idx in range(num_frames):
                action = _demo_action(step_idx, dt, rng)
                _, reward, terminated, truncated, info = env.step(action)
                frame = _to_uint8_rgb(env.render_camera())
                if K is None:
                    K = _camera_intrinsics_from_env(
                        env=env,
                        width=int(frame.shape[1]),
                        height=int(frame.shape[0]),
                    )
                frame_path = out_dir / f"frame_{step_idx:04d}.png"
                _write_rgb_png(frame_path, frame)
                assert K is not None
                ann_line = _collect_frame_annotation(
                    env=env,
                    marker_geom_ids=marker_geom_ids,
                    marker_family=marker_family,
                    K=K,
                    frame_idx=step_idx,
                    timestamp_s=float((step_idx + 1) * dt),
                    action_input=action,
                    info=info,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    image_path=frame_path,
                )
                ann_file.write(json.dumps(ann_line, ensure_ascii=False) + "\n")
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
        print(f"  annotations: {DATASET_ROOT / family / ANNOTATIONS_FILE}")

    print("Dataset generation complete.")
    print(f"Root: {DATASET_ROOT.resolve()}")


if __name__ == "__main__":
    main()
