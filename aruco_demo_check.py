from __future__ import annotations

# README
# 1) Run: python aruco_demo_check.py
# 2) Script loads frames from assets/dataset/aruco and runs ArUco pipeline.
# 3) It prints detected/missing IDs, state vector shape, and sample segment state.

from pathlib import Path
from typing import Any

import numpy as np

from aruco_pipeline import ArucoConfig, ArucoPipeline, make_camera_matrix_from_fovy
from marker_defaults import CAM_TOP_FOVY, MARKER_LENGTH_M, MARKER_LENGTH_PER_ID


DATASET_DIR = Path("./assets/dataset/aruco")
N_STEPS_LIMIT: int | None = None
SAVE_DEBUG_IMAGES = True
USE_POSE3D = True
CALIBRATION_NPZ = Path("./assets/camera_calibration_top.npz")
CONTROL_DT = 0.02


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


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for dataset image loading.") from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_camera_calibration_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        data = np.load(path)
        K = np.asarray(data["camera_matrix"], dtype=np.float32).reshape(3, 3)
        D = np.asarray(data["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
        return K, D
    except Exception:
        return None


def main() -> None:
    frame_paths = sorted(DATASET_DIR.glob("frame_*.png"))
    if N_STEPS_LIMIT is not None:
        frame_paths = frame_paths[: int(max(0, N_STEPS_LIMIT))]
    if not frame_paths:
        print(f"No dataset frames found in: {DATASET_DIR}")
        print("Run: python make_dataset.py")
        return

    first_frame = _read_rgb(frame_paths[0])
    h, w = int(first_frame.shape[0]), int(first_frame.shape[1])

    calib = _load_camera_calibration_npz(CALIBRATION_NPZ) if USE_POSE3D else None
    if USE_POSE3D and calib is not None:
        camera_matrix, dist_coeffs = calib
        calib_src = f"npz:{CALIBRATION_NPZ}"
    elif USE_POSE3D:
        camera_matrix = make_camera_matrix_from_fovy(
            width=w,
            height=h,
            fovy_deg=float(CAM_TOP_FOVY),
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        calib_src = "fovy_fallback"
    else:
        camera_matrix = None
        dist_coeffs = None
        calib_src = "disabled"

    ar_cfg = ArucoConfig(
        dict_name="DICT_4X4_50",
        marker_length_m=float(MARKER_LENGTH_M),
        marker_length_per_id=dict(MARKER_LENGTH_PER_ID),
        output_mode="pose3d",
        expected_ids=list(range(19)),
        input_color="RGB",
        useRefineDetectedMarkers=True,
        enable_kalman=True,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        include_tracking_features=True,
        alive_k_frames=3,
        include_age_k=True,
        age_norm_mode="frames_norm",
        require_world_for_pose3d=True,
    )
    pipeline = ArucoPipeline(ar_cfg)

    debug_dir = Path("./debug")
    if SAVE_DEBUG_IMAGES:
        debug_dir.mkdir(parents=True, exist_ok=True)

    last_state = None
    last_info = None
    last_frame = first_frame
    detected_counts: list[int] = []

    for step_idx, frame_path in enumerate(frame_paths):
        frame = _read_rgb(frame_path) if step_idx > 0 else first_frame
        timestamp_s = float(step_idx) * float(CONTROL_DT)
        state_vec, info = pipeline.step(frame, timestamp=timestamp_s, gt_centers_px=None)
        detected_counts.append(int(info.get("num_detected", 0)))
        last_state = state_vec
        last_info = info
        last_frame = frame

        if SAVE_DEBUG_IMAGES and step_idx % 4 == 0:
            dbg = _draw_debug(frame, info)
            try:
                import cv2  # type: ignore

                bgr = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"aruco_debug_{step_idx:03d}.png"), bgr)
            except Exception:
                pass

    if last_state is None or last_info is None:
        print("No frames processed.")
        return

    print("=== ArUco Demo Summary ===")
    print(f"mode: {'pose3d' if USE_POSE3D else 'image2d'}")
    print(f"dataset_dir: {DATASET_DIR}")
    print(f"camera_calibration_source: {calib_src}")
    print(f"detected_ids: {last_info['detected_ids']}")
    print(f"missing_ids:  {last_info['missing_ids']}")
    print(f"state_vec shape: {tuple(last_state.shape)}")
    if USE_POSE3D and last_state.size >= 10:
        print(f"segment_01 state head: {last_state[:10].tolist()}")
    elif last_state.size >= 5:
        print(f"segment_01 state head: {last_state[:5].tolist()}")
    print(f"world_missing: {last_info.get('world_missing', True)}")
    print(f"mean_reprojection_error: {last_info.get('mean_reprojection_error')}")
    print(f"track_consistency_mae_px: {last_info.get('track_consistency_mae_px')}")
    print(f"track_consistency_rmse_px: {last_info.get('track_consistency_rmse_px')}")
    print(f"det_recall_18: {last_info.get('det_recall_18')}")
    print(f"gate_reject_rate_on_detected: {last_info.get('gate_reject_rate_on_detected')}")
    print(f"track_alive_recall_18: {last_info.get('track_alive_recall_18')} (alive_k={last_info.get('alive_k_frames')})")
    print(f"gt_mae_px: {last_info.get('gt_mae_px')}")
    print(f"gt_rmse_px: {last_info.get('gt_rmse_px')}")
    print(f"gt_recall: {last_info.get('gt_recall')}")
    print(f"last_frame_mean_intensity: {float(np.mean(last_frame)):.3f}")
    if detected_counts:
        print(f"detected_ids_avg: {float(np.mean(detected_counts)):.2f}")
        print(f"detected_ids_max: {int(np.max(detected_counts))}")


if __name__ == "__main__":
    main()
