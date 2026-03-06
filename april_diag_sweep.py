from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from april_pipeline import AprilConfig, AprilPipeline, AprilTagDetector, make_camera_matrix_from_fovy
from marker_defaults import CAM_TOP_FOVY, MARKER_LENGTH_M, MARKER_LENGTH_PER_ID

DATASET_DIR = Path("./assets/dataset/apriltag")
DEBUG_DIR = Path("./debug")
CALIBRATION_NPZ = Path("./assets/camera_calibration_top.npz")
CONTROL_DT = 0.02

# Sweep grid requested by supervisor comments.
DECODE_SWEEP = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
QUAD_SIGMA_VALUES = [0.0, 0.8]


def _read_rgb(path: Path) -> np.ndarray:
    import cv2  # type: ignore

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


def _build_cfg(
    width: int,
    height: int,
    decode_sharpening: float,
    quad_sigma: float,
) -> AprilConfig:
    calib = _load_camera_calibration_npz(CALIBRATION_NPZ)
    if calib is not None:
        camera_matrix, dist_coeffs = calib
    else:
        camera_matrix = make_camera_matrix_from_fovy(width=width, height=height, fovy_deg=float(CAM_TOP_FOVY))
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    return AprilConfig(
        apriltag_family="tag36h11",
        marker_length_m=float(MARKER_LENGTH_M),
        marker_length_per_id=dict(MARKER_LENGTH_PER_ID),
        output_mode="pose3d",
        expected_ids=list(range(19)),
        input_color="RGB",
        quad_decimate=1.0,
        quad_sigma=float(quad_sigma),
        refine_edges=True,
        decode_sharpening=float(decode_sharpening),
        enable_kalman=True,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        include_tracking_features=True,
        alive_k_frames=3,
        include_age_k=True,
        age_norm_mode="frames_norm",
        require_world_for_pose3d=True,
    )


def _gray_stats(gray: np.ndarray) -> dict[str, float]:
    arr = np.asarray(gray, dtype=np.uint8)
    p1, p50, p99 = np.percentile(arr, [1.0, 50.0, 99.0])
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "p99_minus_p01": float(p99 - p1),
    }


def _save_gray_debug(frame_rgb: np.ndarray, cfg: AprilConfig) -> tuple[Path, dict[str, float], list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    import cv2  # type: ignore

    detector = AprilTagDetector(cfg)
    gray = detector._to_gray(frame_rgb)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEBUG_DIR / "debug_in_gray.png"
    ok = cv2.imwrite(str(out_path), gray)
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")

    stats = _gray_stats(gray)

    # Raw detector output before custom post-filters in AprilTagDetector.detect.
    raw_tags = detector._detector.detect(gray, estimate_tag_pose=False)
    raw_items: list[dict[str, Any]] = []
    for tag in raw_tags:
        raw_items.append(
            {
                "id": int(getattr(tag, "tag_id", -1)),
                "decision_margin": float(getattr(tag, "decision_margin", 0.0)),
                "hamming": int(getattr(tag, "hamming", -1)),
            }
        )

    default_filtered = detector.detect(frame_rgb)

    # Disable custom post-filters to isolate base detector behavior.
    no_filter_cfg = replace(
        cfg,
        output_mode="image2d",
        min_side_px=0.0,
        min_square_ratio=0.0,
        min_detection_score=0.0,
        min_decision_margin=-1.0,
        max_reprojection_error_px=1e9,
    )
    detector_no_filter = AprilTagDetector(no_filter_cfg)
    no_filter_out = detector_no_filter.detect(frame_rgb)

    return out_path, stats, raw_items, default_filtered, no_filter_out


def _run_sweep(frames: list[np.ndarray]) -> list[dict[str, float]]:
    if not frames:
        return []
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    rows: list[dict[str, float]] = []

    for quad_sigma in QUAD_SIGMA_VALUES:
        for decode_sharpening in DECODE_SWEEP:
            cfg = _build_cfg(
                width=w,
                height=h,
                decode_sharpening=float(decode_sharpening),
                quad_sigma=float(quad_sigma),
            )
            pipeline = AprilPipeline(cfg)
            recalls = []
            for i, frame in enumerate(frames):
                _, info = pipeline.step(frame, timestamp=float(i) * float(CONTROL_DT), gt_centers_px=None)
                recalls.append(float(info.get("det_recall_18", 0.0)))

            arr = np.asarray(recalls, dtype=np.float32)
            rows.append(
                {
                    "quad_sigma": float(quad_sigma),
                    "decode_sharpening": float(decode_sharpening),
                    "det_recall_18_mean": float(np.mean(arr)),
                    "det_recall_18_median": float(np.median(arr)),
                    "det_recall_18_last": float(arr[-1]),
                }
            )
    return rows


def _save_sweep_csv(rows: list[dict[str, float]]) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DEBUG_DIR / "april_decode_sweep.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "quad_sigma",
                "decode_sharpening",
                "det_recall_18_mean",
                "det_recall_18_median",
                "det_recall_18_last",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_csv


def _print_sweep_table(rows: list[dict[str, float]]) -> None:
    print("\n=== Sweep Table (decode_sharpening -> det_recall_18_mean) ===")
    grouped: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(float(row["quad_sigma"]), []).append(row)

    for q in sorted(grouped.keys()):
        print(f"quad_sigma={q}")
        print("decode_sharpening | det_recall_18_mean")
        for row in sorted(grouped[q], key=lambda x: float(x["decode_sharpening"])):
            ds = float(row["decode_sharpening"])
            recall = float(row["det_recall_18_mean"])
            print(f"{ds:>16.2f} | {recall:.6f}")
        best = max(grouped[q], key=lambda x: float(x["det_recall_18_mean"]))
        print(
            "best:",
            f"decode_sharpening={float(best['decode_sharpening']):.2f},",
            f"det_recall_18_mean={float(best['det_recall_18_mean']):.6f}",
        )
        print()


def main() -> None:
    frame_paths = sorted(DATASET_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {DATASET_DIR}")

    frames = [_read_rgb(p) for p in frame_paths]
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    cfg = _build_cfg(width=w, height=h, decode_sharpening=0.25, quad_sigma=0.0)

    out_path, stats, raw_items, default_filtered, no_filter_out = _save_gray_debug(frames[0], cfg)

    print("=== Gray Input Check ===")
    print(f"saved: {out_path}")
    print(
        "stats:",
        ", ".join(f"{k}={v:.3f}" for k, v in stats.items()),
    )

    raw_ids = sorted(int(x["id"]) for x in raw_items if int(x["id"]) >= 0)
    raw_margins = [float(x["decision_margin"]) for x in raw_items if int(x["id"]) >= 0]
    print("\n=== One-Frame Detection (first frame) ===")
    print(f"raw_detector_count: {len(raw_ids)}")
    print(f"raw_detector_ids: {raw_ids}")
    if raw_margins:
        print(f"raw_detector_margin_mean: {float(np.mean(raw_margins)):.3f}")
        print(f"raw_detector_margin_min: {float(np.min(raw_margins)):.3f}")
    print(f"default_filtered_count: {len(default_filtered)}")
    print(f"default_filtered_ids: {sorted(default_filtered.keys())}")
    print(f"no_post_filters_count: {len(no_filter_out)}")
    print(f"no_post_filters_ids: {sorted(no_filter_out.keys())}")

    rows = _run_sweep(frames)
    csv_path = _save_sweep_csv(rows)
    _print_sweep_table(rows)
    print(f"saved_csv: {csv_path}")


if __name__ == "__main__":
    main()
