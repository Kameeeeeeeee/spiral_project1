from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from april_pipeline import AprilConfig, AprilPipeline
from marker_defaults import MARKER_LENGTH_M, MARKER_LENGTH_PER_ID


DATASET_DIR = Path("./assets/dataset/apriltag")
GT_PATH = DATASET_DIR / "segments_gt.jsonl"
OUT_ROOT = Path("./debug")
EXAMPLES_LIMIT = 30
SAVE_ALL_OVERLAY_FRAMES = True


@dataclass
class FrameGT:
    frame_idx: int
    timestamp_s: float
    image_path: Path
    centers_px: dict[int, np.ndarray]
    visible_ids_1_18: list[int]
    K: np.ndarray


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required.") from exc
    return cv2


def _read_rgb(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_rmse(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.sqrt(np.nanmean(arr * arr)))


def _fit_affine_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, float]:
    s = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if s.shape[0] < 3 or d.shape[0] != s.shape[0]:
        return np.eye(3, dtype=np.float64), float("nan")
    X = np.hstack([s, np.ones((s.shape[0], 1), dtype=np.float64)])
    A = np.linalg.lstsq(X, d, rcond=None)[0]  # (3,2)
    pred = X @ A
    err = np.linalg.norm(pred - d, axis=1)
    T = np.array(
        [
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T, float(np.mean(err))


def _apply_affine_2d(T: np.ndarray, p_xy: np.ndarray) -> np.ndarray:
    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    x = float(T[0, 0]) * float(p[0]) + float(T[0, 1]) * float(p[1]) + float(T[0, 2])
    y = float(T[1, 0]) * float(p[0]) + float(T[1, 1]) * float(p[1]) + float(T[1, 2])
    return np.array([x, y], dtype=np.float64)


def _load_gt(path: Path) -> tuple[list[FrameGT], np.ndarray]:
    if not path.exists():
        raise RuntimeError(f"GT file not found: {path}")
    frames: list[FrameGT] = []
    K_first: np.ndarray | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            frame_idx = int(item["frame_idx"])
            timestamp_s = float(item.get("timestamp_s", 0.02 * (frame_idx + 1)))
            image_path = Path(str(item["image_path"]))

            camera = item.get("camera", {})
            K_flat = np.asarray(camera.get("K", []), dtype=np.float64).reshape(-1)
            if K_flat.size != 9:
                raise RuntimeError("Invalid camera K in GT JSONL.")
            K = K_flat.reshape(3, 3)
            if K_first is None:
                K_first = K.copy()

            centers: dict[int, np.ndarray] = {}
            visible_ids: list[int] = []
            for seg in item.get("segments", []):
                sid = int(seg.get("id", -1))
                if sid < 1 or sid > 18:
                    continue
                if not bool(seg.get("marker_center_in_image", False)):
                    continue
                c = np.asarray(seg.get("marker_center_px", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
                if c.size != 2 or not np.all(np.isfinite(c)):
                    continue
                centers[sid] = c.astype(np.float64)
                visible_ids.append(sid)

            frames.append(
                FrameGT(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    image_path=image_path,
                    centers_px=centers,
                    visible_ids_1_18=sorted(visible_ids),
                    K=K,
                )
            )
    if not frames or K_first is None:
        raise RuntimeError(f"Empty GT file: {path}")
    return frames, K_first


class LengthEstimator:
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = int(n_nodes)
        self.samples: list[list[float]] = [[] for _ in range(max(0, self.n_nodes - 1))]
        self.lengths = np.full((max(0, self.n_nodes - 1),), 0.012, dtype=np.float64)

    def update(self, points: np.ndarray, anchors: np.ndarray) -> None:
        for i in range(self.n_nodes - 1):
            if bool(anchors[i]) and bool(anchors[i + 1]):
                p0 = points[i]
                p1 = points[i + 1]
                if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                    d = float(np.linalg.norm(p1 - p0))
                    if 1e-5 < d < 1.0:
                        self.samples[i].append(d)
                        if len(self.samples[i]) > 200:
                            self.samples[i].pop(0)
        for i in range(self.n_nodes - 1):
            if self.samples[i]:
                self.lengths[i] = float(np.median(np.asarray(self.samples[i], dtype=np.float64)))
        for i in range(self.n_nodes - 1):
            if not np.isfinite(self.lengths[i]) or self.lengths[i] <= 1e-5:
                left = self.lengths[i - 1] if i - 1 >= 0 else np.nan
                right = self.lengths[i + 1] if i + 1 < self.n_nodes - 1 else np.nan
                if np.isfinite(left) and np.isfinite(right):
                    self.lengths[i] = float(0.5 * (left + right))
                elif np.isfinite(left):
                    self.lengths[i] = float(left)
                elif np.isfinite(right):
                    self.lengths[i] = float(right)
                else:
                    self.lengths[i] = 0.012


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _build_initial_positions(
    points_obs: np.ndarray,
    anchors: np.ndarray,
    lengths: np.ndarray,
    prev_positions: np.ndarray | None,
) -> np.ndarray:
    pts = np.asarray(points_obs, dtype=np.float64).copy()
    n = pts.shape[0]
    valid = np.all(np.isfinite(pts), axis=1)

    if prev_positions is not None and prev_positions.shape == pts.shape:
        for i in range(n):
            if not valid[i] and np.all(np.isfinite(prev_positions[i])):
                pts[i] = prev_positions[i]
                valid[i] = True

    if not np.any(valid):
        pts[0] = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        valid[0] = True

    valid_idx = np.where(valid)[0].tolist()
    first = int(valid_idx[0])
    for i in range(first - 1, -1, -1):
        guess_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        if i + 2 < n and np.all(np.isfinite(pts[i + 2])):
            guess_dir = pts[i + 1] - pts[i + 2]
        guess_dir = _normalize(guess_dir, np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        pts[i] = pts[i + 1] + guess_dir * float(lengths[i])
        valid[i] = True

    valid_idx = np.where(valid)[0].tolist()
    last = int(valid_idx[-1])
    for i in range(last + 1, n):
        guess_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if i - 2 >= 0 and np.all(np.isfinite(pts[i - 2])):
            guess_dir = pts[i - 1] - pts[i - 2]
        guess_dir = _normalize(guess_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        pts[i] = pts[i - 1] + guess_dir * float(lengths[i - 1])
        valid[i] = True

    valid_idx = np.where(valid)[0].tolist()
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        if b - a <= 1:
            continue
        seg_lengths = lengths[a:b]
        total = float(np.sum(seg_lengths))
        vec = pts[b] - pts[a]
        if total <= 1e-9:
            continue
        cum = 0.0
        for i in range(a + 1, b):
            cum += float(lengths[i - 1])
            pts[i] = pts[a] + vec * (cum / total)

    pts[anchors] = points_obs[anchors]
    return pts


def _solve_two_anchor_subchain(
    pts: np.ndarray,
    lengths: np.ndarray,
    a: int,
    b: int,
    iters: int = 12,
) -> None:
    if b - a <= 1:
        return
    p = pts[a : b + 1].copy()
    seg_len = lengths[a:b].copy()
    n = p.shape[0] - 1
    root = p[0].copy()
    end = p[-1].copy()

    total_len = float(np.sum(seg_len))
    d = float(np.linalg.norm(end - root))
    if d > total_len + 1e-9:
        direction = _normalize(end - root, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        acc = 0.0
        p[0] = root
        for i in range(1, n + 1):
            acc += float(seg_len[i - 1])
            p[i] = root + direction * acc
        pts[a : b + 1] = p
        return

    for _ in range(iters):
        p[-1] = end
        for i in range(n - 1, -1, -1):
            direction = _normalize(p[i] - p[i + 1], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            p[i] = p[i + 1] + direction * float(seg_len[i])

        p[0] = root
        for i in range(1, n + 1):
            direction = _normalize(p[i] - p[i - 1], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            p[i] = p[i - 1] + direction * float(seg_len[i - 1])

    p[0] = root
    p[-1] = end
    pts[a : b + 1] = p


def _fabrik_fill(
    points_obs: np.ndarray,
    anchors: np.ndarray,
    lengths: np.ndarray,
    prev_positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = points_obs.shape[0]
    pts = _build_initial_positions(points_obs=points_obs, anchors=anchors, lengths=lengths, prev_positions=prev_positions)
    anchor_idx = np.where(anchors)[0].tolist()

    if len(anchor_idx) >= 2:
        for a, b in zip(anchor_idx[:-1], anchor_idx[1:]):
            if b - a > 1:
                _solve_two_anchor_subchain(pts=pts, lengths=lengths, a=int(a), b=int(b), iters=15)
    elif len(anchor_idx) == 1:
        k = int(anchor_idx[0])
        for i in range(k + 1, n):
            direction = _normalize(pts[i] - pts[i - 1], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            pts[i] = pts[i - 1] + direction * float(lengths[i - 1])
        for i in range(k - 1, -1, -1):
            direction = _normalize(pts[i] - pts[i + 1], np.array([-1.0, 0.0, 0.0], dtype=np.float64))
            pts[i] = pts[i + 1] + direction * float(lengths[i])

    pts[anchors] = points_obs[anchors]
    recovered_mask = (~anchors) & np.all(np.isfinite(pts), axis=1)
    return pts, recovered_mask


def _project_cam_point_to_px(cam_point: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    p = np.asarray(cam_point, dtype=np.float64).reshape(3)
    z = float(p[2])
    if z <= 1e-6:
        return None
    u = float(K[0, 0]) * (float(p[0]) / z) + float(K[0, 2])
    v = float(K[1, 1]) * (float(p[1]) / z) + float(K[1, 2])
    return np.array([u, v], dtype=np.float64)


def _draw_example(
    frame_rgb: np.ndarray,
    gt_centers: dict[int, np.ndarray],
    det_centers: dict[int, np.ndarray],
    rec_centers: dict[int, np.ndarray],
    recovered_ids: list[int],
    out_path: Path,
) -> None:
    cv2 = _import_cv2()
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for sid, c in gt_centers.items():
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)
    for sid, c in det_centers.items():
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        cv2.circle(img, (x, y), 3, (255, 80, 80), -1, cv2.LINE_AA)
    for sid in recovered_ids:
        if sid not in rec_centers:
            continue
        c = rec_centers[sid]
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        cv2.circle(img, (x, y), 4, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{sid}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {out_path}")
    out_path.write_bytes(enc.tobytes())


def _draw_april_fabrik_overlay(
    frame_rgb: np.ndarray,
    det_centers: dict[int, np.ndarray],
    rec_centers: dict[int, np.ndarray],
    recovered_ids: list[int],
    out_path: Path,
) -> None:
    cv2 = _import_cv2()
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Found by AprilTag detector: green filled points.
    for sid, c in det_centers.items():
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        cv2.circle(img, (x, y), 4, (40, 220, 40), -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"A{sid}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )

    # Restored by FABRIK: yellow rings (only for missing AprilTag IDs).
    for sid in recovered_ids:
        c = rec_centers.get(sid)
        if c is None:
            continue
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        cv2.circle(img, (x, y), 7, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x - 6, y), (x + 6, y), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x, y - 6), (x, y + 6), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"F{sid}",
            (x + 6, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        "A=id found by AprilTag, F=id filled by FABRIK",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {out_path}")
    out_path.write_bytes(enc.tobytes())


def _build_pipeline(K: np.ndarray) -> AprilPipeline:
    cfg = AprilConfig(
        apriltag_family="tag36h11",
        marker_length_m=float(MARKER_LENGTH_M),
        marker_length_per_id=dict(MARKER_LENGTH_PER_ID),
        output_mode="pose3d",
        expected_ids=list(range(19)),
        input_color="RGB",
        quad_decimate=1.0,
        refine_edges=True,
        enable_kalman=True,
        camera_matrix=K.astype(np.float32),
        dist_coeffs=np.zeros((5, 1), dtype=np.float32),
        include_tracking_features=True,
        alive_k_frames=3,
        include_age_k=True,
        age_norm_mode="frames_norm",
        require_world_for_pose3d=True,
    )
    return AprilPipeline(cfg)


def _estimate_gt_alignment_affine(
    frame_paths: list[Path],
    gt_by_name: dict[str, FrameGT],
    pipeline: AprilPipeline,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    pairs = 0
    for fp in frame_paths:
        gt = gt_by_name[fp.name]
        frame_rgb = _read_rgb(fp)
        dets = pipeline.detector.detect(frame_rgb)
        for sid in range(1, 19):
            if sid not in gt.centers_px or sid not in dets:
                continue
            c_gt = np.asarray(gt.centers_px[sid], dtype=np.float64).reshape(2)
            c_det = np.asarray(dets[sid]["center_px"], dtype=np.float64).reshape(2)
            if np.all(np.isfinite(c_gt)) and np.all(np.isfinite(c_det)):
                src.append(c_gt)
                dst.append(c_det)
                pairs += 1
    if pairs < 30:
        return None, {"pairs": pairs, "enabled": False, "reason": "not_enough_pairs"}

    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    T, mae = _fit_affine_2d(src_arr, dst_arr)
    if not np.isfinite(mae):
        return None, {"pairs": pairs, "enabled": False, "reason": "fit_failed"}
    enabled = bool(mae <= 5.0)
    return (
        T if enabled else None,
        {
            "pairs": pairs,
            "enabled": enabled,
            "fit_mae_px": float(mae),
            "transform_rowmajor_3x3": T.reshape(-1).tolist(),
            "reason": "ok" if enabled else "residual_too_large",
        },
    )


def main() -> None:
    frame_paths = sorted(DATASET_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {DATASET_DIR}")

    gt_rows, K = _load_gt(GT_PATH)
    gt_by_name = {Path(x.image_path).name: x for x in gt_rows}
    for fp in frame_paths:
        if fp.name not in gt_by_name:
            raise RuntimeError(f"Missing GT for frame: {fp.name}")

    out_dir = (OUT_ROOT / f"april_fabrik_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "recovered_examples"
    overlay_dir = out_dir / "april_plus_fabrik_frames"
    examples_saved = 0

    pipeline = _build_pipeline(K=K)
    align_probe_pipeline = _build_pipeline(K=K)
    gt_affine_T, gt_alignment_info = _estimate_gt_alignment_affine(
        frame_paths=frame_paths,
        gt_by_name=gt_by_name,
        pipeline=align_probe_pipeline,
    )
    length_estimator = LengthEstimator(n_nodes=18)
    prev_filled: np.ndarray | None = None

    rows: list[dict[str, Any]] = []
    base_gt_mae_vals: list[float] = []
    base_gt_rmse_vals: list[float] = []
    base_gt_recall_vals: list[float] = []
    base_det_recall_vals: list[float] = []
    rec_gt_mae_vals: list[float] = []
    rec_gt_rmse_vals: list[float] = []
    rec_gt_recall_vals: list[float] = []
    base_gt_mae_raw_vals: list[float] = []
    base_gt_rmse_raw_vals: list[float] = []
    rec_gt_mae_raw_vals: list[float] = []
    rec_gt_rmse_raw_vals: list[float] = []
    gt_consistency_warning: str | None = None

    for i, frame_path in enumerate(frame_paths):
        gt = gt_by_name[frame_path.name]
        frame_rgb = _read_rgb(frame_path)

        state_vec, info = pipeline.step(
            frame_rgb,
            timestamp=float(gt.timestamp_s),
            gt_centers_px=gt.centers_px,
        )
        _ = state_vec

        detections = pipeline.detector.detect(frame_rgb)
        det_centers_px: dict[int, np.ndarray] = {}
        det_tvec = np.full((18, 3), np.nan, dtype=np.float64)
        anchors = np.zeros((18,), dtype=bool)
        for sid in range(1, 19):
            det = detections.get(sid)
            if det is None:
                continue
            center = np.asarray(det.get("center_px", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
            tvec = det.get("tvec")
            if center.size == 2 and np.all(np.isfinite(center)):
                det_centers_px[sid] = center.astype(np.float64)
            if tvec is not None:
                tv = np.asarray(tvec, dtype=np.float64).reshape(-1)
                if tv.size == 3 and np.all(np.isfinite(tv)):
                    det_tvec[sid - 1] = tv
                    anchors[sid - 1] = True

        if i == 0:
            # GT sanity check: if direct GT-vs-detector error is huge but affine fit is tiny,
            # GT pixel projection is likely inconsistent (wrong camera intrinsics/signs).
            p_gt: list[np.ndarray] = []
            p_det: list[np.ndarray] = []
            for sid in range(1, 19):
                if sid in gt.centers_px and sid in det_centers_px:
                    p_gt.append(gt.centers_px[sid])
                    p_det.append(det_centers_px[sid])
            if len(p_gt) >= 6:
                gt_arr = np.asarray(p_gt, dtype=np.float64)
                det_arr = np.asarray(p_det, dtype=np.float64)
                direct = float(np.mean(np.linalg.norm(det_arr - gt_arr, axis=1)))
                _, affine_mae = _fit_affine_2d(src_xy=gt_arr, dst_xy=det_arr)
                if np.isfinite(direct) and np.isfinite(affine_mae):
                    if direct > 80.0 and affine_mae < 5.0:
                        gt_consistency_warning = (
                            "GT pixel projection mismatch detected: large direct GT-vs-detector error "
                            f"(~{direct:.1f}px) but tiny affine residual (~{affine_mae:.2f}px). "
                            "Regenerate dataset annotations with fixed make_dataset.py."
                        )

        length_estimator.update(points=det_tvec, anchors=anchors)
        filled_tvec, recovered_mask = _fabrik_fill(
            points_obs=det_tvec,
            anchors=anchors,
            lengths=length_estimator.lengths,
            prev_positions=prev_filled,
        )
        prev_filled = filled_tvec.copy()

        rec_centers_px: dict[int, np.ndarray] = {}
        for sid in range(1, 19):
            idx = sid - 1
            if anchors[idx] and sid in det_centers_px:
                rec_centers_px[sid] = det_centers_px[sid]
                continue
            if not bool(recovered_mask[idx]):
                continue
            c = _project_cam_point_to_px(filled_tvec[idx], K=K)
            if c is not None and np.all(np.isfinite(c)):
                rec_centers_px[sid] = c

        visible_ids = list(gt.visible_ids_1_18)
        gt_eval_centers: dict[int, np.ndarray] = {}
        for sid in visible_ids:
            c_raw = np.asarray(gt.centers_px[sid], dtype=np.float64).reshape(2)
            if gt_affine_T is not None:
                gt_eval_centers[sid] = _apply_affine_2d(gt_affine_T, c_raw)
            else:
                gt_eval_centers[sid] = c_raw
        base_errs: list[float] = []
        rec_errs: list[float] = []
        base_errs_raw: list[float] = []
        rec_errs_raw: list[float] = []
        base_matched = 0
        rec_matched = 0
        for sid in visible_ids:
            c_gt = gt_eval_centers[sid]
            c_gt_raw = gt.centers_px[sid]
            if sid in det_centers_px:
                base_matched += 1
                base_errs.append(float(np.linalg.norm(det_centers_px[sid] - c_gt)))
                base_errs_raw.append(float(np.linalg.norm(det_centers_px[sid] - c_gt_raw)))
            if sid in rec_centers_px:
                rec_matched += 1
                rec_errs.append(float(np.linalg.norm(rec_centers_px[sid] - c_gt)))
                rec_errs_raw.append(float(np.linalg.norm(rec_centers_px[sid] - c_gt_raw)))

        base_gt_mae = _safe_mean(base_errs)
        base_gt_rmse = _safe_rmse(base_errs)
        base_gt_recall = float(base_matched / max(1, len(visible_ids)))
        rec_gt_mae = _safe_mean(rec_errs)
        rec_gt_rmse = _safe_rmse(rec_errs)
        rec_gt_recall = float(rec_matched / max(1, len(visible_ids)))
        base_gt_mae_raw = _safe_mean(base_errs_raw)
        base_gt_rmse_raw = _safe_rmse(base_errs_raw)
        rec_gt_mae_raw = _safe_mean(rec_errs_raw)
        rec_gt_rmse_raw = _safe_rmse(rec_errs_raw)

        base_gt_mae_vals.append(base_gt_mae)
        base_gt_rmse_vals.append(base_gt_rmse)
        base_gt_recall_vals.append(base_gt_recall)
        base_det_recall_vals.append(float(info.get("det_recall_18", np.nan)))
        rec_gt_mae_vals.append(rec_gt_mae)
        rec_gt_rmse_vals.append(rec_gt_rmse)
        rec_gt_recall_vals.append(rec_gt_recall)
        base_gt_mae_raw_vals.append(base_gt_mae_raw)
        base_gt_rmse_raw_vals.append(base_gt_rmse_raw)
        rec_gt_mae_raw_vals.append(rec_gt_mae_raw)
        rec_gt_rmse_raw_vals.append(rec_gt_rmse_raw)

        recovered_ids = sorted([sid for sid in range(1, 19) if (sid not in det_centers_px) and (sid in rec_centers_px)])
        if recovered_ids and examples_saved < EXAMPLES_LIMIT:
            _draw_example(
                frame_rgb=frame_rgb,
                gt_centers=gt.centers_px,
                det_centers=det_centers_px,
                rec_centers=rec_centers_px,
                recovered_ids=recovered_ids,
                out_path=examples_dir / f"{frame_path.stem}_recovered.png",
            )
            examples_saved += 1
        if SAVE_ALL_OVERLAY_FRAMES:
            _draw_april_fabrik_overlay(
                frame_rgb=frame_rgb,
                det_centers=det_centers_px,
                rec_centers=rec_centers_px,
                recovered_ids=recovered_ids,
                out_path=overlay_dir / f"{frame_path.stem}_april_fabrik.png",
            )

        rows.append(
            {
                "frame": frame_path.name,
                "timestamp_s": float(gt.timestamp_s),
                "visible_ids_count": int(len(visible_ids)),
                "base_detected_count": int(len(det_centers_px)),
                "recovered_total_count": int(len(rec_centers_px)),
                "recovered_missing_count": int(len(recovered_ids)),
                "recovered_missing_ids": " ".join(str(x) for x in recovered_ids),
                "base_gt_mae_px": float(base_gt_mae),
                "base_gt_rmse_px": float(base_gt_rmse),
                "base_gt_recall": float(base_gt_recall),
                "base_det_recall_18_info": float(info.get("det_recall_18", np.nan)),
                "rec_gt_mae_px": float(rec_gt_mae),
                "rec_gt_rmse_px": float(rec_gt_rmse),
                "rec_gt_recall": float(rec_gt_recall),
                "rec_det_recall_18": float(rec_matched / 18.0),
                "base_gt_mae_px_raw": float(base_gt_mae_raw),
                "base_gt_rmse_px_raw": float(base_gt_rmse_raw),
                "rec_gt_mae_px_raw": float(rec_gt_mae_raw),
                "rec_gt_rmse_px_raw": float(rec_gt_rmse_raw),
                "mean_reprojection_error": float(info.get("mean_reprojection_error", np.nan)),
                "track_consistency_mae_px": float(info.get("track_consistency_mae_px", np.nan)),
                "track_consistency_rmse_px": float(info.get("track_consistency_rmse_px", np.nan)),
                "gate_reject_rate_on_detected": float(info.get("gate_reject_rate_on_detected", np.nan)),
                "track_alive_recall_18": float(info.get("track_alive_recall_18", np.nan)),
                "world_missing": int(bool(info.get("world_missing", True))),
            }
        )

        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(frame_paths)}] processed")

    csv_path = out_dir / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    frames_with_recovery = int(sum(1 for r in rows if int(r["recovered_missing_count"]) > 0))
    total_recovered_missing = int(sum(int(r["recovered_missing_count"]) for r in rows))
    improved_recall_frames = int(
        sum(1 for r in rows if float(r["rec_gt_recall"]) > float(r["base_gt_recall"]))
    )
    improved_mae_frames = int(
        sum(
            1
            for r in rows
            if np.isfinite(float(r["base_gt_mae_px"]))
            and np.isfinite(float(r["rec_gt_mae_px"]))
            and float(r["rec_gt_mae_px"]) < float(r["base_gt_mae_px"])
        )
    )

    summary = {
        "dataset_dir": str(DATASET_DIR.resolve()).replace("\\", "/"),
        "gt_path": str(GT_PATH.resolve()).replace("\\", "/"),
        "frames_total": len(rows),
        "frames_with_recovery": frames_with_recovery,
        "total_recovered_missing_ids": total_recovered_missing,
        "improved_recall_frames": improved_recall_frames,
        "improved_mae_frames": improved_mae_frames,
        "base_metrics_mean": {
            "det_recall_18": _safe_mean(base_det_recall_vals),
            "gt_mae_px": _safe_mean(base_gt_mae_vals),
            "gt_rmse_px": _safe_mean(base_gt_rmse_vals),
            "gt_recall": _safe_mean(base_gt_recall_vals),
            "track_consistency_mae_px": _safe_mean([float(r["track_consistency_mae_px"]) for r in rows]),
            "track_consistency_rmse_px": _safe_mean([float(r["track_consistency_rmse_px"]) for r in rows]),
            "gate_reject_rate_on_detected": _safe_mean([float(r["gate_reject_rate_on_detected"]) for r in rows]),
            "track_alive_recall_18": _safe_mean([float(r["track_alive_recall_18"]) for r in rows]),
            "mean_reprojection_error": _safe_mean([float(r["mean_reprojection_error"]) for r in rows]),
        },
        "recovered_metrics_mean": {
            "det_recall_18": _safe_mean([float(r["rec_det_recall_18"]) for r in rows]),
            "gt_mae_px": _safe_mean(rec_gt_mae_vals),
            "gt_rmse_px": _safe_mean(rec_gt_rmse_vals),
            "gt_recall": _safe_mean(rec_gt_recall_vals),
        },
        "raw_metrics_mean_without_gt_alignment": {
            "base_gt_mae_px": _safe_mean(base_gt_mae_raw_vals),
            "base_gt_rmse_px": _safe_mean(base_gt_rmse_raw_vals),
            "rec_gt_mae_px": _safe_mean(rec_gt_mae_raw_vals),
            "rec_gt_rmse_px": _safe_mean(rec_gt_rmse_raw_vals),
        },
        "gt_alignment": gt_alignment_info,
        "paths": {
            "per_frame_csv": str(csv_path.resolve()).replace("\\", "/"),
            "examples_dir": str(examples_dir.resolve()).replace("\\", "/"),
            "overlay_frames_dir": str(overlay_dir.resolve()).replace("\\", "/"),
        },
        "warnings": [gt_consistency_warning] if gt_consistency_warning is not None else [],
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== FABRIK Recovery Eval ===")
    print(f"output_dir: {out_dir}")
    print(f"frames_total: {summary['frames_total']}")
    print(f"frames_with_recovery: {summary['frames_with_recovery']}")
    print(f"total_recovered_missing_ids: {summary['total_recovered_missing_ids']}")
    print(f"base_det_recall_18_mean: {summary['base_metrics_mean']['det_recall_18']:.6f}")
    print(f"rec_det_recall_18_mean: {summary['recovered_metrics_mean']['det_recall_18']:.6f}")
    print(f"base_gt_mae_px_mean: {summary['base_metrics_mean']['gt_mae_px']:.6f}")
    print(f"rec_gt_mae_px_mean: {summary['recovered_metrics_mean']['gt_mae_px']:.6f}")
    print(f"base_gt_rmse_px_mean: {summary['base_metrics_mean']['gt_rmse_px']:.6f}")
    print(f"rec_gt_rmse_px_mean: {summary['recovered_metrics_mean']['gt_rmse_px']:.6f}")
    print(f"base_gt_recall_mean: {summary['base_metrics_mean']['gt_recall']:.6f}")
    print(f"rec_gt_recall_mean: {summary['recovered_metrics_mean']['gt_recall']:.6f}")
    print(f"summary_json: {summary_path}")
    print(f"per_frame_csv: {csv_path}")
    print(f"overlay_frames_dir: {overlay_dir}")
    print(
        "gt_alignment:",
        f"enabled={gt_alignment_info.get('enabled', False)}",
        f"pairs={gt_alignment_info.get('pairs', 0)}",
        f"fit_mae_px={gt_alignment_info.get('fit_mae_px', float('nan'))}",
    )
    if gt_consistency_warning is not None:
        print("WARNING:", gt_consistency_warning)


if __name__ == "__main__":
    main()
