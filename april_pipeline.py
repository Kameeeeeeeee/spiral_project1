from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from aruco_pipeline import ArucoTracker, make_camera_matrix_from_fovy, pack_state, pose_to_matrix


@dataclass
class AprilConfig:
    # AprilTag family name, for example tag36h11, tag25h9, tag16h5.
    apriltag_family: str = "tag36h11"
    # Physical marker side length in meters, used for pose estimation.
    marker_length_m: float = 0.007
    # Optional per-ID side length override in meters.
    marker_length_per_id: dict[int, float] = field(default_factory=dict)
    # Camera intrinsics for pose3d mode.
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    # Output mode. pose3d returns xyz+quat+valid.
    output_mode: Literal["pose3d", "image2d"] = "image2d"
    # Expected marker IDs. 0 is world marker, 1..18 are segment markers.
    expected_ids: list[int] = field(default_factory=lambda: list(range(19)))

    # AprilTag detector parameters.
    nthreads: int = 2
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25

    # Geometric and quality filtering.
    min_side_px: float = 7.0
    min_square_ratio: float = 0.60
    min_detection_score: float = 0.22
    min_decision_margin: float = 8.0
    decision_margin_ref: float = 40.0
    max_reprojection_error_px: float = 3.0

    # Tracking controls.
    max_lost_frames: int = 12
    smoothing_alpha: float = 0.6
    enable_kalman: bool = True
    enable_gating: bool = True
    gating_max_jump_px: float = 26.0
    gating_lost_relax_per_frame: float = 0.35
    enable_quality_gating: bool = True
    quality_min_score_ratio: float = 0.70

    # Input color order of frame.
    input_color: Literal["BGR", "RGB"] = "RGB"

    # RL state controls.
    include_tracking_features: bool = True
    alive_k_frames: int = 3
    include_age_k: bool = True
    age_norm_mode: str = "frames_norm"
    require_world_for_pose3d: bool = True


class AprilTagDetector:
    """AprilTag detection and optional pose estimation using OpenCV solvePnP."""

    def __init__(self, config: AprilConfig) -> None:
        self.config = config
        self._cv2 = self._import_cv2()
        self._detector = self._import_detector()

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenCV is required. Install opencv-contrib-python.") from exc
        return cv2

    def _import_detector(self):
        try:
            from pupil_apriltags import Detector  # type: ignore
        except Exception as exc:
            raise RuntimeError("pupil_apriltags is required. Install with: pip install pupil-apriltags") from exc
        return Detector(
            families=str(self.config.apriltag_family),
            nthreads=int(self.config.nthreads),
            quad_decimate=float(self.config.quad_decimate),
            quad_sigma=float(self.config.quad_sigma),
            refine_edges=1 if bool(self.config.refine_edges) else 0,
            decode_sharpening=float(self.config.decode_sharpening),
        )

    def _to_uint8(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        arr = frame.astype(np.float32, copy=False)
        vmax = float(np.max(arr)) if arr.size > 0 else 0.0
        if vmax <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        frame_u8 = self._to_uint8(frame)
        if frame_u8.ndim == 2:
            return frame_u8
        if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
            raise ValueError(f"Unsupported frame shape: {frame_u8.shape}")
        if self.config.input_color.upper() == "RGB":
            return self._cv2.cvtColor(frame_u8, self._cv2.COLOR_RGB2GRAY)
        return self._cv2.cvtColor(frame_u8, self._cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _order_corners_tl_tr_br_bl(corners_px: np.ndarray) -> np.ndarray:
        c = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        s = c.sum(axis=1)
        d = np.diff(c, axis=1).reshape(4)
        tl = c[np.argmin(s)]
        br = c[np.argmax(s)]
        tr = c[np.argmin(d)]
        bl = c[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    def _geometry_ok(self, c: np.ndarray) -> bool:
        pts = c.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        if min(sides) < float(self.config.min_side_px):
            return False
        long_side = max(sides)
        short_side = max(1e-6, min(sides))
        return (short_side / long_side) >= float(self.config.min_square_ratio)

    def _estimate_pose(
        self, corners_px: np.ndarray, marker_id: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        cfg = self.config
        if cfg.output_mode != "pose3d":
            return None, None, None
        if cfg.camera_matrix is None or cfg.dist_coeffs is None:
            return None, None, None
        marker_len = float(cfg.marker_length_per_id.get(marker_id, cfg.marker_length_m))
        obj_pts = np.array(
            [
                [-0.5 * marker_len, 0.5 * marker_len, 0.0],
                [0.5 * marker_len, 0.5 * marker_len, 0.0],
                [0.5 * marker_len, -0.5 * marker_len, 0.0],
                [-0.5 * marker_len, -0.5 * marker_len, 0.0],
            ],
            dtype=np.float32,
        )
        ok, rvec, tvec = self._cv2.solvePnP(
            obj_pts,
            corners_px.astype(np.float32),
            cfg.camera_matrix.astype(np.float32),
            cfg.dist_coeffs.astype(np.float32),
            flags=self._cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None, None, None
        proj, _ = self._cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            cfg.camera_matrix.astype(np.float32),
            cfg.dist_coeffs.astype(np.float32),
        )
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners_px, axis=1)))
        return rvec.reshape(3).astype(np.float32), tvec.reshape(3).astype(np.float32), err

    def detect(self, frame: np.ndarray) -> dict[int, dict[str, Any]]:
        gray = self._to_gray(frame)
        tags = self._detector.detect(gray, estimate_tag_pose=False)

        merged: dict[int, dict[str, Any]] = {}
        for tag in tags:
            marker_id = int(getattr(tag, "tag_id", -1))
            if marker_id < 0:
                continue
            margin = float(getattr(tag, "decision_margin", 0.0))
            if margin < float(self.config.min_decision_margin):
                continue
            corners = self._order_corners_tl_tr_br_bl(np.asarray(tag.corners, dtype=np.float32))
            if not self._geometry_ok(corners):
                continue
            center = np.mean(corners, axis=0).astype(np.float32)
            score = float(np.clip(margin / max(1e-6, float(self.config.decision_margin_ref)), 0.0, 1.0))
            if score < float(self.config.min_detection_score):
                continue
            rvec, tvec, reproj = self._estimate_pose(corners, marker_id)
            if reproj is not None and float(reproj) > float(self.config.max_reprojection_error_px):
                continue
            candidate = {
                "corners_px": corners,
                "center_px": center,
                "rvec": rvec,
                "tvec": tvec,
                "reprojection_error": reproj,
                "detection_score": score,
            }
            old = merged.get(marker_id)
            if old is None or float(candidate["detection_score"]) > float(old["detection_score"]):
                merged[marker_id] = candidate
        return merged


class AprilPipeline:
    """Main RL-facing API matching ArucoPipeline contract."""

    def __init__(self, config: AprilConfig) -> None:
        self.config = config
        self.detector = AprilTagDetector(config)
        self.tracker = ArucoTracker(config)  # API-compatible tracker/state layout reuse.
        if config.output_mode == "pose3d":
            base = 8
            extra = 2 if config.include_age_k else 0
            per_seg = base + extra
        else:
            raise ValueError("Use pose3d only for sim-to-real")
        self.state_dim = 18 * per_seg
        self._step_count = 0

    def step(
        self,
        frame: np.ndarray,
        timestamp: float | int,
        gt_centers_px: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count += 1
        detections_raw = self.detector.detect(frame)
        expected = set(int(x) for x in self.config.expected_ids)
        detections = {mid: det for mid, det in detections_raw.items() if mid in expected}
        unexpected_ids = sorted(int(mid) for mid in detections_raw.keys() if mid not in expected)
        tracks = self.tracker.update(detections, timestamp=timestamp)
        world_track = tracks.get(0, {})
        world_valid = (
            bool(world_track.get("valid", False))
            and world_track.get("rvec") is not None
            and world_track.get("tvec") is not None
        )
        world_pose = (
            pose_to_matrix(np.asarray(world_track["rvec"]), np.asarray(world_track["tvec"])) if world_valid else None
        )
        state_vec, pack_info = pack_state(
            tracks,
            world_pose=world_pose,
            mode=self.config.output_mode,
            frame_shape=frame.shape,
            config=self.config,
        )
        detected_ids = sorted(int(mid) for mid in detections.keys())
        seg_expected = [int(mid) for mid in self.config.expected_ids if int(mid) != 0]
        missing_ids = [mid for mid in seg_expected if mid not in detected_ids]
        reproj_errs = [float(d["reprojection_error"]) for d in detections.values() if d.get("reprojection_error") is not None]
        mean_reproj = float(np.mean(reproj_errs)) if reproj_errs else float("nan")

        seg_ids = list(range(1, 19))
        detected_seg_mask = np.zeros((18,), dtype=np.uint8)
        for i, mid in enumerate(seg_ids):
            if mid in detections:
                detected_seg_mask[i] = 1
        det_recall_18 = float(np.mean(detected_seg_mask))

        matched_ids = [mid for mid in seg_ids if mid in detections]
        track_consistency_mae_px = float("nan")
        track_consistency_rmse_px = float("nan")
        track_num = 0
        if matched_ids:
            abs_errs = []
            sq_errs = []
            for mid in matched_ids:
                c_det = np.asarray(detections[mid]["center_px"], dtype=np.float32).reshape(2)
                c_trk = np.asarray(tracks[mid]["center_px"], dtype=np.float32).reshape(2)
                d = c_det - c_trk
                abs_errs.append(np.abs(d))
                sq_errs.append(float(np.sum(d * d)))
                track_num += 1
            track_consistency_mae_px = float(np.mean(np.concatenate(abs_errs))) if abs_errs else float("nan")
            track_consistency_rmse_px = float(np.sqrt(np.mean(sq_errs))) if sq_errs else float("nan")

        gate_rejected_mask = np.zeros((18,), dtype=np.uint8)
        for i, mid in enumerate(seg_ids):
            tr = tracks.get(mid)
            if tr is not None and bool(tr.get("gate_rejected", False)):
                gate_rejected_mask[i] = 1
        det_count = int(np.sum(detected_seg_mask))
        gate_reject_rate_on_detected = float(np.sum(gate_rejected_mask) / max(1, det_count))

        alive_k = max(1, int(self.config.alive_k_frames))
        track_alive_mask = np.zeros((18,), dtype=np.uint8)
        for i, mid in enumerate(seg_ids):
            tr = tracks.get(mid)
            if tr is None:
                continue
            lost = int(tr.get("lost_frames", alive_k + 999))
            corners = np.asarray(tr.get("corners_px", np.zeros((4, 2))), dtype=np.float32)
            has_signal = bool(np.max(np.abs(corners)) > 1e-6)
            if lost <= alive_k and has_signal:
                track_alive_mask[i] = 1
        track_alive_recall_18 = float(np.mean(track_alive_mask))

        gt_mae_px = float("nan")
        gt_rmse_px = float("nan")
        gt_per_id_l2 = np.full((18,), np.nan, dtype=np.float32)
        gt_num_visible = 0
        gt_num_matched = 0
        if gt_centers_px is not None:
            for i, mid in enumerate(seg_ids):
                if mid not in gt_centers_px:
                    continue
                gt_num_visible += 1
                if mid in detections:
                    c_det = np.asarray(detections[mid]["center_px"], dtype=np.float32).reshape(2)
                    c_gt = np.asarray(gt_centers_px[mid], dtype=np.float32).reshape(2)
                    d = c_det - c_gt
                    gt_per_id_l2[i] = float(np.sqrt(np.sum(d * d)))
                    gt_num_matched += 1
            valid = gt_per_id_l2[~np.isnan(gt_per_id_l2)]
            if valid.size > 0:
                gt_mae_px = float(np.mean(valid))
                gt_rmse_px = float(np.sqrt(np.mean(valid * valid)))

        gt_recall = float(gt_num_matched / max(1, gt_num_visible))
        info: dict[str, Any] = {
            "detected_ids": detected_ids,
            "missing_ids": missing_ids,
            "valid_mask": pack_info.get("valid_mask", np.zeros(18, dtype=np.uint8)),
            "detected_seg_mask": detected_seg_mask,
            "det_recall_18": det_recall_18,
            "mean_reprojection_error": mean_reproj,
            "track_consistency_mae_px": track_consistency_mae_px,
            "track_consistency_rmse_px": track_consistency_rmse_px,
            "num_track_consistency_ids": track_num,
            "gate_rejected_mask": gate_rejected_mask,
            "gate_reject_rate_on_detected": gate_reject_rate_on_detected,
            "track_alive_mask": track_alive_mask,
            "track_alive_recall_18": track_alive_recall_18,
            "alive_k_frames": int(alive_k),
            "gt_mae_px": gt_mae_px,
            "gt_rmse_px": gt_rmse_px,
            "gt_per_id_l2_px": gt_per_id_l2,
            "gt_num_visible": gt_num_visible,
            "gt_num_matched": gt_num_matched,
            "gt_recall": gt_recall,
            "num_gt_error_ids": gt_num_matched,
            "world_missing": bool(pack_info.get("world_missing", True)),
            "num_detected": len(detected_ids),
            "num_missing": len(missing_ids),
            "unexpected_ids": unexpected_ids,
            "step_count": self._step_count,
            "tracks": tracks,
        }
        return state_vec.astype(np.float32), info
