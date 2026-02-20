from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np


@dataclass
class ArucoConfig:
    # OpenCV dictionary name. DICT_4X4_50 covers IDs 0..49 and works for 0..18.
    dict_name: str = "DICT_4X4_50"
    # Physical marker side length in meters, used for pose estimation.
    marker_length_m: float = 0.007
    # Optional per-ID side length override in meters.
    marker_length_per_id: dict[int, float] = field(default_factory=dict)
    # Camera intrinsics for pose3d mode.
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    # Output mode. pose3d returns xyz+quat+valid. image2d returns normalized center+valid.
    output_mode: Literal["pose3d", "image2d"] = "image2d"
    # Expected marker IDs. 0 is world marker, 1..18 are segment markers.
    expected_ids: list[int] = field(default_factory=lambda: list(range(19)))
    # Detector threshold settings. Wider range improves robustness to highlights.
    adaptiveThreshWinSizeMin: int = 3
    adaptiveThreshWinSizeMax: int = 121
    adaptiveThreshWinSizeStep: int = 6
    # Corner refinement settings. SUBPIX usually gives smoother center/pose estimates.
    cornerRefinementMethod: int = 1
    cornerRefinementWinSize: int = 11
    # Allowed marker perimeter range, normalized by image size.
    # Higher minimum rejects tiny unstable markers and most false positives.
    minMarkerPerimeterRate: float = 0.01
    maxMarkerPerimeterRate: float = 4.0
    # Error correction for code decoding. Higher catches harder cases but can increase false IDs.
    errorCorrectionRate: float = 0.82
    # Additional geometric filtering to suppress false detections.
    min_side_px: float = 7.0
    min_square_ratio: float = 0.60
    # Hard filter by quality score to reject weak decoded markers.
    min_detection_score: float = 0.22
    # Optional refine pass after detection.
    useRefineDetectedMarkers: bool = False
    # Reject unstable pose solutions when in pose3d mode.
    max_reprojection_error_px: float = 3.0
    # Tracking controls.
    max_lost_frames: int = 12
    smoothing_alpha: float = 0.6
    enable_kalman: bool = True
    # Gating rejects sudden center jumps that are likely ID swaps/outliers.
    enable_gating: bool = True
    gating_max_jump_px: float = 26.0
    gating_lost_relax_per_frame: float = 0.35
    # Reject updates with too-low quality compared to recent valid score.
    enable_quality_gating: bool = True
    quality_min_score_ratio: float = 0.70
    # Input color order of frame.
    input_color: Literal["BGR", "RGB"] = "RGB"
    # Pack per-segment tracking age features into RL state.
    include_tracking_features: bool = True
    # Track is considered alive for control if lost_frames <= alive_k_frames.
    alive_k_frames: int = 3
    # In pose3d mode, append [age, k] to each segment state.
    include_age_k: bool = True
    # age normalization strategy: "frames_norm" -> age in [0,1], k=1.
    age_norm_mode: str = "frames_norm"
    # In pose3d mode, require world marker (id=0) to emit valid segment poses.
    require_world_for_pose3d: bool = True


class ArucoDetector:
    """ArUco detection and optional pose estimation."""

    def __init__(self, config: ArucoConfig) -> None:
        self.config = config
        self._cv2 = self._import_cv2()
        self._aruco = self._get_aruco_module(self._cv2)
        self._dict = self._get_dictionary()
        self._params = self._build_detector_params()
        self._detector = self._build_detector_if_supported()

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenCV is required. Install opencv-contrib-python.") from exc
        return cv2

    @staticmethod
    def _get_aruco_module(cv2_mod):
        aruco = getattr(cv2_mod, "aruco", None)
        if aruco is None:
            raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python.")
        return aruco

    def _get_dictionary(self):
        if not hasattr(self._aruco, self.config.dict_name):
            raise ValueError(f"Unknown ArUco dictionary: {self.config.dict_name}")
        dict_id = getattr(self._aruco, self.config.dict_name)
        if hasattr(self._aruco, "getPredefinedDictionary"):
            return self._aruco.getPredefinedDictionary(dict_id)
        return self._aruco.Dictionary_get(dict_id)

    def _build_detector_params(self):
        if hasattr(self._aruco, "DetectorParameters"):
            params = self._aruco.DetectorParameters()
        else:
            params = self._aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = int(self.config.adaptiveThreshWinSizeMin)
        params.adaptiveThreshWinSizeMax = int(self.config.adaptiveThreshWinSizeMax)
        params.adaptiveThreshWinSizeStep = int(self.config.adaptiveThreshWinSizeStep)
        params.cornerRefinementMethod = int(self.config.cornerRefinementMethod)
        params.cornerRefinementWinSize = int(self.config.cornerRefinementWinSize)
        params.minMarkerPerimeterRate = float(self.config.minMarkerPerimeterRate)
        params.maxMarkerPerimeterRate = float(self.config.maxMarkerPerimeterRate)
        # Useful on synthetic renders where contrast polarity can flip due to lighting.
        if hasattr(params, "detectInvertedMarker"):
            params.detectInvertedMarker = True
        if hasattr(params, "errorCorrectionRate"):
            params.errorCorrectionRate = float(self.config.errorCorrectionRate)
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 2
        if hasattr(params, "minMarkerDistanceRate"):
            params.minMarkerDistanceRate = 0.01
        if hasattr(params, "adaptiveThreshConstant"):
            params.adaptiveThreshConstant = 7.0
        if hasattr(params, "polygonalApproxAccuracyRate"):
            params.polygonalApproxAccuracyRate = 0.03
        if hasattr(params, "minCornerDistanceRate"):
            params.minCornerDistanceRate = 0.02
        if hasattr(params, "cornerRefinementMaxIterations"):
            params.cornerRefinementMaxIterations = 50
        if hasattr(params, "cornerRefinementMinAccuracy"):
            params.cornerRefinementMinAccuracy = 0.01
        return params

    def _build_detector_if_supported(self):
        if hasattr(self._aruco, "ArucoDetector"):
            return self._aruco.ArucoDetector(self._dict, self._params)
        return None

    def _to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert renderer output to uint8 in range [0, 255].
        This improves threshold-based ArUco detection stability.
        """
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

    def _run_detect(self, gray: np.ndarray):
        if self._detector is not None:
            corners, ids, rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, rejected = self._aruco.detectMarkers(gray, self._dict, parameters=self._params)
        return corners, ids, rejected

    def _run_refine(
        self,
        gray: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        rejected: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        if not self.config.useRefineDetectedMarkers:
            return corners, ids, rejected
        try:
            if hasattr(self._aruco, "refineDetectedMarkers"):
                refined = self._aruco.refineDetectedMarkers(
                    image=gray,
                    board=None,
                    detectedCorners=corners,
                    detectedIds=ids,
                    rejectedCorners=rejected,
                    cameraMatrix=self.config.camera_matrix,
                    distCoeffs=self.config.dist_coeffs,
                    parameters=self._params,
                )
                if isinstance(refined, tuple) and len(refined) >= 3:
                    corners = refined[0]
                    ids = refined[1]
                    rejected = refined[2]
        except Exception:
            pass
        return corners, ids, rejected

    @staticmethod
    def _perimeter(c: np.ndarray) -> float:
        pts = c.reshape(4, 2).astype(np.float32)
        p = 0.0
        for i in range(4):
            p += float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        return p

    def _geometry_ok(self, c: np.ndarray) -> bool:
        pts = c.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        if min(sides) < float(self.config.min_side_px):
            return False
        long_side = max(sides)
        short_side = max(1e-6, min(sides))
        ratio = short_side / long_side
        if ratio < float(self.config.min_square_ratio):
            return False
        return True

    def _patch_stats(self, gray: np.ndarray, corners_px: np.ndarray) -> tuple[float, float]:
        x0 = int(max(0, np.floor(np.min(corners_px[:, 0]))))
        x1 = int(min(gray.shape[1] - 1, np.ceil(np.max(corners_px[:, 0]))))
        y0 = int(max(0, np.floor(np.min(corners_px[:, 1]))))
        y1 = int(min(gray.shape[0] - 1, np.ceil(np.max(corners_px[:, 1]))))
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
        if roi.size < 9:
            return 0.0, 0.0
        contrast = float(np.std(roi))
        lap = self._cv2.Laplacian(roi, self._cv2.CV_32F)
        sharpness = float(np.var(lap))
        return contrast, sharpness

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
        """
        Detect on full resolution frame.
        Returns dict[id] with corners_px, center_px, optional rvec/tvec, and quality values.
        """
        gray = self._to_gray(frame)
        gray = self._cv2.GaussianBlur(gray, (3, 3), 0)
        variants: list[tuple[np.ndarray, float]] = [(gray, 1.0)]
        # Keep only one additional upscaled pass to improve tiny-marker recall
        # while limiting false positives from aggressive preprocessing variants.
        gray_up = self._cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=self._cv2.INTER_CUBIC)
        variants.append((gray_up, 2.0))

        merged: dict[int, dict[str, Any]] = {}
        for gray_i, scale_i in variants:
            corners, ids, rejected = self._run_detect(gray_i)
            corners, ids, _ = self._run_refine(gray_i, corners, ids, rejected)
            if ids is None or len(ids) == 0:
                continue
            for i, marker_id in enumerate(ids.reshape(-1).astype(int)):
                c = corners[i].reshape(4, 2).astype(np.float32)
                if scale_i != 1.0:
                    c = c / float(scale_i)
                if not self._geometry_ok(c):
                    continue
                center = np.mean(c, axis=0).astype(np.float32)
                peri = self._perimeter(c)
                # Score must be computed in the same coordinate system as corners (base gray).
                contrast, sharpness = self._patch_stats(gray, c)
                refine_bonus = 0.15 if self.config.cornerRefinementMethod != 0 else 0.0
                score = 0.0
                score += np.clip(peri / 400.0, 0.0, 0.5)
                score += np.clip(contrast / 80.0, 0.0, 0.25)
                score += np.clip(sharpness / 1200.0, 0.0, 0.25)
                score = float(np.clip(score + refine_bonus, 0.0, 1.0))
                if score < float(self.config.min_detection_score):
                    continue
                rvec, tvec, reproj = self._estimate_pose(c, marker_id)
                if reproj is not None and float(reproj) > float(self.config.max_reprojection_error_px):
                    continue
                candidate = {
                    "corners_px": c,
                    "center_px": center,
                    "rvec": rvec,
                    "tvec": tvec,
                    "reprojection_error": reproj,
                    "detection_score": score,
                }
                old = merged.get(int(marker_id))
                if old is None or float(candidate["detection_score"]) > float(old["detection_score"]):
                    merged[int(marker_id)] = candidate
        return merged


@dataclass
class _TrackState:
    valid: bool = False
    lost_frames: int = 0
    center_px: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    corners_px: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    last_timestamp: float | None = None
    kf_x: np.ndarray | None = None
    kf_P: np.ndarray | None = None
    gate_rejected: bool = False
    last_score: float = 0.0


class ArucoTracker:
    """Temporal filter that keeps fixed marker slots and handles losses."""

    def __init__(self, config: ArucoConfig) -> None:
        self.config = config
        self._tracks: dict[int, _TrackState] = {mid: _TrackState() for mid in config.expected_ids}

    @staticmethod
    def _ema(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        return (alpha * new + (1.0 - alpha) * prev).astype(np.float32)

    def _kalman_predict(self, st: _TrackState, dt: float) -> None:
        if st.kf_x is None or st.kf_P is None:
            return
        A = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        Q = np.diag([2e-2, 2e-2, 1e-1, 1e-1]).astype(np.float32)
        st.kf_x = A @ st.kf_x
        st.kf_P = A @ st.kf_P @ A.T + Q

    def _kalman_update(self, st: _TrackState, z: np.ndarray) -> np.ndarray:
        if st.kf_x is None or st.kf_P is None:
            st.kf_x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
            st.kf_P = np.eye(4, dtype=np.float32) * 0.5
            return z.astype(np.float32)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        R = np.diag([1.5, 1.5]).astype(np.float32)
        y = z - H @ st.kf_x
        S = H @ st.kf_P @ H.T + R
        K = st.kf_P @ H.T @ np.linalg.inv(S)
        st.kf_x = st.kf_x + K @ y
        st.kf_P = (np.eye(4, dtype=np.float32) - K @ H) @ st.kf_P
        return st.kf_x[:2].astype(np.float32)

    def _gating_ok(self, st: _TrackState, c_new: np.ndarray, score_new: float) -> bool:
        if not self.config.enable_gating:
            return True
        if not st.valid:
            return True
        if self.config.enable_kalman and st.kf_x is not None:
            pred = st.kf_x[:2].astype(np.float32)
        else:
            pred = st.center_px.astype(np.float32)
        jump = float(np.linalg.norm(c_new - pred))
        relax = 1.0 + float(max(0, st.lost_frames)) * float(self.config.gating_lost_relax_per_frame)
        max_jump = float(self.config.gating_max_jump_px) * relax
        if jump > max_jump:
            return False
        if self.config.enable_quality_gating and st.last_score > 1e-6:
            min_allowed = st.last_score * float(self.config.quality_min_score_ratio)
            if score_new < min_allowed:
                return False
        return True

    def update(self, detections: dict[int, dict[str, Any]], timestamp: float | int) -> dict[int, dict[str, Any]]:
        t = float(timestamp)
        alpha = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
        max_lost = int(max(0, self.config.max_lost_frames))
        out: dict[int, dict[str, Any]] = {}
        for marker_id in self.config.expected_ids:
            st = self._tracks[marker_id]
            dt = 1.0 / 30.0 if st.last_timestamp is None else max(1e-3, t - st.last_timestamp)
            st.last_timestamp = t
            if self.config.enable_kalman:
                self._kalman_predict(st, dt)
            det = detections.get(marker_id)
            st.gate_rejected = False
            if det is not None:
                c_new = np.asarray(det["center_px"], dtype=np.float32).reshape(2)
                score_new = float(det.get("detection_score", 0.0))
                if not self._gating_ok(st, c_new, score_new):
                    st.gate_rejected = True
                    det = None
            if det is not None:
                c_new = np.asarray(det["center_px"], dtype=np.float32).reshape(2)
                q_new = np.asarray(det["corners_px"], dtype=np.float32).reshape(4, 2)
                c_filt = self._kalman_update(st, c_new) if self.config.enable_kalman else c_new
                if st.valid:
                    st.center_px = self._ema(st.center_px, c_filt, alpha)
                    st.corners_px = self._ema(st.corners_px, q_new, alpha)
                else:
                    st.center_px = c_filt.astype(np.float32)
                    st.corners_px = q_new.astype(np.float32)
                rvec = det.get("rvec")
                tvec = det.get("tvec")
                if rvec is not None and tvec is not None:
                    rvec_n = np.asarray(rvec, dtype=np.float32).reshape(3)
                    tvec_n = np.asarray(tvec, dtype=np.float32).reshape(3)
                    st.rvec = rvec_n if st.rvec is None else self._ema(st.rvec, rvec_n, alpha)
                    st.tvec = tvec_n if st.tvec is None else self._ema(st.tvec, tvec_n, alpha)
                st.valid = True
                st.lost_frames = 0
                st.last_score = float(det.get("detection_score", 0.0))
            else:
                st.lost_frames += 1
                if self.config.enable_kalman and st.kf_x is not None and st.valid:
                    pred_center = st.kf_x[:2].astype(np.float32)
                    delta = pred_center - np.mean(st.corners_px, axis=0)
                    st.center_px = pred_center
                    st.corners_px = (st.corners_px + delta[None, :]).astype(np.float32)
                # Keep state for continuity, but mark invalid when no fresh detection in this frame.
                st.valid = False
                if st.lost_frames > max_lost:
                    st.rvec = None
                    st.tvec = None
                    st.last_score = 0.0
            out[marker_id] = {
                "valid": bool(st.valid),
                "lost_frames": int(st.lost_frames),
                "center_px": st.center_px.astype(np.float32, copy=True),
                "corners_px": st.corners_px.astype(np.float32, copy=True),
                "rvec": None if st.rvec is None else st.rvec.astype(np.float32, copy=True),
                "tvec": None if st.tvec is None else st.tvec.astype(np.float32, copy=True),
                "gate_rejected": bool(st.gate_rejected),
                "detection_score": float(st.last_score),
            }
        return out


def pose_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for pose conversions.") from exc
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def relative_pose(T_cam_marker: np.ndarray, T_cam_world: np.ndarray) -> np.ndarray:
    return invert_pose(T_cam_world) @ T_cam_marker


def _matrix_to_quat_wxyz(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3].astype(np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return q


def pack_state(
    tracks: dict[int, dict[str, Any]],
    world_pose: np.ndarray | None,
    mode: str,
    frame_shape: tuple[int, ...],
    config: ArucoConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    H, W = int(frame_shape[0]), int(frame_shape[1])
    info: dict[str, Any] = {}
    if mode == "pose3d":
        world_valid = world_pose is not None
        T_cam_world = world_pose
        parts: list[np.ndarray] = []
        valid_mask: list[int] = []
        alive_mask: list[int] = []
        alive_k = max(1, int(config.alive_k_frames))
        base = 8
        extra = 2 if bool(config.include_age_k) else 0
        for marker_id in range(1, 19):
            tr = tracks.get(marker_id, {})
            valid = bool(tr.get("valid", False)) and tr.get("rvec") is not None and tr.get("tvec") is not None
            has_signal = bool(np.max(np.abs(np.asarray(tr.get("corners_px", np.zeros((4, 2))), dtype=np.float32))) > 1e-6)
            lost = float(max(0, tr.get("lost_frames", 0)))
            alive = int((lost <= alive_k) and has_signal)
            can_emit = valid and (world_valid or not bool(config.require_world_for_pose3d))
            v = np.zeros(base + extra, dtype=np.float32)
            if can_emit:
                T_cam_marker = pose_to_matrix(np.asarray(tr["rvec"]), np.asarray(tr["tvec"]))
                T_ref = relative_pose(T_cam_marker, T_cam_world) if T_cam_world is not None else T_cam_marker
                v[0:3] = T_ref[:3, 3]
                v[3:7] = _matrix_to_quat_wxyz(T_ref)
                v[7] = 1.0
                valid_mask.append(1)
            else:
                valid_mask.append(0)
            if bool(config.include_age_k):
                k_frames = float(max(1, int(config.alive_k_frames)))
                if str(config.age_norm_mode) == "frames_norm":
                    age = min(lost / max(1.0, k_frames), 1.0)
                    k_val = 1.0
                else:
                    age = lost
                    k_val = k_frames
                v[8] = float(age)
                v[9] = float(k_val)
            alive_mask.append(alive)
            parts.append(v)
        info["world_missing"] = bool(not world_valid)
        info["valid_mask"] = np.asarray(valid_mask, dtype=np.uint8)
        info["alive_mask"] = np.asarray(alive_mask, dtype=np.uint8)
        info["track_feature_dim"] = extra
        return np.concatenate(parts, axis=0).astype(np.float32), info
    if mode == "image2d":
        world = tracks.get(0, {})
        world_valid = bool(world.get("valid", False))
        parts = []
        valid_mask = []
        alive_mask = []
        lost_max = max(1, int(config.max_lost_frames))
        alive_k = max(1, int(config.alive_k_frames))
        track_feat_dim = 2 if bool(config.include_tracking_features) else 0
        for marker_id in range(1, 19):
            tr = tracks.get(marker_id, {})
            v = np.zeros(3 + track_feat_dim, dtype=np.float32)
            valid = bool(tr.get("valid", False))
            c = np.asarray(tr.get("center_px", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
            corners = np.asarray(tr.get("corners_px", np.zeros((4, 2))), dtype=np.float32)
            has_signal = bool(np.max(np.abs(corners)) > 1e-6)
            lost = int(max(0, tr.get("lost_frames", lost_max)))
            alive = int((lost <= alive_k) and has_signal)

            if valid or has_signal:
                v[0] = float(np.clip(c[0] / max(1.0, float(W)), 0.0, 1.0))
                v[1] = float(np.clip(c[1] / max(1.0, float(H)), 0.0, 1.0))
                v[2] = 1.0 if valid else 0.0
            if bool(config.include_tracking_features):
                v[3] = float(np.clip(lost / float(lost_max), 0.0, 1.0))
                v[4] = float(alive)
            valid_mask.append(1 if valid else 0)
            alive_mask.append(alive)
            parts.append(v)
        info["world_missing"] = bool(not world_valid)
        info["valid_mask"] = np.asarray(valid_mask, dtype=np.uint8)
        info["alive_mask"] = np.asarray(alive_mask, dtype=np.uint8)
        info["track_feature_dim"] = track_feat_dim
        return np.concatenate(parts, axis=0).astype(np.float32), info
    raise ValueError(f"Unsupported mode: {mode}")


class ArucoPipeline:
    """Main RL-facing API that hides OpenCV details."""

    def __init__(self, config: ArucoConfig) -> None:
        self.config = config
        self.detector = ArucoDetector(config)
        self.tracker = ArucoTracker(config)
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
        if self.config.output_mode == "pose3d":
            reproj_errs = [
                float(d["reprojection_error"]) for d in detections.values() if d.get("reprojection_error") is not None
            ]
            mean_reproj = float(np.mean(reproj_errs)) if reproj_errs else float("nan")
        else:
            mean_reproj = float("nan")

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
            if tr is None:
                continue
            if bool(tr.get("gate_rejected", False)):
                gate_rejected_mask[i] = 1
        det_count = int(np.sum(detected_seg_mask))
        gate_reject_rate_on_detected = float(np.sum(gate_rejected_mask) / max(1, det_count))

        # Alive means not too stale for control, not internal max_lost bookkeeping.
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


def render_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Offscreen RGB render helper for MuJoCo.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {camera_name}")
    renderer = mujoco.Renderer(model, width=int(width), height=int(height))
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
    finally:
        renderer.close()
    if rgb.dtype != np.uint8:
        arr = rgb.astype(np.float32)
        if float(np.max(arr)) <= 1.0:
            arr *= 255.0
        rgb = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return rgb


def make_camera_matrix_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """
    Build pinhole intrinsics from vertical FOV and render resolution.
    dist_coeffs for MuJoCo can usually be zeros.
    For real cameras use calibrated intrinsics/distortion instead of this helper.
    """
    h = float(height)
    w = float(width)
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * h / np.tan(0.5 * fovy)
    fx = fy
    cx = 0.5 * w
    cy = 0.5 * h
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K
