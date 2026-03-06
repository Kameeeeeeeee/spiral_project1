from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from april_pipeline import AprilConfig, AprilTagDetector


DEFAULT_TAG_IMAGE = Path("./assets/AprilTag36h11/tag36h11_id_00.png")
DEFAULT_GRAY_INPUT_DIR = Path("./debug/input_FOR_APRILTAG")
DEFAULT_DATASET_DIR = Path("./assets/dataset/apriltag")
OUT_ROOT = Path("./debug")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required. Install opencv-contrib-python.") from exc
    return cv2


def _read_gray(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _read_rgb(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_gray(path: Path, gray: np.ndarray) -> None:
    cv2 = _import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.asarray(gray, dtype=np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")
    path.write_bytes(encoded.tobytes())


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ids_to_str(ids: list[int]) -> str:
    return " ".join(str(x) for x in sorted(ids))


def _base_cfg() -> AprilConfig:
    return AprilConfig(
        apriltag_family="tag36h11",
        output_mode="image2d",
        expected_ids=list(range(19)),
        input_color="RGB",
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        min_decision_margin=8.0,
        min_detection_score=0.22,
        max_reprojection_error_px=3.0,
        min_side_px=7.0,
        min_square_ratio=0.60,
    )


def _relaxed_cfg(cfg: AprilConfig) -> AprilConfig:
    return replace(
        cfg,
        min_decision_margin=0.0,
        min_detection_score=0.0,
        max_reprojection_error_px=1e9,
        min_side_px=0.0,
        min_square_ratio=0.0,
    )


def _raw_detect(detector: AprilTagDetector, gray: np.ndarray) -> list[dict[str, Any]]:
    tags = detector._detector.detect(gray, estimate_tag_pose=False)
    out: list[dict[str, Any]] = []
    for tag in tags:
        marker_id = int(getattr(tag, "tag_id", -1))
        if marker_id < 0:
            continue
        out.append(
            {
                "id": marker_id,
                "decision_margin": float(getattr(tag, "decision_margin", 0.0)),
                "hamming": int(getattr(tag, "hamming", -1)),
            }
        )
    return out


def _summarize_raw(raw: list[dict[str, Any]]) -> tuple[list[int], float, float]:
    ids = sorted(int(x["id"]) for x in raw)
    margins = [float(x["decision_margin"]) for x in raw]
    if margins:
        return ids, float(np.min(margins)), float(np.mean(margins))
    return ids, 0.0, 0.0


def _variant_rows(
    detector_default: AprilTagDetector,
    detector_relaxed: AprilTagDetector,
    variants: dict[str, np.ndarray],
    out_img_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, gray in variants.items():
        gray_u8 = np.asarray(gray, dtype=np.uint8)
        _write_gray(out_img_dir / f"{name}.png", gray_u8)

        raw = _raw_detect(detector_default, gray_u8)
        raw_ids, raw_margin_min, raw_margin_mean = _summarize_raw(raw)

        default_out = detector_default.detect(gray_u8)
        default_ids = sorted(int(x) for x in default_out.keys())

        relaxed_out = detector_relaxed.detect(gray_u8)
        relaxed_ids = sorted(int(x) for x in relaxed_out.keys())

        rows.append(
            {
                "variant": name,
                "raw_count": len(raw_ids),
                "raw_ids": _ids_to_str(raw_ids),
                "raw_margin_min": raw_margin_min,
                "raw_margin_mean": raw_margin_mean,
                "default_count": len(default_ids),
                "default_ids": _ids_to_str(default_ids),
                "relaxed_count": len(relaxed_ids),
                "relaxed_ids": _ids_to_str(relaxed_ids),
            }
        )
    return rows


def _find_problem_frame(
    detector_default: AprilTagDetector,
    gray_paths: list[Path],
) -> tuple[Path, list[dict[str, Any]]]:
    if not gray_paths:
        raise RuntimeError("No grayscale inputs found for mirror test.")

    scan_rows: list[dict[str, Any]] = []
    for path in gray_paths:
        gray = _read_gray(path)
        raw = _raw_detect(detector_default, gray)
        raw_ids, raw_margin_min, raw_margin_mean = _summarize_raw(raw)
        default_out = detector_default.detect(gray)
        default_ids = sorted(int(x) for x in default_out.keys())
        scan_rows.append(
            {
                "frame": path.name,
                "raw_count": len(raw_ids),
                "raw_margin_min": raw_margin_min,
                "raw_margin_mean": raw_margin_mean,
                "default_count": len(default_ids),
                "default_ids": _ids_to_str(default_ids),
                "path": str(path).replace("\\", "/"),
            }
        )

    scan_rows.sort(
        key=lambda r: (
            int(r["default_count"]),
            float(r["raw_margin_mean"]) if int(r["raw_count"]) > 0 else float("inf"),
            float(r["raw_margin_min"]) if int(r["raw_count"]) > 0 else float("inf"),
            str(r["frame"]),
        )
    )
    chosen_name = str(scan_rows[0]["frame"])
    chosen_path = next(p for p in gray_paths if p.name == chosen_name)
    return chosen_path, scan_rows


def _test1_rotation(
    detector_default: AprilTagDetector,
    detector_relaxed: AprilTagDetector,
    out_dir: Path,
) -> dict[str, Any]:
    tag_path = DEFAULT_TAG_IMAGE
    if not tag_path.exists():
        dataset_frames = sorted(DEFAULT_DATASET_DIR.glob("frame_*.png"))
        if not dataset_frames:
            raise RuntimeError(
                f"Tag source not found: {DEFAULT_TAG_IMAGE} and no dataset frames in {DEFAULT_DATASET_DIR}"
            )
        gray = _read_gray(dataset_frames[0])
        source_path_str = str(dataset_frames[0]).replace("\\", "/")
    else:
        gray = _read_gray(tag_path)
        source_path_str = str(tag_path).replace("\\", "/")

    variants = {
        "original": gray,
        "rot90_k1": np.rot90(gray, k=1).copy(),
        "rot90_k2": np.rot90(gray, k=2).copy(),
        "rot90_k3": np.rot90(gray, k=3).copy(),
    }
    rows = _variant_rows(
        detector_default=detector_default,
        detector_relaxed=detector_relaxed,
        variants=variants,
        out_img_dir=out_dir / "images",
    )
    _write_csv(
        out_dir / "test1_rotation.csv",
        rows=rows,
        fieldnames=[
            "variant",
            "raw_count",
            "raw_ids",
            "raw_margin_min",
            "raw_margin_mean",
            "default_count",
            "default_ids",
            "relaxed_count",
            "relaxed_ids",
        ],
    )
    return {
        "source_path": source_path_str,
        "rows": rows,
        "csv_path": str((out_dir / "test1_rotation.csv").resolve()).replace("\\", "/"),
    }


def _test2_mirror(
    detector_default: AprilTagDetector,
    detector_relaxed: AprilTagDetector,
    out_dir: Path,
) -> dict[str, Any]:
    gray_paths = sorted(DEFAULT_GRAY_INPUT_DIR.glob("*.png"))
    if not gray_paths:
        dataset_paths = sorted(DEFAULT_DATASET_DIR.glob("frame_*.png"))
        if not dataset_paths:
            raise RuntimeError(
                f"No inputs for mirror test: {DEFAULT_GRAY_INPUT_DIR} and {DEFAULT_DATASET_DIR}"
            )
        gray_paths = dataset_paths

    chosen_path, scan_rows = _find_problem_frame(detector_default, gray_paths)

    if chosen_path.suffix.lower() == ".png":
        gray = _read_gray(chosen_path)
    else:
        gray = _read_gray(chosen_path)

    variants = {
        "original": gray,
        "fliplr": np.fliplr(gray).copy(),
        "flipud": np.flipud(gray).copy(),
    }
    rows = _variant_rows(
        detector_default=detector_default,
        detector_relaxed=detector_relaxed,
        variants=variants,
        out_img_dir=out_dir / "images",
    )
    _write_csv(
        out_dir / "test2_mirror.csv",
        rows=rows,
        fieldnames=[
            "variant",
            "raw_count",
            "raw_ids",
            "raw_margin_min",
            "raw_margin_mean",
            "default_count",
            "default_ids",
            "relaxed_count",
            "relaxed_ids",
        ],
    )
    _write_csv(
        out_dir / "test2_problem_scan.csv",
        rows=scan_rows,
        fieldnames=[
            "frame",
            "raw_count",
            "raw_margin_min",
            "raw_margin_mean",
            "default_count",
            "default_ids",
            "path",
        ],
    )
    shutil.copy2(chosen_path, out_dir / "chosen_problem_frame.png")
    return {
        "chosen_frame_path": str(chosen_path).replace("\\", "/"),
        "rows": rows,
        "csv_path": str((out_dir / "test2_mirror.csv").resolve()).replace("\\", "/"),
        "scan_csv_path": str((out_dir / "test2_problem_scan.csv").resolve()).replace("\\", "/"),
    }


def _iter_filter_test_inputs() -> list[Path]:
    gray_paths = sorted(DEFAULT_GRAY_INPUT_DIR.glob("*.png"))
    if gray_paths:
        return gray_paths
    return sorted(DEFAULT_DATASET_DIR.glob("frame_*.png"))


def _test3_filters(
    detector_default: AprilTagDetector,
    detector_relaxed: AprilTagDetector,
    out_dir: Path,
) -> dict[str, Any]:
    inputs = _iter_filter_test_inputs()
    if not inputs:
        raise RuntimeError(
            f"No inputs for filter test: {DEFAULT_GRAY_INPUT_DIR} or {DEFAULT_DATASET_DIR}"
        )

    gain_dir = out_dir / "frames_with_gain"
    gain_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in inputs:
        if path.parent.resolve() == DEFAULT_DATASET_DIR.resolve():
            gray = _import_cv2().cvtColor(_read_rgb(path), _import_cv2().COLOR_RGB2GRAY)
        else:
            gray = _read_gray(path)

        raw = _raw_detect(detector_default, gray)
        raw_ids, raw_margin_min, raw_margin_mean = _summarize_raw(raw)

        default_out = detector_default.detect(gray)
        default_ids = sorted(int(x) for x in default_out.keys())

        relaxed_out = detector_relaxed.detect(gray)
        relaxed_ids = sorted(int(x) for x in relaxed_out.keys())

        gained = sorted(x for x in relaxed_ids if x not in set(default_ids))
        if gained:
            _write_gray(gain_dir / path.name, gray)

        rows.append(
            {
                "frame": path.name,
                "raw_count": len(raw_ids),
                "raw_margin_min": raw_margin_min,
                "raw_margin_mean": raw_margin_mean,
                "default_count": len(default_ids),
                "default_ids": _ids_to_str(default_ids),
                "relaxed_count": len(relaxed_ids),
                "relaxed_ids": _ids_to_str(relaxed_ids),
                "gained_count_relaxed_minus_default": len(gained),
                "gained_ids_relaxed_minus_default": _ids_to_str(gained),
                "source_path": str(path).replace("\\", "/"),
            }
        )

    _write_csv(
        out_dir / "test3_filters_per_frame.csv",
        rows=rows,
        fieldnames=[
            "frame",
            "raw_count",
            "raw_margin_min",
            "raw_margin_mean",
            "default_count",
            "default_ids",
            "relaxed_count",
            "relaxed_ids",
            "gained_count_relaxed_minus_default",
            "gained_ids_relaxed_minus_default",
            "source_path",
        ],
    )

    default_counts = np.asarray([int(r["default_count"]) for r in rows], dtype=np.float32)
    relaxed_counts = np.asarray([int(r["relaxed_count"]) for r in rows], dtype=np.float32)
    gained_counts = np.asarray([int(r["gained_count_relaxed_minus_default"]) for r in rows], dtype=np.float32)
    frames_with_gain = [r for r in rows if int(r["gained_count_relaxed_minus_default"]) > 0]

    gained_ids_global: set[int] = set()
    for r in frames_with_gain:
        if str(r["gained_ids_relaxed_minus_default"]).strip():
            gained_ids_global.update(int(x) for x in str(r["gained_ids_relaxed_minus_default"]).split())

    return {
        "inputs_count": len(rows),
        "frames_with_gain": len(frames_with_gain),
        "frames_with_gain_ratio": float(len(frames_with_gain) / max(1, len(rows))),
        "default_count_mean": float(np.mean(default_counts)),
        "relaxed_count_mean": float(np.mean(relaxed_counts)),
        "default_count_min": int(np.min(default_counts)),
        "relaxed_count_min": int(np.min(relaxed_counts)),
        "total_gained_observations": int(np.sum(gained_counts)),
        "max_gained_in_one_frame": int(np.max(gained_counts)),
        "gained_ids_global": sorted(gained_ids_global),
        "csv_path": str((out_dir / "test3_filters_per_frame.csv").resolve()).replace("\\", "/"),
        "gain_frames_dir": str(gain_dir.resolve()).replace("\\", "/"),
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (OUT_ROOT / f"april_three_tests_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    default_cfg = _base_cfg()
    relaxed_cfg = _relaxed_cfg(default_cfg)
    detector_default = AprilTagDetector(default_cfg)
    detector_relaxed = AprilTagDetector(relaxed_cfg)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir).replace("\\", "/"),
        "default_thresholds": {
            "min_decision_margin": default_cfg.min_decision_margin,
            "min_detection_score": default_cfg.min_detection_score,
            "max_reprojection_error_px": default_cfg.max_reprojection_error_px,
            "min_side_px": default_cfg.min_side_px,
            "min_square_ratio": default_cfg.min_square_ratio,
        },
        "relaxed_thresholds": {
            "min_decision_margin": relaxed_cfg.min_decision_margin,
            "min_detection_score": relaxed_cfg.min_detection_score,
            "max_reprojection_error_px": relaxed_cfg.max_reprojection_error_px,
            "min_side_px": relaxed_cfg.min_side_px,
            "min_square_ratio": relaxed_cfg.min_square_ratio,
        },
    }

    test1_dir = out_dir / "test1_rotation"
    test2_dir = out_dir / "test2_mirror"
    test3_dir = out_dir / "test3_filters"
    test1_dir.mkdir(parents=True, exist_ok=True)
    test2_dir.mkdir(parents=True, exist_ok=True)
    test3_dir.mkdir(parents=True, exist_ok=True)

    summary["test1_rotation"] = _test1_rotation(
        detector_default=detector_default,
        detector_relaxed=detector_relaxed,
        out_dir=test1_dir,
    )
    summary["test2_mirror"] = _test2_mirror(
        detector_default=detector_default,
        detector_relaxed=detector_relaxed,
        out_dir=test2_dir,
    )
    summary["test3_filters"] = _test3_filters(
        detector_default=detector_default,
        detector_relaxed=detector_relaxed,
        out_dir=test3_dir,
    )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results: {out_dir}")
    print(f"Summary JSON: {summary_path}")
    print(
        "Filter test:",
        f"frames={summary['test3_filters']['inputs_count']},",
        f"frames_with_gain={summary['test3_filters']['frames_with_gain']},",
        f"total_gained={summary['test3_filters']['total_gained_observations']}",
    )


if __name__ == "__main__":
    main()
