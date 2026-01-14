from __future__ import annotations

from pathlib import Path

import numpy as np


# Edit these constants directly, no CLI.
SEED = 0
NUM_SAMPLES = 16
DATASET_PATH = Path("runs_spiral/vision_dataset.npz")
OUTPUT_DIR = Path("runs_spiral/vision_preview")


def _save_pgm(path: Path, img: np.ndarray) -> None:
    h, w = img.shape
    header = f"P5 {w} {h} 255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img.tobytes())


def _save_ppm(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    header = f"P6 {w} {h} 255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img.tobytes())


def _save_image(path: Path, img: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None

    if Image is not None:
        Image.fromarray(img).save(path.with_suffix(".png"))
        return

    if img.ndim == 2:
        _save_pgm(path.with_suffix(".pgm"), img)
    else:
        _save_ppm(path.with_suffix(".ppm"), img)


def _to_hwc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] in (1, 3):
        if img.shape[0] == 1:
            return img[0]
        return np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        if img.shape[-1] == 1:
            return img[:, :, 0]
        return img
    raise ValueError(f"Unexpected image shape: {img.shape}")


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    data = np.load(DATASET_PATH)
    images = data["images"]
    labels = data["labels"]
    if len(images) != len(labels):
        raise SystemExit("Images/labels length mismatch.")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(images), size=min(NUM_SAMPLES, len(images)), replace=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "index.txt").open("w", encoding="ascii") as f:
        f.write("index,ball_rel_x,ball_rel_y\n")
        for idx in indices:
            img = _to_hwc(images[idx])
            img = np.clip(img, 0, 255).astype(np.uint8)
            _save_image(OUTPUT_DIR / f"sample_{int(idx):05d}", img)
            x, y = labels[idx][:2]
            f.write(f"{int(idx)},{x:.6f},{y:.6f}\n")

    print(f"Saved {len(indices)} samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
