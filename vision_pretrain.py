from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from spiral_env import SpiralEnv, EnvCfg
import deb_v2 as deb
from vision_defaults import IMAGE_SIZE, CAMERA_MODE, CAMERA_DISTANCE_SCALE, CAMERA_LOOKAT_OFFSET

try:
    from tqdm import tqdm
except Exception:
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])

        def update(self, n=1):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(iterable=None, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)
from vision_models import VisionModelCfg, BallRegressor


# Edit these constants directly, no CLI.
SEED = 0
NUM_SAMPLES = 100_000
VAL_SPLIT = 0.1
EPOCHS = 40
BATCH_SIZE = 1024
LR = 3e-4
DR_STRENGTH = 1.0
DOMAIN_RANDOMIZATION = True
VISION_RANDOMIZATION = True
REGENERATE_DATASET = True
RANDOM_WARMUP_STEPS = (8, 20)
RANDOM_ACTION_SCALE = 1.0
FILTER_BY_REACH = True
MAX_BALL_DISTANCE_SCALE = 1.0

OUTPUT_DIR = Path("runs_spiral")
DATASET_PATH = OUTPUT_DIR / "vision_dataset.npz"
BACKBONE_PATH = OUTPUT_DIR / "vision_backbone.pt"
REGRESSOR_PATH = OUTPUT_DIR / "vision_regressor.pt"


class NumpyImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).float().div(255.0)
        label = torch.from_numpy(self.labels[idx]).float()
        return img, label


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = EnvCfg(
        seed=SEED,
        domain_randomization=DOMAIN_RANDOMIZATION,
        obs_mode="vision",
        state_include_ball=False,
        image_size=IMAGE_SIZE,
        image_grayscale=True,
        vision_randomization=VISION_RANDOMIZATION,
        camera_mode=CAMERA_MODE,
        camera_distance_scale=CAMERA_DISTANCE_SCALE,
        camera_lookat_offset=CAMERA_LOOKAT_OFFSET,
    )
    env = SpiralEnv(render_mode=None, cfg=cfg)
    env.set_dr_strength(DR_STRENGTH)

    images = np.empty((NUM_SAMPLES, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    labels = np.empty((NUM_SAMPLES, 2), dtype=np.float32)

    rng = np.random.default_rng(SEED + 123)
    max_ball_distance = float(deb.TARGET_LENGTH_M) * float(MAX_BALL_DISTANCE_SCALE)
    i = 0
    skipped = 0
    with tqdm(total=NUM_SAMPLES, desc="Collect") as pbar:
        while i < NUM_SAMPLES:
            env.reset()
            warmup = int(rng.integers(RANDOM_WARMUP_STEPS[0], RANDOM_WARMUP_STEPS[1] + 1))
            for _ in range(warmup):
                action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) * float(RANDOM_ACTION_SCALE)
                env.step(action)
            ball_rel = env.get_ball_rel()
            if FILTER_BY_REACH:
                dist = float(np.linalg.norm(ball_rel))
                if dist > max_ball_distance:
                    skipped += 1
                    continue
            images[i] = env.render_camera()
            labels[i] = ball_rel[:2]
            i += 1
            pbar.update(1)
    if skipped > 0:
        print(f"Skipped {skipped} samples beyond reach (>{max_ball_distance:.3f} m)")

    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels)
    return images, labels


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["images"], data["labels"]


def main() -> None:
    _set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and not REGENERATE_DATASET:
        images, labels = _load_dataset(DATASET_PATH)
        print(f"Loaded dataset: {DATASET_PATH} ({len(images)} samples)")
    else:
        images, labels = _collect_dataset(DATASET_PATH)
        print(f"Saved dataset: {DATASET_PATH} ({len(images)} samples)")

    dataset = NumpyImageDataset(images, labels)
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=use_cuda,
    )
    model_cfg = VisionModelCfg(image_size=IMAGE_SIZE, in_channels=1, features_dim=128)
    model = BallRegressor(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        model.train()
        train_loss = 0.0
        for imgs, labels_batch in train_loader:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * imgs.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)
                pred = model(imgs)
                loss = criterion(pred, labels_batch)
                val_loss += float(loss.item()) * imgs.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        print(f"Epoch {epoch:02d}/{EPOCHS} - train MSE: {train_loss:.6f} - val MSE: {val_loss:.6f}")

    torch.save({"cfg": asdict(model_cfg), "state_dict": model.state_dict()}, REGRESSOR_PATH)
    torch.save({"cfg": asdict(model_cfg), "state_dict": model.backbone.state_dict()}, BACKBONE_PATH)
    print("Saved:", REGRESSOR_PATH, BACKBONE_PATH)


if __name__ == "__main__":
    main()
