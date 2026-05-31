#!/usr/bin/env python3
"""
Surprise-weighted training with synthetic left/right images.

This is for datasets collected by scripts such as collect_teleop_mv.py where
left/right camera images and videos may exist, but training should ignore the
real left/right images. For every CSV row:

  - center target uses <episode>_center.npy unchanged
  - left target uses <episode>_center.npy shifted left
  - right target uses <episode>_center.npy shifted right

The loss weighting follows learning_surprise.py: each target angle is weighted
by Shannon surprise, -log(p(bin(angle))).
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

VIEWS = ("center", "left", "right")
DEFAULT_SAVE_MODEL = "model_surprise_center_shift.pt"


def load_config(filename: str = "config.yaml") -> dict:
    script_dir = Path(__file__).resolve().parent
    env_cfg = os.environ.get("NAV_CONFIG")
    if env_cfg:
        config_path = Path(env_cfg).expanduser().resolve()
    else:
        config_path = script_dir.parent / "config" / filename
    if not config_path.is_file():
        return {}
    with config_path.open("r") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


config = load_config()


def cfg_value(env_key: str, config_key: str, default):
    raw = os.environ.get(env_key)
    if raw is not None:
        return raw
    return config.get(config_key, default)


def cfg_int(env_key: str, config_key: str, default: int) -> int:
    return int(cfg_value(env_key, config_key, default))


def cfg_float(env_key: str, config_key: str, default: float) -> float:
    return float(cfg_value(env_key, config_key, default))


PC_USER_NAME = str(cfg_value("NAV_PC_USER_NAME", "pc_user_name", os.environ.get("USER", "shin")))
WS_NAME = str(cfg_value("NAV_WS_NAME", "ws_name", "ochi_ws"))
BATCH_SIZE = cfg_int("NAV_BATCH_SIZE", "batch_size", 32)
BIN = cfg_int("NAV_BIN", "bin", 20)
TIME = str(cfg_value("NAV_TIME", "time", "")).strip()
EPOCH = cfg_int("NAV_EPOCH", "epoch", 100)
LOAD_DATASET_IMG = str(cfg_value("NAV_LOAD_DATASET_IMG", "load_dataset_img", "img"))
LOAD_DATASET_VEL = str(cfg_value("NAV_LOAD_DATASET_VEL", "load_dataset_vel", "vel"))
SAVE_MODEL = str(os.environ.get("NAV_SAVE_MODEL", DEFAULT_SAVE_MODEL))
MAX_STEPS = cfg_int("NAV_MAX_STEPS", "max_steps", 0)
OUTPUT_TIME = str(os.environ.get("NAV_OUTPUT_TIME", "")).strip()
CAMERA_ANGLE = cfg_float("NAV_CAMERA_ANGLE", "camera_angle", 24.61)
CAMERA_FOV = cfg_float("NAV_CAMERA_FOV", "camera_fov", 150.0)

train_times_env = os.environ.get("NAV_TRAIN_TIMES")
if train_times_env:
    TRAIN_TIMES = [s.strip() for s in train_times_env.split(",") if s.strip()]
else:
    TRAIN_TIMES = config.get("train_times", [])


def normalize_train_times(raw_times) -> List[str]:
    if raw_times is None:
        return []
    if isinstance(raw_times, str):
        raw_times = [raw_times]
    if not isinstance(raw_times, (list, tuple)):
        return []
    normalized = []
    seen = set()
    for t in raw_times:
        t_str = str(t).strip()
        if t_str and t_str not in seen:
            seen.add(t_str)
            normalized.append(t_str)
    return normalized


TRAIN_TIMES = normalize_train_times(TRAIN_TIMES)


class Net(nn.Module):
    def __init__(self, n_channel: int, n_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)

        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.flatten,
        )
        self.fc_layer = nn.Sequential(self.fc4, self.relu, self.fc5)

    def forward(self, x):
        return self.fc_layer(self.cnn_layer(x))


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected image shape: {arr.shape}, expected HWC image")
    return arr


def shifted_view(image: np.ndarray, view: str, camera_angle: float, camera_fov: float) -> np.ndarray:
    if view == "center":
        return image
    yaw_angle = camera_angle if view == "left" else -camera_angle
    height, width = image.shape[:2]
    focal = (width / 2.0) / math.tan(math.radians(camera_fov / 2.0))
    horizontal_shift = focal * math.tan(math.radians(yaw_angle))
    y, x = np.indices((height, width), dtype=np.float32)
    map_x = x - np.float32(horizontal_shift)
    map_y = y
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class CenterShiftImgCsvDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        csv_path: Path,
        views: Sequence[str] = VIEWS,
        max_steps: int = 0,
        camera_angle: float = CAMERA_ANGLE,
        camera_fov: float = CAMERA_FOV,
    ):
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(csv_path)
        if max_steps and max_steps > 0:
            self.df = self.df.head(int(max_steps)).reset_index(drop=True)
        self.views = tuple(views)
        self.camera_angle = float(camera_angle)
        self.camera_fov = float(camera_fov)
        self.data_pairs: List[Tuple[Path, str, float]] = []

        for _, row in self.df.iterrows():
            episode = str(row["episode"]).split(".")[0]
            center_path = self.img_dir / f"{episode}_center.npy"
            if not center_path.is_file():
                continue
            for view in self.views:
                if view not in row:
                    continue
                angle = pd.to_numeric(row[view], errors="coerce")
                if not np.isfinite(angle):
                    continue
                self.data_pairs.append((center_path, view, float(angle)))

        if not self.data_pairs:
            raise RuntimeError(f"no samples found: img_dir={img_dir}, csv={csv_path}")

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int):
        center_path, view, angle = self.data_pairs[idx]
        img = normalize_image(np.load(center_path, mmap_mode="r"))
        img = shifted_view(img, view, self.camera_angle, self.camera_fov)
        img = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1)
        return img, torch.tensor(angle, dtype=torch.float32).unsqueeze(0)


def calculate_weighted_shannon_surprise(angles, bins, bin_probs, eps=1e-6):
    angles_binned = torch.bucketize(angles, bins) - 1
    angles_binned = angles_binned.clamp(min=0, max=bin_probs.numel() - 1)
    return -torch.log(bin_probs[angles_binned] + eps)


def resolve_data_root() -> Path:
    env_data_dir = os.environ.get("NAV_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser().resolve()
    default = Path(f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data")
    if default.exists():
        return default
    return Path(__file__).resolve().parent.parent / "data"


def resolve_dataset_paths(public_path: Path, time_id: str):
    dataset_base = public_path / time_id / "dataset"
    img_dir = dataset_base / LOAD_DATASET_IMG
    csv_path = dataset_base / LOAD_DATASET_VEL / "data.csv"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"img dir not found: {img_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    return img_dir, csv_path


def build_train_dataset(public_path: Path):
    if TRAIN_TIMES:
        datasets = []
        csv_paths = []
        used_times = []
        for time_id in TRAIN_TIMES:
            try:
                img_dir, csv_path = resolve_dataset_paths(public_path, time_id)
                ds = CenterShiftImgCsvDataset(img_dir, csv_path, max_steps=MAX_STEPS)
                datasets.append(ds)
                csv_paths.append(csv_path)
                used_times.append(time_id)
            except Exception as e:
                print(f"[WARN] skip {time_id}: {e}")
        if not datasets:
            raise RuntimeError("NAV_TRAIN_TIMES/train_times was set but no valid dataset was found.")
        if len(datasets) == 1:
            return datasets[0], csv_paths, used_times[0], used_times
        return ConcatDataset(datasets), csv_paths, "multi", used_times

    if not TIME:
        raise RuntimeError("Set NAV_TIME or train_times/NAV_TRAIN_TIMES.")
    img_dir, csv_path = resolve_dataset_paths(public_path, TIME)
    dataset = CenterShiftImgCsvDataset(img_dir, csv_path, max_steps=MAX_STEPS)
    return dataset, [csv_path], TIME, [TIME]


def iter_csv_angles(csv_paths: Iterable[Path]) -> np.ndarray:
    all_angles = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if MAX_STEPS and MAX_STEPS > 0:
            df = df.head(int(MAX_STEPS)).reset_index(drop=True)
        use_cols = [c for c in VIEWS if c in df.columns]
        if not use_cols:
            continue
        arr = df[use_cols].apply(pd.to_numeric, errors="coerce").values.flatten()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            all_angles.append(arr.astype(np.float32))
    if not all_angles:
        raise RuntimeError("No valid angle values were found in CSV files.")
    return np.concatenate(all_angles)


def compute_global_bins(csv_paths: Iterable[Path], device):
    all_angles = torch.tensor(iter_csv_angles(csv_paths), dtype=torch.float32, device=device)
    global_min_val, global_max_val = all_angles.min(), all_angles.max()
    if global_min_val == global_max_val:
        global_min_val = global_min_val - 1e-6
        global_max_val = global_max_val + 1e-6
    global_bins = torch.linspace(global_min_val, global_max_val, BIN + 1, device=device)
    all_angles_binned = torch.bucketize(all_angles, global_bins) - 1
    all_angles_binned = all_angles_binned.clamp(min=0, max=BIN - 1)
    global_bin_counts = torch.bincount(all_angles_binned, minlength=BIN).float()
    global_bin_probs = global_bin_counts / (global_bin_counts.sum() + 1e-6)
    return global_bins, global_bin_probs


def progress(total: int):
    if tqdm is None:
        class Dummy:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self, *_args):
                return None

        return Dummy()
    return tqdm(total=total)


def main():
    public_path = resolve_data_root()
    dataset, csv_paths, out_time, used_times = build_train_dataset(public_path)
    if OUTPUT_TIME:
        out_time = OUTPUT_TIME
    output_model_path = public_path / out_time / "model" / str(EPOCH)
    result_path = public_path / out_time / "result" / str(EPOCH) / "center_shift_surprise"
    output_model_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(result_path / "run"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Use data root: {public_path}")
    if len(used_times) > 1:
        print(f"Use train_times ({len(used_times)}): {', '.join(used_times)}")
    else:
        print(f"Use train time: {used_times[0]}")
    print("Image policy: use only *_center.npy; synthesize left/right by shifting center.")
    print(f"camera_angle={CAMERA_ANGLE}, camera_fov={CAMERA_FOV}")
    if OUTPUT_TIME:
        print(f"Save output time: {out_time}")
    if MAX_STEPS and MAX_STEPS > 0:
        print(f"Use max_steps: {MAX_STEPS} (expected samples ~= {MAX_STEPS * len(VIEWS)})")
    print(f"The dataset contains {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    sample_img, _ = dataset[0]
    model = Net(n_channel=int(sample_img.shape[0]), n_out=1).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), eps=1e-8, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)
    global_bins, global_bin_probs = compute_global_bins(csv_paths, device)

    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        with progress(len(dataset)) as pbar:
            for images, angles in dataloader:
                images = images.to(device)
                angles = angles.to(device).squeeze(1)
                shannon_weights = calculate_weighted_shannon_surprise(
                    angles,
                    global_bins,
                    global_bin_probs,
                )
                optimizer.zero_grad()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, angles)
                weighted_loss = (loss * shannon_weights).sum()
                weighted_loss.backward()
                optimizer.step()
                running_loss += weighted_loss.item()
                pbar.update(len(images))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_loss = running_loss / max(1, len(dataloader))
        writer.add_scalar("loss", avg_loss, epoch)
        print(f"epoch [{epoch + 1}/{EPOCH}], loss: {avg_loss:.4f}, lr: {current_lr:.6f}")

    writer.close()
    torch.save(model.state_dict(), output_model_path / SAVE_MODEL)
    print(f"Saved model: {output_model_path / SAVE_MODEL}")


if __name__ == "__main__":
    main()
