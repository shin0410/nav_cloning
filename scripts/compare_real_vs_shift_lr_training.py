#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/evaluate nav_cloning variants:
  1. real left/right camera images + default loss
  2. real left/right camera images + surprise-weighted loss
  3. center image shifted to synthesize left/right + default loss
  4. center image shifted to synthesize left/right + surprise-weighted loss
  5. five-view training: real left/center/right + shifted-center left/right
  6. wide-LR training: real left/center/right + real left/right shifted farther

The test set is evaluated on the center camera image, which is the usual policy
input. Outputs include models, prediction CSVs, MAE summary, and comparison
figures for angle distributions and teacher/prediction traces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


VIEWS = ("center", "left", "right")
TRAIN_TIMES_DEFAULT = (
    "20260428_12:31:05",
    "20260428_16:50:47",
    "20260428_19:29:22",
)
TEST_TIME_DEFAULT = "20260426_13_39_33"


class Net(nn.Module):
    def __init__(self, n_channel: int = 3, n_out: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layer(self.cnn_layer(x))


@dataclass(frozen=True)
class Sample:
    img_dir: Path
    episode: str
    view: str
    file_view: str
    transform: str
    target: float


def parse_times(raw: str | Sequence[str]) -> List[str]:
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = raw
    out: List[str] = []
    seen = set()
    for p in parts:
        s = str(p).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def dataset_paths(data_root: Path, time_id: str) -> Tuple[Path, Path]:
    base = data_root / time_id / "dataset"
    img_dir = base / "img"
    csv_path = base / "vel" / "data.csv"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"img dir not found: {img_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"data.csv not found: {csv_path}")
    return img_dir, csv_path


def load_csv(csv_path: Path, max_steps: int = 0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"episode", *VIEWS} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    if "phase" in df.columns:
        df = df[df["phase"] == "training"].reset_index(drop=True)
    if max_steps and max_steps > 0:
        df = df.head(int(max_steps)).reset_index(drop=True)
    return df


def normalize_img(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected image shape: {arr.shape}, expected HWC RGB/BGR")
    return arr


def shifted_view(
    image: np.ndarray,
    view: str,
    camera_angle: float,
    camera_fov: float,
) -> np.ndarray:
    if view == "center":
        return image
    yaw_angle = camera_angle if view == "left" else -camera_angle
    height, width = image.shape[:2]
    f = (width / 2.0) / math.tan(math.radians(camera_fov / 2.0))
    horizontal_shift = f * math.tan(math.radians(yaw_angle))
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


class NavDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        time_ids: Sequence[str],
        image_mode: str,
        max_steps: int = 0,
        camera_angle: float = 24.61,
        camera_fov: float = 150.0,
        extra_offset: float = 0.3,
    ):
        if image_mode not in {"real", "shift", "five", "wide_lr"}:
            raise ValueError("image_mode must be real, shift, five, or wide_lr")
        self.image_mode = image_mode
        self.camera_angle = float(camera_angle)
        self.camera_fov = float(camera_fov)
        self.extra_offset = float(extra_offset)
        self.samples: List[Sample] = []

        for time_id in time_ids:
            img_dir, csv_path = dataset_paths(data_root, time_id)
            df = load_csv(csv_path, max_steps=max_steps)
            for _, row in df.iterrows():
                episode = str(row["episode"]).split(".")[0]
                for view, file_view, transform, target in self._sample_specs(row):
                    if not math.isfinite(target):
                        continue
                    img_name = f"{episode}_{file_view}.npy"
                    if not (img_dir / img_name).is_file():
                        continue
                    self.samples.append(Sample(img_dir, episode, view, file_view, transform, target))

        if not self.samples:
            raise RuntimeError(f"no samples for image_mode={image_mode}, time_ids={time_ids}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_specs(self, row: pd.Series) -> List[Tuple[str, str, str, float]]:
        if self.image_mode == "real":
            return [(v, v, "none", float(row[v])) for v in VIEWS]

        if self.image_mode == "shift":
            return [(v, "center", "shift", float(row[v])) for v in VIEWS]

        if self.image_mode == "five":
            return [
                ("center", "center", "none", float(row["center"])),
                ("left", "left", "none", float(row["left"])),
                ("right", "right", "none", float(row["right"])),
                ("shift_center_left", "center", "shift", float(row["left"])),
                ("shift_center_right", "center", "shift", float(row["right"])),
            ]

        if self.image_mode == "wide_lr":
            return [
                ("center", "center", "none", float(row["center"])),
                ("left", "left", "none", float(row["left"])),
                ("right", "right", "none", float(row["right"])),
                ("wide_left", "left", "shift", float(row["left"]) - self.extra_offset),
                ("wide_right", "right", "shift", float(row["right"]) + self.extra_offset),
            ]

        raise ValueError(f"unsupported image_mode: {self.image_mode}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        arr = normalize_img(np.load(s.img_dir / f"{s.episode}_{s.file_view}.npy", mmap_mode="r"))
        if s.transform == "shift":
            shift_view = "left" if "left" in s.view else "right" if "right" in s.view else "center"
            arr = shifted_view(arr, shift_view, self.camera_angle, self.camera_fov)
        x = torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1).contiguous()
        y = torch.tensor([s.target], dtype=torch.float32)
        return x, y


class CenterEvalDataset(Dataset):
    def __init__(self, data_root: Path, time_id: str, max_steps: int = 0):
        img_dir, csv_path = dataset_paths(data_root, time_id)
        self.img_dir = img_dir
        self.df = load_csv(csv_path, max_steps=max_steps).copy()
        self.df["episode"] = self.df["episode"].astype(str).str.replace(r"\.0$", "", regex=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        episode = str(self.df.iloc[idx]["episode"])
        arr = normalize_img(np.load(self.img_dir / f"{episode}_center.npy", mmap_mode="r"))
        x = torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1).contiguous()
        return x, idx


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_targets(dataset: NavDataset) -> np.ndarray:
    return np.asarray([s.target for s in dataset.samples], dtype=np.float32)


def make_surprise_bins(targets: np.ndarray, bin_count: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = targets[np.isfinite(targets)]
    if targets.size == 0:
        raise RuntimeError("no valid targets for surprise bins")
    lo = float(targets.min())
    hi = float(targets.max())
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6
    bins = torch.linspace(lo, hi, int(bin_count) + 1, device=device)
    vals = torch.tensor(targets, dtype=torch.float32, device=device)
    idx = torch.bucketize(vals, bins) - 1
    idx = idx.clamp(min=0, max=int(bin_count) - 1)
    counts = torch.bincount(idx, minlength=int(bin_count)).float()
    probs = counts / (counts.sum() + 1e-6)
    return bins, probs


def surprise_weights(y: torch.Tensor, bins: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(y.flatten(), bins) - 1
    idx = idx.clamp(min=0, max=probs.numel() - 1)
    return -torch.log(probs[idx] + 1e-6)


def train_one(
    train_ds: NavDataset,
    loss_mode: str,
    epochs: int,
    batch_size: int,
    lr_min: float,
    device: torch.device,
    out_model: Path,
    bin_count: int,
    num_workers: int,
) -> Dict[str, List[float]]:
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    model = Net(3, 1).to(device)
    optimizer = optim.Adam(model.parameters(), eps=1e-2, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr_min)
    mse = nn.MSELoss(reduction="none" if loss_mode == "surprise" else "mean")
    bins = probs = None
    if loss_mode == "surprise":
        bins, probs = make_surprise_bins(collect_targets(train_ds), bin_count, device)

    history = {"loss": [], "lr": []}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x).squeeze(1)
            if loss_mode == "surprise":
                assert bins is not None and probs is not None
                weights = surprise_weights(y, bins, probs)
                loss_vec = mse(pred, y)
                # Match the existing learning_surprise.py policy: weighted MSE is
                # summed, not normalized by batch size or total weights.
                loss = (loss_vec * weights).sum()
            else:
                loss = mse(pred, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        avg_loss = running / max(1, len(loader))
        history["loss"].append(avg_loss)
        history["lr"].append(lr)
        print(f"  epoch {epoch + 1:03d}/{epochs} loss={avg_loss:.6f} lr={lr:.6g}", flush=True)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)
    return history


def predict_center(
    model_path: Path,
    eval_ds: CenterEvalDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = Net(3, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    preds = np.full(len(eval_ds), np.nan, dtype=np.float32)
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = model(x).squeeze(1).detach().cpu().numpy().astype(np.float32)
            preds[np.asarray(idx)] = y
    return preds


def calc_metrics(df: pd.DataFrame, pred_col: str, teacher_col: str = "center") -> Dict[str, float]:
    sub = df[[teacher_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    y = sub[teacher_col].to_numpy(dtype=float)
    yhat = sub[pred_col].to_numpy(dtype=float)
    err = yhat - y
    abs_err = np.abs(err)
    return {
        "count": int(len(sub)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_abs_error": float(np.percentile(abs_err, 50)),
        "p80_abs_error": float(np.percentile(abs_err, 80)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
        "bias_mean_error": float(np.mean(err)),
    }


def save_history_plot(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(np.arange(1, len(hist["loss"]) + 1), hist["loss"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=180)
    plt.close()


def plot_angle_distributions(data_root: Path, train_times: Sequence[str], test_time: str, out_dir: Path, max_steps: int) -> None:
    rows = []
    for time_id in [*train_times, test_time]:
        _, csv_path = dataset_paths(data_root, time_id)
        df = load_csv(csv_path, max_steps=max_steps)
        split = "test" if time_id == test_time else "train"
        for view in VIEWS:
            for val in pd.to_numeric(df[view], errors="coerce").dropna().to_numpy(dtype=float):
                rows.append({"time": time_id, "split": split, "view": view, "yaw_rate": val})
    all_df = pd.DataFrame(rows)
    all_df.to_csv(out_dir / "angle_distribution_values.csv", index=False)

    edges = np.linspace(-1.2, 1.2, 61)
    plt.figure(figsize=(11, 6))
    for time_id in train_times:
        vals = all_df[(all_df["time"] == time_id) & (all_df["view"] == "center")]["yaw_rate"]
        plt.hist(vals, bins=edges, histtype="step", linewidth=1.6, label=f"{time_id} center")
    vals = all_df[(all_df["time"] == test_time) & (all_df["view"] == "center")]["yaw_rate"]
    plt.hist(vals, bins=edges, histtype="step", linewidth=2.2, label=f"{test_time} center(test)")
    plt.xlabel("yaw rate [rad/s]")
    plt.ylabel("count")
    plt.title("Center teacher yaw-rate distribution by dataset")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "angle_distribution_center_by_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for view in VIEWS:
        vals = all_df[(all_df["split"] == "train") & (all_df["view"] == view)]["yaw_rate"]
        plt.hist(vals, bins=edges, alpha=0.35, label=f"train {view}")
    vals = all_df[(all_df["split"] == "test") & (all_df["view"] == "center")]["yaw_rate"]
    plt.hist(vals, bins=edges, histtype="step", linewidth=2.2, label="test center")
    plt.xlabel("yaw rate [rad/s]")
    plt.ylabel("count")
    plt.title("Train label distribution by view vs test center")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "angle_distribution_train_views_vs_test.png", dpi=180)
    plt.close()


def plot_condition_target_distributions(condition_targets: Dict[str, np.ndarray], out_dir: Path) -> None:
    rows = []
    for condition, vals in condition_targets.items():
        for val in vals[np.isfinite(vals)]:
            rows.append({"condition": condition, "yaw_rate": float(val)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "condition_train_target_distribution_values.csv", index=False)

    edges = np.linspace(-1.4, 1.4, 71)
    plt.figure(figsize=(12, 6))
    for condition, vals in condition_targets.items():
        vals = vals[np.isfinite(vals)]
        plt.hist(vals, bins=edges, histtype="step", linewidth=1.4, label=condition)
    plt.xlabel("training target yaw rate [rad/s]")
    plt.ylabel("count")
    plt.title("Training target distribution by condition")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "condition_train_target_distribution.png", dpi=180)
    plt.close()


def plot_teacher_by_dataset(data_root: Path, train_times: Sequence[str], test_time: str, out_dir: Path, max_steps: int) -> None:
    fig, axes = plt.subplots(len(train_times) + 1, 1, figsize=(12, 2.5 * (len(train_times) + 1)), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, time_id in zip(axes, [*train_times, test_time]):
        _, csv_path = dataset_paths(data_root, time_id)
        df = load_csv(csv_path, max_steps=max_steps)
        x = np.arange(len(df))
        ax.plot(x, df["center"].to_numpy(dtype=float), label="center", linewidth=1.0)
        ax.plot(x, df["left"].to_numpy(dtype=float), label="left", linewidth=0.8, alpha=0.8)
        ax.plot(x, df["right"].to_numpy(dtype=float), label="right", linewidth=0.8, alpha=0.8)
        role = "test" if time_id == test_time else "train"
        ax.set_title(f"{time_id} ({role}) teacher labels")
        ax.set_ylabel("yaw rate")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("episode")
    axes[0].legend(loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(out_dir / "teacher_labels_by_dataset_time.png", dpi=180)
    plt.close(fig)


def plot_predictions(df: pd.DataFrame, condition_names: Sequence[str], out_dir: Path) -> None:
    x = np.arange(len(df))
    plt.figure(figsize=(13, 6))
    plt.plot(x, df["center"].to_numpy(dtype=float), label="teacher center", color="black", linewidth=1.8)
    for name in condition_names:
        plt.plot(x, df[f"pred_{name}"].to_numpy(dtype=float), label=name, linewidth=1.0, alpha=0.85)
    plt.xlabel("episode")
    plt.ylabel("yaw rate [rad/s]")
    plt.title("Test teacher vs predictions")
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "test_teacher_vs_predictions_all.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 5))
    for name in condition_names:
        err = np.abs(df[f"pred_{name}"].to_numpy(dtype=float) - df["center"].to_numpy(dtype=float))
        plt.plot(x, err, label=name, linewidth=1.0)
    plt.xlabel("episode")
    plt.ylabel("absolute error [rad/s]")
    plt.title("Test absolute error over time")
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "test_abs_error_over_time_all.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(len(condition_names), 1, figsize=(13, 2.4 * len(condition_names)), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, name in zip(axes, condition_names):
        ax.plot(x, df["center"].to_numpy(dtype=float), label="teacher", color="black", linewidth=1.3)
        ax.plot(x, df[f"pred_{name}"].to_numpy(dtype=float), label="pred", linewidth=1.0)
        ax.set_title(name)
        ax.set_ylabel("yaw rate")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("episode")
    plt.tight_layout()
    plt.savefig(out_dir / "test_teacher_vs_prediction_each_condition.png", dpi=180)
    plt.close(fig)


def plot_mae(summary: pd.DataFrame, out_dir: Path) -> None:
    order = summary["condition"].tolist()
    plt.figure(figsize=(max(8, len(order) * 1.1), 5))
    color_by_mode = {
        "real": "#4c78a8",
        "shift": "#f58518",
        "five": "#54a24b",
        "wide_lr": "#b279a2",
    }
    colors = [color_by_mode.get(str(m), "#777777") for m in summary["image_mode"].tolist()]
    plt.bar(order, summary["mae"], color=colors)
    for i, v in enumerate(summary["mae"]):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("MAE [rad/s]")
    plt.title("Test MAE by training condition")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_condition.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--train-times", default=",".join(TRAIN_TIMES_DEFAULT))
    parser.add_argument("--test-time", default=TEST_TIME_DEFAULT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bin", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-angle", type=float, default=24.61)
    parser.add_argument("--camera-fov", type=float, default=150.0)
    parser.add_argument("--extra-offset", type=float, default=0.3)
    parser.add_argument(
        "--condition-set",
        choices=("basic", "new", "extended"),
        default="extended",
        help="basic: original 4, new: five/wide_lr only, extended: all 8",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    train_times = parse_times(args.train_times)
    test_time = str(args.test_time).strip()
    if not train_times:
        raise ValueError("--train-times is empty")
    if not test_time:
        raise ValueError("--test-time is empty")

    default_out = data_root / f"experiment_real_vs_shift_lr_train3_test_{test_time.replace(':', '-')}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] train_times={train_times}")
    print(f"[INFO] test_time={test_time}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] device={device} epochs={args.epochs} batch={args.batch_size}")
    shift_px = ((64 / 2.0) / math.tan(math.radians(args.camera_fov / 2.0))) * math.tan(math.radians(args.camera_angle))
    print(f"[INFO] synthetic left/right shift ~= {shift_px:.4f}px at 64px width")
    print(f"[INFO] wide_lr extra label offset={args.extra_offset}")

    basic_conditions = [
        ("real_default", "real", "default"),
        ("real_surprise", "real", "surprise"),
        ("shift_default", "shift", "default"),
        ("shift_surprise", "shift", "surprise"),
    ]
    new_conditions = [
        ("five_default", "five", "default"),
        ("five_surprise", "five", "surprise"),
        ("wide_lr_default", "wide_lr", "default"),
        ("wide_lr_surprise", "wide_lr", "surprise"),
    ]
    if args.condition_set == "basic":
        conditions = basic_conditions
    elif args.condition_set == "new":
        conditions = new_conditions
    else:
        conditions = basic_conditions + new_conditions

    histories: Dict[str, Dict[str, List[float]]] = {}
    condition_targets: Dict[str, np.ndarray] = {}
    for condition, image_mode, loss_mode in conditions:
        print(f"[TRAIN] {condition}", flush=True)
        train_ds = NavDataset(
            data_root=data_root,
            time_ids=train_times,
            image_mode=image_mode,
            max_steps=args.max_steps,
            camera_angle=args.camera_angle,
            camera_fov=args.camera_fov,
            extra_offset=args.extra_offset,
        )
        condition_targets[condition] = collect_targets(train_ds)
        model_path = out_dir / "models" / f"{condition}.pt"
        hist_path = out_dir / "models" / f"{condition}_history.json"
        if args.skip_existing and model_path.is_file() and hist_path.is_file():
            histories[condition] = json.loads(hist_path.read_text(encoding="utf-8"))
            print(f"  skip existing model: {model_path}")
            continue
        print(f"  samples={len(train_ds)}")
        history = train_one(
            train_ds=train_ds,
            loss_mode=loss_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_min=args.lr_min,
            device=device,
            out_model=model_path,
            bin_count=args.bin,
            num_workers=args.num_workers,
        )
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        histories[condition] = history

    print("[EVAL] center-camera test inference", flush=True)
    eval_ds = CenterEvalDataset(data_root=data_root, time_id=test_time, max_steps=args.max_steps)
    pred_df = eval_ds.df.copy()
    for condition, _, _ in conditions:
        model_path = out_dir / "models" / f"{condition}.pt"
        preds = predict_center(
            model_path=model_path,
            eval_ds=eval_ds,
            batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        pred_df[f"pred_{condition}"] = preds
    pred_csv = out_dir / "test_predictions_center.csv"
    pred_df.to_csv(pred_csv, index=False)

    summary_rows = []
    metrics_by_condition = {}
    for condition, image_mode, loss_mode in conditions:
        m = calc_metrics(pred_df, f"pred_{condition}", teacher_col="center")
        row = {"condition": condition, "image_mode": image_mode, "loss_mode": loss_mode, **m}
        summary_rows.append(row)
        metrics_by_condition[condition] = row
    summary = pd.DataFrame(summary_rows).sort_values("mae").reset_index(drop=True)
    summary.to_csv(out_dir / "mae_summary.csv", index=False)
    (out_dir / "mae_summary.json").write_text(json.dumps(metrics_by_condition, ensure_ascii=False, indent=2), encoding="utf-8")

    save_history_plot(histories, out_dir)
    plot_angle_distributions(data_root, train_times, test_time, out_dir, max_steps=args.max_steps)
    plot_condition_target_distributions(condition_targets, out_dir)
    plot_teacher_by_dataset(data_root, train_times, test_time, out_dir, max_steps=args.max_steps)
    condition_names = [c[0] for c in conditions]
    plot_predictions(pred_df, condition_names, out_dir)
    plot_mae(summary, out_dir)

    manifest = {
        "data_root": str(data_root),
        "train_times": train_times,
        "test_time": test_time,
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "bin": args.bin,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "camera_angle": args.camera_angle,
        "camera_fov": args.camera_fov,
        "extra_offset": args.extra_offset,
        "shift_px_at_width_64": shift_px,
        "device": str(device),
        "condition_set": args.condition_set,
        "conditions": condition_names,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]")
    print(summary[["condition", "mae", "rmse", "median_abs_error", "p95_abs_error"]].to_string(index=False))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
