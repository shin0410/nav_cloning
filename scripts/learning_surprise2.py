#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning_surprise2.py（改）
- ネットワーク構造は learning_surprise.py と同一（conv1/2/3 + fc4/5, cnn_layer/fc_layer）
- 複数データセット結合（config.yaml の multi_datasets を利用）
- Shannon surprise の重みは正規化しない（-log p）
- 損失は weighted MSE を **sum** 集計（元の learning_surprise.py と同じ）

※ 入力画像は (H=48, W=64, C=1 or 3) を想定します。別サイズだと fc4(960) 次元と不整合になります。
"""
import os
import sys
import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2  # 使わないが将来の拡張のため残す
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# =====================
# 設定読み込み
# =====================

def load_config(filename: str = "config.yaml") -> dict:
    """../config/config.yaml を読み込む（従来互換）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
PC_USER_NAME: str = config["pc_user_name"]
WS_NAME: str = config["ws_name"]
BATCH_SIZE: int = int(config.get("batch_size", 64))
BIN: int = int(config.get("bin", 50))
TIME: str = config.get("time", "")  # 単一データセット用のtime（後方互換）
EPOCH: int = int(config.get("epoch", 30))
LOAD_DATASET_IMG: str = config.get("load_dataset_img", "img")
LOAD_DATASET_VEL: str = config.get("load_dataset_vel", "vel")
SAVE_MODEL: str = config.get("save_model", "model.pth")
MULTI_DATASETS: List[dict] = config.get("multi_datasets", [])

# CSVの角速度列名（従来の3視点）
VIEWS = ["center", "left", "right"]

# =====================
# Dataset
# =====================
class ImgCsvDataset(Dataset):
    """CSVの episode と各viewの角速度から、(npy画像, 角速度) を列挙するDataset。
       画像は {episode}_{view}.npy を想定。
    """
    def __init__(self, img_dir: str, csv_path: str, views: List[str] = VIEWS):
        self.img_dir = img_dir
        self.views = views
        self.data_pairs: List[Tuple[str, float]] = []

        df = pd.read_csv(csv_path)
        # 既知のカラムだけ残す
        keep_cols = [c for c in ["episode", *views] if c in df.columns]
        df = df[keep_cols].copy()

        # エピソードID正規化 & 値のクレンジング
        def _safe_angle(x):
            try:
                v = float(x)
            except Exception:
                return np.nan
            if math.isfinite(v):
                return v
            return np.nan

        for _, row in df.iterrows():
            if "episode" not in row or pd.isna(row["episode"]):
                continue
            episode = str(row["episode"]).split(".")[0]
            for view in self.views:
                if view not in row:
                    continue
                angle = _safe_angle(row[view])
                if np.isnan(angle):
                    continue
                img_path = os.path.join(self.img_dir, f"{episode}_{view}.npy")
                if not os.path.exists(img_path):
                    # 画像欠損はスキップ
                    continue
                self.data_pairs.append((img_path, angle))

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int):
        path, angle = self.data_pairs[idx]
        # .npyは (H, W, C) 期待。float32 0..1 に正規化。
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        # (H, W, C) -> (C, H, W)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:  # グレースケール対策
            arr = np.expand_dims(arr, 0)
        tensor_img = torch.from_numpy(arr)
        angle_t = torch.tensor([angle], dtype=torch.float32)
        return tensor_img, angle_t

# =====================
# ネットワーク（learning_surprise.py と同一構造）
# =====================
class Net(nn.Module):
    def __init__(self, n_channel: int = 3, n_out: int = 1):
        super().__init__()
        # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)  # 入力が(3,48,64)前提
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
        # <Weight init>
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        self.flatten = nn.Flatten()
        # <Sequential blocks>
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.flatten,
        )
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

    def forward(self, x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2

# =====================
# マルチデータセット / Surprise 重み / bin 計算
# =====================

def build_multi_dataset(public_path: str):
    """multi_datasets があればそれを結合。無ければ単一データセットを返す。
    Returns: (dataset, csv_paths, out_time)
    """
    datasets: List[Dataset] = []
    csv_paths: List[str] = []

    if MULTI_DATASETS:
        for d in MULTI_DATASETS:
            time_i = d.get("time")
            img_dir_name = d.get("img_dir", LOAD_DATASET_IMG)
            vel_dir_name = d.get("vel_dir", LOAD_DATASET_VEL)
            if not time_i:
                continue
            base = os.path.join(public_path, time_i, "dataset")
            img_dir = os.path.join(base, img_dir_name)
            csv_path = os.path.join(base, vel_dir_name, "data.csv")
            ds = ImgCsvDataset(img_dir, csv_path)
            if len(ds) == 0:
                continue
            datasets.append(ds)
            csv_paths.append(csv_path)
        if not datasets:
            raise RuntimeError("multi_datasets が指定されていますが有効なデータが見つかりませんでした。")
        return ConcatDataset(datasets), csv_paths, "multi"

    # 単一データセット（従来どおり）
    if not TIME:
        raise RuntimeError("config.yaml に time を指定してください（または multi_datasets を使用）。")
    base = os.path.join(public_path, TIME, "dataset")
    img_dir = os.path.join(base, LOAD_DATASET_IMG)
    csv_path = os.path.join(base, LOAD_DATASET_VEL, "data.csv")
    ds = ImgCsvDataset(img_dir, csv_path)
    if len(ds) == 0:
        raise RuntimeError("単一データセットが空です。img/CSVのパスや内容を確認してください。")
    return ds, [csv_path], TIME


def compute_global_bins(csv_paths: List[str], bin_count: int, device: torch.device):
    """全CSVの角速度分布から共通binとbin確率を算出。"""
    angles_list = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # 既知列から値を抜き出し
        cols = [c for c in VIEWS if c in df.columns]
        if not cols:
            continue
        vals = df[cols].values.astype(np.float32).reshape(-1)
        # クリーニング（NaN/Inf除去）
        vals = vals[np.isfinite(vals)]
        if vals.size:
            angles_list.append(vals)
    if not angles_list:
        raise RuntimeError("CSVから角速度が読み取れませんでした。")
    all_angles_np = np.concatenate(angles_list, axis=0)
    all_angles = torch.tensor(all_angles_np, dtype=torch.float32, device=device)

    global_min = torch.min(all_angles)
    global_max = torch.max(all_angles)
    if global_min == global_max:
        # すべて同一値のときは微小幅を与える
        global_min = global_min - 1e-6
        global_max = global_max + 1e-6

    bins = torch.linspace(global_min, global_max, bin_count + 1, device=device)
    # bin index を作成
    idx = torch.bucketize(all_angles, bins) - 1
    idx = torch.clamp(idx, min=0, max=bin_count - 1)
    counts = torch.bincount(idx, minlength=bin_count).float()
    probs = counts / (counts.sum() + 1e-6)
    return bins, probs


def surprise_weights(targets: torch.Tensor, bins: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """各ターゲット角速度に対する Shannon surprise 重み。
    w = -log( p(bin(target)) )。**正規化しない**（元の learning_surprise.py 準拠）
    """
    bin_count = probs.numel()
    idx = torch.bucketize(targets.flatten(), bins) - 1
    idx = torch.clamp(idx, min=0, max=bin_count - 1)
    p = probs[idx]
    w = -torch.log(p + 1e-6)
    return w

# =====================
# メイン学習ループ
# =====================

def main():
    public_path = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data"

    # 出力先は multi/ か time/ に分ける
    dataset, csv_paths, out_time = build_multi_dataset(public_path)

    output_model_path = os.path.join(public_path, out_time, "model", str(EPOCH))
    result_path = os.path.join(public_path, out_time, "result", str(EPOCH))
    os.makedirs(output_model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(result_path, "run"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 共通bin確率を算出
    global_bins, global_probs = compute_global_bins(csv_paths, BIN, device)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    # モデル/最適化器
    # 入力チャンネルは .npy のCに合わせる（一般には3）。最初のサンプルでチャンネル数を推定。
    sample_x, _ = dataset[0]
    in_ch = int(sample_x.shape[0])
    model = Net(n_channel=in_ch, n_out=1).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), eps=1e-8, weight_decay=5e-4)  # lr はデフォルト(1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    print(f"The dataset contains {len(dataset)} samples.")

    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        pbar = tqdm(total=len(dataloader.dataset), ncols=80, desc=f"epoch {epoch+1}/{EPOCH}")

        for images, targets in dataloader:
            images = images.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)  # (B,1)
            mse = criterion(outputs, targets)  # (B,1)

            w = surprise_weights(targets, global_bins, global_probs)  # (B,)
            # 元の learning_surprise.py と同じく **sum** で集計
            loss = (mse.flatten() * w).sum()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.update(images.size(0))

        pbar.close()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_loss = running_loss / max(1, len(dataloader))

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        print(f"epoch [{epoch+1}/{EPOCH}]  loss: {avg_loss:.6f}  lr: {current_lr:.6f}")

    # 保存
    torch.save(model.state_dict(), os.path.join(output_model_path, SAVE_MODEL))
    print(f"Saved model to: {os.path.join(output_model_path, SAVE_MODEL)}")


if __name__ == "__main__":
    main()

