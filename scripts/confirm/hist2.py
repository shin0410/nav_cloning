import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

# 設定
BIN = 5
BATCH_SIZE = 128
try_num = "1"

# パス構成
pc_user_name = config["pc_user_name"]
ws_name = config["ws_name"]
speed = config["speed"]
mode = config["mode"]

public_path = f"/home/{pc_user_name}/ws/{ws_name}/src/nav_cloning/data"
input_dataset_path = os.path.join(public_path, speed, mode, "dataset", try_num, "tensor")

# データ読み込み
tensor_data = torch.load(os.path.join(input_dataset_path, "dataset.pt"))
print(f"The dataset contains {len(tensor_data)} samples.")

tensor_dataloader = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=False)

# 全角度を収集
all_angles_tensor = torch.cat([targets[:, 0] for _, targets in tensor_dataloader], dim=0)
all_angles = all_angles_tensor.cpu().numpy()

# BIN 計算（学習側と一致）
global_min_val, global_max_val = all_angles_tensor.min(), all_angles_tensor.max()
shared_bins = torch.linspace(global_min_val, global_max_val, BIN + 1).cpu().numpy()

# 統計情報
print(f"\nデータの総数: {len(all_angles)}")
print(f"角度の最小値: {np.min(all_angles):.4f}")
print(f"角度の最大値: {np.max(all_angles):.4f}")
print(f"角度の平均値: {np.mean(all_angles):.4f}")
print(f"角度の標準偏差: {np.std(all_angles):.4f}")

# ヒストグラム（件数・割合）
counts, bins = np.histogram(all_angles, bins=shared_bins)
hist_percent = counts / counts.sum() * 100

print("\n各範囲のデータ数と割合（%）:")
for i in range(len(counts)):
    if counts[i] > 0:
        print(f"範囲 [{bins[i]:.4f}, {bins[i+1]:.4f}]: {counts[i]}個 ({hist_percent[i]:.2f}%)")

# プロット（割合）
bin_centers = (bins[:-1] + bins[1:]) / 2
width = (bins[1] - bins[0])

plt.bar(bin_centers, hist_percent, width=width, color='b', alpha=0.9)
plt.title('Histogram of Angles (% per bin)')
plt.xlabel('Angle')
plt.ylabel('Percentage (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
