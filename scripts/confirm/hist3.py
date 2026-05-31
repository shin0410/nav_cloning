import os
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
TRY_NUM_LIST = [str(i) for i in range(1, 11)]

# パス情報
pc_user_name = config["pc_user_name"]
ws_name = config["ws_name"]
speed = config["speed"]
mode = config["mode"]
public_path = f"/home/{pc_user_name}/ws/{ws_name}/src/nav_cloning/data"

# 角度データ収集
all_angles_dict = {}
for try_num in TRY_NUM_LIST:
    input_dataset_path = os.path.join(public_path, speed, mode, "dataset", try_num, "tensor")
    dataset_path = os.path.join(input_dataset_path, "dataset.pt")
    
    if not os.path.exists(dataset_path):
        print(f"try_num {try_num}: データセットが存在しません: {dataset_path}")
        continue

    tensor_data = torch.load(dataset_path)
    dataloader = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=False)
    angles_tensor = torch.cat([targets[:, 0] for _, targets in dataloader], dim=0)
    all_angles_dict[try_num] = angles_tensor.cpu().numpy()

# 全体の min/max から共通の bin を計算
all_angles_concat = np.concatenate(list(all_angles_dict.values()))
global_min = np.min(all_angles_concat)
global_max = np.max(all_angles_concat)
shared_bins = np.linspace(global_min, global_max, BIN + 1)
bin_centers = (shared_bins[:-1] + shared_bins[1:]) / 2
bar_width = (shared_bins[1] - shared_bins[0])

# プロット（2列×5行）
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))
axes = axes.flatten()

for idx, try_num in enumerate(TRY_NUM_LIST):
    if try_num not in all_angles_dict:
        continue
    angles = all_angles_dict[try_num]
    counts, _ = np.histogram(angles, bins=shared_bins)
    hist_percent = counts / counts.sum() * 100

    ax = axes[idx]
    ax.bar(bin_centers, hist_percent, width=bar_width, color='b', alpha=0.9)
    ax.set_title(f"try {try_num} (n={len(angles)})")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Percentage (%)")
    ax.grid(True)

# 不要なサブプロットを非表示
for i in range(len(TRY_NUM_LIST), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
