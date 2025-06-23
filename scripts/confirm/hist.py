import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

PC_USER_NAME = config["pc_user_name"]
WS_NAME = config["ws_name"]
TIME = config["time"]
BIN = int(config["bin"])
INPUT_CONFIRM_DATASET_VEL = config["input_confirm_dataset_vel"]

public_path = f"/home/{PC_USER_NAME}/ws/{WS_NAME}/src/nav_cloning/data"
input_dataset_path = os.path.join(public_path, TIME, "dataset")

# 画像ディレクトリとcsvファイルのパス
csv_path = os.path.join(input_dataset_path, INPUT_CONFIRM_DATASET_VEL, "data.csv")

# CSVの読み込み
df = pd.read_csv(csv_path)

# center, left, rightの全角速度をまとめる
angles_center = df["center"].values
angles_left = df["left"].values
angles_right = df["right"].values

all_angles = np.concatenate([angles_center, angles_left, angles_right])

# 統計情報の表示
print(f"\nデータの総数: {len(all_angles)}")
print(f"角度の最小値: {np.min(all_angles):.4f}")
print(f"角度の最大値: {np.max(all_angles):.4f}")
print(f"角度の平均値: {np.mean(all_angles):.4f}")
print(f"角度の標準偏差: {np.std(all_angles):.4f}")

# ヒストグラム計算
counts, bins = np.histogram(all_angles, bins=BIN)
print("\n各範囲のデータ数:")
for i in range(len(counts)):
    if counts[i] > 0:
        print(f"範囲 [{bins[i]:.4f}, {bins[i+1]:.4f}]: {counts[i]}個")

# ヒストグラム描画
plt.hist(all_angles, bins=BIN, density=True, alpha=1, color='b')
plt.title('Histogram of Angles (CSV)')
plt.xlabel('Angle')
plt.ylabel('Density')
plt.show()
