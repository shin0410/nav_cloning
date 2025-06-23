import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import yaml
import pandas as pd

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# 設定の読み込み
config = load_config()

PC_USER_NAME = config["pc_user_name"]
WS_NAME = config["ws_name"]
TIME = config["time"]
INPUT_CONFIRM_DATASET_IMG = config["input_confirm_dataset_img"]
INPUT_CONFIRM_DATASET_VEL = config["input_confirm_dataset_vel"]

public_path = f"/home/{PC_USER_NAME}/ws/{WS_NAME}/src/nav_cloning/data"
input_dataset_path = os.path.join(public_path, TIME, "dataset")

# 画像ディレクトリとcsvファイルのパス
img_dir = os.path.join(input_dataset_path, INPUT_CONFIRM_DATASET_IMG)
csv_path = os.path.join(input_dataset_path, INPUT_CONFIRM_DATASET_VEL, "data.csv")

df = pd.read_csv(csv_path)

episodes = df['episode'].unique()
num_episodes = len(episodes)
views = ['left', 'center', 'right']

index = [0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(bottom=0.3)

img_disps = []
titles = []

def load_images_and_angles(ep):
    images = {}
    angles = {}
    for v in views:
        filename = f"{ep}_{v}.npy"
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = np.load(img_path, mmap_mode='r')  # もし巨大ならmmap_mode使うのもあり
        # BGR → RGB に変換
        img = img[..., ::-1]
        images[v] = img
        angle = df.loc[df['episode'] == ep, v].values[0]
        angles[v] = angle
    return images, angles

# 初期表示
images, angles = load_images_and_angles(episodes[0])
for ax, v in zip(axes, views):
    img_disp = ax.imshow(images[v])
    title = ax.set_title(f"{v}: {angles[v]:.4f}\n{episodes[0]}_{v}.npy")
    ax.axis('off')
    img_disps.append(img_disp)
    titles.append(title)

ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
slider = Slider(ax_slider, 'Episode Index', 0, num_episodes - 1, valinit=0, valstep=1)

axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_prev = Button(axprev, 'Previous')
btn_next = Button(axnext, 'Next')

def update_display(idx):
    ep = episodes[idx]
    images, angles = load_images_and_angles(ep)
    for i, v in enumerate(views):
        img_disps[i].set_data(images[v])
        titles[i].set_text(f"{v}: {angles[v]:.4f}\n{ep}_{v}.npy")
    if slider.val != idx:
        slider.set_val(idx)
    fig.canvas.draw_idle()

def on_slider_change(val):
    idx = int(val)
    index[0] = idx
    update_display(idx)

def on_prev(event):
    if index[0] > 0:
        index[0] -= 1
        update_display(index[0])

def on_next(event):
    if index[0] < num_episodes - 1:
        index[0] += 1
        update_display(index[0])

slider.on_changed(on_slider_change)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

plt.show()