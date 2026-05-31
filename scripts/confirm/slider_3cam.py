import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import yaml
import pandas as pd


def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_data_root(config: dict, data_root_override: str = None) -> str:
    if data_root_override:
        p = os.path.abspath(os.path.expanduser(data_root_override))
        if os.path.isdir(p):
            return p
        raise FileNotFoundError(f"data_root not found: {p}")

    pc_user_name = str(config.get("pc_user_name", "")).strip()
    ws_name = str(config.get("ws_name", "")).strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_data = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))

    candidates = []
    if pc_user_name and ws_name:
        candidates.append(f"/home/{pc_user_name}/{ws_name}/src/nav_cloning/data")
        candidates.append(f"/home/{pc_user_name}/ws/{ws_name}/src/nav_cloning/data")
    candidates.append(local_data)

    for c in candidates:
        if os.path.isdir(c):
            return c

    raise FileNotFoundError(f"data root not found. tried={candidates}")


def parse_args():
    ap = argparse.ArgumentParser(description="3cam slider viewer (config override supported)")
    ap.add_argument("--config", default="config.yaml", help="config filename under nav_cloning/config")
    ap.add_argument("--time", default=None, help="dataset time id (e.g., 20260210_155034)")
    ap.add_argument("--data_root", default=None, help="data root path (default: infer from config)")
    ap.add_argument("--img_dir", default=None, help="image dir name under dataset/ (default: config input_confirm_dataset_img)")
    ap.add_argument("--vel_dir", default=None, help="vel dir name under dataset/ (default: config input_confirm_dataset_vel)")
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    time_id = args.time if args.time else str(config["time"])
    img_dir_name = args.img_dir if args.img_dir else str(config.get("input_confirm_dataset_img", "img"))
    vel_dir_name = args.vel_dir if args.vel_dir else str(config.get("input_confirm_dataset_vel", "vel"))

    public_path = resolve_data_root(config, args.data_root)
    input_dataset_path = os.path.join(public_path, time_id, "dataset")

    img_dir = os.path.join(input_dataset_path, img_dir_name)
    csv_path = os.path.join(input_dataset_path, vel_dir_name, "data.csv")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img dir not found: {img_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    episodes = df["episode"].unique()
    num_episodes = len(episodes)
    if num_episodes == 0:
        raise RuntimeError(f"no episodes found in csv: {csv_path}")

    views = ["left", "center", "right"]
    index = [0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(bottom=0.3)
    fig.suptitle(f"time={time_id}  img={img_dir_name}  vel={vel_dir_name}")

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
            img = np.load(img_path, mmap_mode="r")
            img = img[..., ::-1]  # BGR -> RGB
            images[v] = img
            angle = df.loc[df["episode"] == ep, v].values[0]
            angles[v] = angle
        return images, angles

    images, angles = load_images_and_angles(episodes[0])
    for ax, v in zip(axes, views):
        img_disp = ax.imshow(images[v])
        title = ax.set_title(f"{v}: {angles[v]:.4f}\n{episodes[0]}_{v}.npy")
        ax.axis("off")
        img_disps.append(img_disp)
        titles.append(title)

    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    slider = Slider(ax_slider, "Episode Index", 0, num_episodes - 1, valinit=0, valstep=1)

    axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
    btn_prev = Button(axprev, "Previous")
    btn_next = Button(axnext, "Next")

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

    def on_prev(_event):
        if index[0] > 0:
            index[0] -= 1
            update_display(index[0])

    def on_next(_event):
        if index[0] < num_episodes - 1:
            index[0] += 1
            update_display(index[0])

    slider.on_changed(on_slider_change)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    plt.show()


if __name__ == "__main__":
    main()
