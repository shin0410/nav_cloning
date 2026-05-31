#!/usr/bin/env python3
import os, sys, glob, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

VIEWS = ["left", "center", "right"]

def expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def is_dataset_dir(d: str) -> bool:
    img_dir = os.path.join(d, "dataset", "img")
    csv_path = os.path.join(d, "dataset", "vel", "data.csv")
    return os.path.isdir(img_dir) and os.path.isfile(csv_path)

def scan_datasets(root: str):
    root = expand(root)
    items = []
    if not os.path.isdir(root):
        return items
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and is_dataset_dir(d):
            items.append(d)
    return items

def load_df(ds_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(ds_dir, "dataset", "vel", "data.csv")
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError(f"'episode' column not found in {csv_path}")
    return df

def pick_col(df: pd.DataFrame, view: str):
    if view in df.columns:
        return view
    for cand in [f"wz_{view}", f"ang_{view}", f"steer_{view}"]:
        if cand in df.columns:
            return cand
    return None

def load_image(ds_dir: str, episode, view: str):
    img_dir = os.path.join(ds_dir, "dataset", "img")
    path = os.path.join(img_dir, f"{episode}_{view}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = np.load(path, mmap_mode="r")
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[..., ::-1]  # BGR->RGB
    return img, path

def make_title(ds_name: str, episode, view: str, val, lux):
    s = f"{ds_name}\n{episode}_{view}.npy"
    if val is not None:
        s = f"{ds_name} | {view}={val:.4f}"
        if lux is not None:
            s += f" | lux={lux:.1f}"
        s += f"\n{episode}_{view}.npy"
    return s

def export_grid(ds_dir: str, out_png: str, rows: int = 12, seed: int = 0):
    rng = random.Random(seed)
    ds_name = os.path.basename(ds_dir)
    df = load_df(ds_dir)
    episodes = list(pd.unique(df["episode"]))
    if len(episodes) == 0:
        raise ValueError("No episodes found")

    pick = episodes if len(episodes) <= rows else rng.sample(episodes, rows)

    col_map = {v: pick_col(df, v) for v in VIEWS}
    has_lux = "lux" in df.columns

    fig, axes = plt.subplots(len(pick), 3, figsize=(12, 1.8 * len(pick)))
    if len(pick) == 1:
        axes = np.array([axes])

    for r, ep in enumerate(pick):
        row = df.loc[df["episode"] == ep].iloc[0]
        lux = float(row["lux"]) if has_lux else None
        for c, v in enumerate(VIEWS):
            ax = axes[r, c]
            img, _ = load_image(ds_dir, ep, v)
            ax.imshow(img)
            ax.axis("off")
            val = None
            if col_map[v] is not None:
                val = float(row[col_map[v]])
            ax.set_title(f"{v}={val:.3f}" if val is not None else v, fontsize=9)

    fig.suptitle(f"{ds_name}  (random {len(pick)} samples)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] saved grid: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/challenge_ws/src/nav_cloning/data", help="dataset root dir")
    ap.add_argument("--scan", default=None, help="scan and list datasets under this root and exit")
    ap.add_argument("--dataset", default=None, help="dataset name substring or full dir")
    ap.add_argument("--compare", default=None, help="compare dataset name substring or full dir")
    ap.add_argument("--export_grid", type=int, default=0, help="export montage rows and exit")
    ap.add_argument("--out", default="samples.png", help="output png path for export_grid or Save button prefix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scan is not None:
        ds = scan_datasets(args.scan)
        print("=== datasets ===")
        for i, d in enumerate(ds):
            print(f"[{i:02d}] {os.path.basename(d)}  ->  {d}")
        return

    root = expand(args.root)
    all_ds = scan_datasets(root)
    if len(all_ds) == 0:
        print(f"[ERR] no datasets found under: {root}")
        print("Try: python view_dataset_3cam.py --scan <root>")
        sys.exit(1)

    def resolve(spec: str):
        if spec is None:
            return None
        spec2 = expand(spec)
        if os.path.isdir(spec2) and is_dataset_dir(spec2):
            return spec2
        hits = [d for d in all_ds if spec in os.path.basename(d)]
        if len(hits) == 0:
            print(f"[ERR] dataset not found for: {spec}")
            print("Available:")
            for d in all_ds[:100]:
                print(" -", os.path.basename(d))
            sys.exit(1)
        if len(hits) > 1:
            print(f"[WARN] multiple hits for '{spec}', using first:")
            for h in hits[:10]:
                print(" -", os.path.basename(h))
        return hits[0]

    dsA = resolve(args.dataset) if args.dataset else all_ds[0]
    dsB = resolve(args.compare) if args.compare else None

    if args.export_grid > 0:
        export_grid(dsA, args.out, rows=args.export_grid, seed=args.seed)
        return

    compare_mode = dsB is not None
    if compare_mode:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        plt.subplots_adjust(bottom=0.22)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(bottom=0.22)

    def load_meta(ds_dir):
        df = load_df(ds_dir)
        episodes = list(pd.unique(df["episode"]))
        col_map = {v: pick_col(df, v) for v in VIEWS}
        has_lux = "lux" in df.columns
        return df, episodes, col_map, has_lux

    dfA, epsA, colA, hasLuxA = load_meta(dsA)
    if compare_mode:
        dfB, epsB, colB, hasLuxB = load_meta(dsB)
        setB = set(epsB)

    idx = [0]
    img_objs = []
    title_objs = []

    def draw_row(ax_row, ds_dir, df, eps, col_map, has_lux, row_i, forced_ep=None):
        ep = forced_ep if forced_ep is not None else eps[idx[0] % len(eps)]
        row = df.loc[df["episode"] == ep].iloc[0]
        lux = float(row["lux"]) if has_lux else None
        for j, v in enumerate(VIEWS):
            ax = ax_row[j]
            img, _ = load_image(ds_dir, ep, v)
            k = row_i*3 + j
            if len(img_objs) <= k:
                im = ax.imshow(img)
                ax.axis("off")
                img_objs.append(im)
                title_objs.append(ax.set_title("", fontsize=10))
            else:
                img_objs[k].set_data(img)
            val = None
            if col_map[v] is not None:
                val = float(row[col_map[v]])
            title_objs[k].set_text(make_title(os.path.basename(ds_dir), ep, v, val, lux))

    def redraw():
        epA = epsA[idx[0] % len(epsA)]
        if compare_mode:
            draw_row(axes[0], dsA, dfA, epsA, colA, hasLuxA, 0, forced_ep=epA)
            if epA in setB:
                draw_row(axes[1], dsB, dfB, epsB, colB, hasLuxB, 1, forced_ep=epA)
            else:
                draw_row(axes[1], dsB, dfB, epsB, colB, hasLuxB, 1, forced_ep=epsB[idx[0] % len(epsB)])
        else:
            draw_row(axes, dsA, dfA, epsA, colA, hasLuxA, 0, forced_ep=epA)
        fig.canvas.draw_idle()

    redraw()

    ax_slider = plt.axes([0.2, 0.13, 0.6, 0.03])
    max_idx = len(epsA) - 1
    slider = Slider(ax_slider, "Episode", 0, max_idx, valinit=0, valstep=1)

    ax_prev = plt.axes([0.05, 0.03, 0.12, 0.07])
    ax_next = plt.axes([0.18, 0.03, 0.12, 0.07])
    ax_rand = plt.axes([0.31, 0.03, 0.12, 0.07])
    ax_save = plt.axes([0.57, 0.03, 0.12, 0.07])
    ax_grid = plt.axes([0.70, 0.03, 0.12, 0.07])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    btn_rand = Button(ax_rand, "Random")
    btn_save = Button(ax_save, "Save PNG")
    btn_grid = Button(ax_grid, "Export Grid")

    def on_slider(val):
        idx[0] = int(val)
        redraw()

    def on_prev(_):
        idx[0] = max(0, idx[0]-1)
        slider.set_val(idx[0])

    def on_next(_):
        idx[0] = min(max_idx, idx[0]+1)
        slider.set_val(idx[0])

    def on_rand(_):
        idx[0] = random.randint(0, max_idx)
        slider.set_val(idx[0])

    def on_save(_):
        base = os.path.splitext(args.out)[0]
        ep = epsA[idx[0] % len(epsA)]
        out = f"{base}__{os.path.basename(dsA)}"
        if compare_mode:
            out += f"__VS__{os.path.basename(dsB)}"
        out += f"__ep{ep}.png"
        fig.savefig(out, dpi=200)
        print(f"[OK] saved: {out}")

    def on_grid(_):
        base = os.path.splitext(args.out)[0]
        out = f"{base}__{os.path.basename(dsA)}__grid.png"
        export_grid(dsA, out, rows=12, seed=args.seed)

    slider.on_changed(on_slider)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_rand.on_clicked(on_rand)
    btn_save.on_clicked(on_save)
    btn_grid.on_clicked(on_grid)

    plt.show()

if __name__ == "__main__":
    main()
