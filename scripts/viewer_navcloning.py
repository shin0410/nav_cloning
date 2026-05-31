#!/usr/bin/env python3
import os, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

VIEWS = ["left", "center", "right"]

def expand(p): 
    return Path(os.path.expanduser(p)).resolve()

def load_df(ds_dir: Path) -> pd.DataFrame:
    csv_path = ds_dir / "dataset" / "vel" / "data.csv"
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError(f"'episode' column not found: {csv_path}")
    # 文字列比較で確実にする
    df["_episode_str"] = df["episode"].astype(str)
    return df

def load_rgb(npy_path: Path):
    x = np.load(npy_path, mmap_mode="r")
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x[..., ::-1]  # BGR->RGB
    # 0..1 へ
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        mx = float(np.nanmax(x)) if x.size else 1.0
        x = x / 255.0 if mx > 1.5 else x
    return np.clip(x, 0, 1)

def img_path(ds_dir: Path, ep: str, view: str):
    return ds_dir / "dataset" / "img" / f"{ep}_{view}.npy"

def exists_ep(ds_dir: Path, ep: str) -> bool:
    return img_path(ds_dir, ep, "center").exists()

def resolve_dir(root: Path, spec: str) -> Path:
    p = expand(spec)
    if p.is_dir(): 
        return p
    p = root / spec
    if p.is_dir(): 
        return p
    raise FileNotFoundError(f"dataset dir not found: {spec}")

def get_eps(ds_dir: Path):
    df = load_df(ds_dir)
    eps = [str(e) for e in df["episode"].tolist()]
    if not eps:
        raise ValueError(f"no episodes: {ds_dir}")
    return df, eps

# ------------------------
# Mode A: 時刻順 viewer
# ------------------------
def run_time_viewer(root: Path, times, out_prefix: str):
    ds_dirs = [resolve_dir(root, t) for t in times]
    metas = []
    for d in ds_dirs:
        df, eps = get_eps(d)
        has_lux = "lux" in df.columns
        mean_lux = float(df["lux"].mean()) if has_lux else None
        metas.append((d, df, eps, has_lux, mean_lux))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for a in axes:
        a.axis("off")

    # スライダー（episode）
    plt.subplots_adjust(bottom=0.25)
    ax_s = plt.axes([0.18, 0.16, 0.64, 0.03])
    slider = Slider(ax_s, "Episode", 0, len(metas[0][2])-1, valinit=0, valstep=1)

    # ボタン：時刻 prev/next, episode prev/next, random, save
    ax_tprev = plt.axes([0.02, 0.03, 0.10, 0.08])
    ax_tnext = plt.axes([0.13, 0.03, 0.10, 0.08])
    ax_eprev = plt.axes([0.24, 0.03, 0.10, 0.08])
    ax_enext = plt.axes([0.35, 0.03, 0.10, 0.08])
    ax_rand  = plt.axes([0.46, 0.03, 0.10, 0.08])
    ax_save  = plt.axes([0.80, 0.03, 0.18, 0.08])

    btn_tprev = Button(ax_tprev, "Time Prev")
    btn_tnext = Button(ax_tnext, "Time Next")
    btn_eprev = Button(ax_eprev, "Ep Prev")
    btn_enext = Button(ax_enext, "Ep Next")
    btn_rand  = Button(ax_rand,  "Random")
    btn_save  = Button(ax_save,  "Save PNG")

    ti = [0]  # time index
    ei = [0]  # episode index

    im = [axes[j].imshow(np.zeros((10,10,3))) for j in range(3)]
    titles = [axes[j].set_title("") for j in range(3)]

    def episode_lux(df, ep_str):
        if "lux" not in df.columns:
            return None
        rows = df.loc[df["_episode_str"] == ep_str]
        if len(rows) == 0:
            return None
        try:
            return float(rows.iloc[0]["lux"])
        except Exception:
            return None

    def update_slider_limits():
        d, df, eps, has_lux, mean_lux = metas[ti[0]]
        slider.valmin = 0
        slider.valmax = len(eps)-1
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.set_val(0)
        ei[0] = 0

    def redraw():
        d, df, eps, has_lux, mean_lux = metas[ti[0]]
        ei[0] = max(0, min(ei[0], len(eps)-1))
        ep = eps[ei[0]]

        lux_ep = episode_lux(df, ep)
        lux_text = ""
        if mean_lux is not None:
            lux_text += f" mean_lux={mean_lux:.1f}"
        if lux_ep is not None:
            lux_text += f" ep_lux={lux_ep:.1f}"

        fig.suptitle(
            f"[TIME] {d.name}  ({ti[0]+1}/{len(metas)})  "
            f"episode={ep} ({ei[0]+1}/{len(eps)}){lux_text}",
            fontsize=14
        )

        for j, v in enumerate(VIEWS):
            p = img_path(d, ep, v)
            if not p.exists():
                im[j].set_data(np.zeros((10,10,3)))
                titles[j].set_text(f"{v}: missing")
            else:
                im[j].set_data(load_rgb(p))
                titles[j].set_text(v)
        fig.canvas.draw_idle()

    def on_slider(val):
        ei[0] = int(val)
        redraw()

    def on_time_prev(_):
        ti[0] = max(0, ti[0]-1)
        update_slider_limits()
        redraw()

    def on_time_next(_):
        ti[0] = min(len(metas)-1, ti[0]+1)
        update_slider_limits()
        redraw()

    def on_ep_prev(_):
        ei[0] = max(0, ei[0]-1)
        slider.set_val(ei[0])

    def on_ep_next(_):
        d, df, eps, has_lux, mean_lux = metas[ti[0]]
        ei[0] = min(len(eps)-1, ei[0]+1)
        slider.set_val(ei[0])

    def on_rand(_):
        d, df, eps, has_lux, mean_lux = metas[ti[0]]
        ei[0] = random.randint(0, len(eps)-1)
        slider.set_val(ei[0])

    def on_save(_):
        d, df, eps, has_lux, mean_lux = metas[ti[0]]
        ep = eps[ei[0]]
        out = f"{out_prefix}__TIME__{d.name}__ep{ep}.png"
        fig.savefig(out, dpi=220)
        print("[OK] saved:", out)

    slider.on_changed(on_slider)
    btn_tprev.on_clicked(on_time_prev)
    btn_tnext.on_clicked(on_time_next)
    btn_eprev.on_clicked(on_ep_prev)
    btn_enext.on_clicked(on_ep_next)
    btn_rand.on_clicked(on_rand)
    btn_save.on_clicked(on_save)

    redraw()
    plt.show()

# ------------------------
# Mode B: 4手法並べ viewer
# ------------------------
def run_methods_viewer(root: Path, baseline: str, taw: str, randaug: str, augmix: str,
                      out_prefix: str, prefer_aug_suffix: int = 0):
    ds = {
        "baseline": resolve_dir(root, baseline),
        "taw":      resolve_dir(root, taw),
        "randaug":  resolve_dir(root, randaug),
        "augmix":   resolve_dir(root, augmix),
    }

    df0, eps0 = get_eps(ds["baseline"])

    fig, axes = plt.subplots(3, 4, figsize=(16, 8))  # rows=views, cols=methods
    plt.subplots_adjust(bottom=0.22)

    ax_s = plt.axes([0.18, 0.13, 0.64, 0.03])
    slider = Slider(ax_s, "Episode", 0, len(eps0)-1, valinit=0, valstep=1)

    ax_prev = plt.axes([0.02, 0.03, 0.12, 0.08])
    ax_next = plt.axes([0.15, 0.03, 0.12, 0.08])
    ax_rand = plt.axes([0.28, 0.03, 0.12, 0.08])
    ax_save = plt.axes([0.80, 0.03, 0.18, 0.08])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    btn_rand = Button(ax_rand, "Random")
    btn_save = Button(ax_save, "Save PNG")

    methods = ["baseline", "taw", "randaug", "augmix"]
    col_titles = ["baseline", "TAW", "RandAug", "AugMix"]
    for c in range(4):
        axes[0, c].set_title(col_titles[c], fontsize=14)

    for r in range(3):
        for c in range(4):
            axes[r, c].axis("off")

    ims = [[axes[r,c].imshow(np.zeros((10,10,3))) for c in range(4)] for r in range(3)]
    subtxt = [[axes[r,c].text(0.02, 0.02, "", transform=axes[r,c].transAxes,
                              color="w", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2))
               for c in range(4)] for r in range(3)]

    idx = [0]

    def pick_ep_for_method(ep_base: str, m: str):
        if m == "baseline":
            return ep_base
        ep_aug = f"{ep_base}_{prefer_aug_suffix}"
        return ep_aug if exists_ep(ds[m], ep_aug) else ep_base

    def redraw():
        ep = eps0[idx[0]]
        fig.suptitle(
            f"[METHODS] episode={ep} ({idx[0]+1}/{len(eps0)})   (non-baseline prefers _{prefer_aug_suffix})",
            fontsize=14
        )
        for c, m in enumerate(methods):
            ep_use = pick_ep_for_method(ep, m)
            for r, v in enumerate(VIEWS):
                p = img_path(ds[m], ep_use, v)
                if p.exists():
                    ims[r][c].set_data(load_rgb(p))
                    subtxt[r][c].set_text(f"{m}:{ep_use}\n{v}")
                else:
                    ims[r][c].set_data(np.zeros((10,10,3)))
                    subtxt[r][c].set_text(f"{m}:{ep_use}\n{v}\nmissing")
        fig.canvas.draw_idle()

    def on_slider(val):
        idx[0] = int(val)
        redraw()

    def on_prev(_):
        idx[0] = max(0, idx[0]-1)
        slider.set_val(idx[0])

    def on_next(_):
        idx[0] = min(len(eps0)-1, idx[0]+1)
        slider.set_val(idx[0])

    def on_rand(_):
        idx[0] = random.randint(0, len(eps0)-1)
        slider.set_val(idx[0])

    def on_save(_):
        ep = eps0[idx[0]]
        out = f"{out_prefix}__METHODS__ep{ep}.png"
        fig.savefig(out, dpi=220)
        print("[OK] saved:", out)

    slider.on_changed(on_slider)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_rand.on_clicked(on_rand)
    btn_save.on_clicked(on_save)

    redraw()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/challenge_ws/src/nav_cloning/data")
    ap.add_argument("--mode", choices=["times", "methods"], required=True)

    ap.add_argument("--times", nargs="*", default=[])

    ap.add_argument("--baseline", default="")
    ap.add_argument("--taw", default="")
    ap.add_argument("--randaug", default="")
    ap.add_argument("--augmix", default="")
    ap.add_argument("--prefer_aug_suffix", type=int, default=0)

    ap.add_argument("--out_prefix", default="viewer_export")
    args = ap.parse_args()

    root = expand(args.root)

    if args.mode == "times":
        if not args.times:
            raise SystemExit("times mode requires --times <t1 t2 ...>")
        run_time_viewer(root, args.times, args.out_prefix)
    else:
        need = [args.baseline, args.taw, args.randaug, args.augmix]
        if any(s == "" for s in need):
            raise SystemExit("methods mode requires --baseline --taw --randaug --augmix")
        run_methods_viewer(
            root, args.baseline, args.taw, args.randaug, args.augmix,
            args.out_prefix, prefer_aug_suffix=args.prefer_aug_suffix
        )

if __name__ == "__main__":
    main()
