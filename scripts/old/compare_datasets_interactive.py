
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_datasets_interactive.py

2つのデータセット（ディレクトリ or .pt）を表示しながら、
SSIM / LPIPS / ΔE(Lab) / JS(輝度, Lab) をその場で計算・表示するインタラクティブビューア。

- .pt は list/tuple/ndarray/tensor/dict（imagesキー or left/center/right など）を想定
- 3カメラ(left/center/right)が検出された場合は 2行×3列（上: src, 下: tgt）で表示
- 単一ビューの場合は 1行×2列（左: src, 右: tgt）で表示
- スライダー／Next／Prevボタンでインデックスを移動

依存:
  pip install numpy pillow scikit-image torch torchvision lpips opencv-python matplotlib

使い方:
  python compare_datasets_interactive.py --src /mnt/data/dataset.pt --tgt /mnt/data/dataset_augmix.pt
  python compare_datasets_interactive.py --src /path/to/src_dir --tgt /path/to/tgt_dir --limit 200
  # dictの特定キーを使う場合（例: 'center'）
  python compare_datasets_interactive.py --src dataset.pt --tgt dataset_augmix.pt --image-key center
"""

import os
import sys
import glob
import math
import argparse
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import lpips
except Exception:
    lpips = None

# ---------------------------
# 画像ユーティリティ
# ---------------------------
def to_rgb_float01(x):
    """
    x: PIL.Image | np.ndarray(H,W,3|1|4) | torch.Tensor(C,H,W)/(H,W,C) | str(path)
    -> np.ndarray(H,W,3) float32 [0,1]
    """
    if isinstance(x, Image.Image):
        img = x.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3:
            # HWC or CHW
            if arr.shape[0] in (1,3,4) and (arr.shape[-1] not in (1,3,4)):
                arr = np.transpose(arr, (1,2,0))  # CHW->HWC
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    if torch.is_tensor(x):
        t = x
        if t.ndim == 3 and t.shape[0] in (1,3,4):   # CHW
            t = t[:3] if t.shape[0] >= 3 else t.repeat(3,1,1)
            t = t.permute(1,2,0)  # HWC
        elif t.ndim == 3 and t.shape[-1] in (1,3,4):  # HWC
            t = t[..., :3]
        elif t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1,1,3)
        arr = t.detach().cpu().numpy().astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    if isinstance(x, str) and os.path.isfile(x):
        return to_rgb_float01(Image.open(x))

    raise ValueError("Unsupported image type for to_rgb_float01")

def ssim_wrapper(img1, img2):
    # skimage >=0.19: channel_axis, 古い版: multichannel
    try:
        return float(ssim(img1, img2, data_range=1.0, channel_axis=2))
    except TypeError:
        return float(ssim(img1, img2, data_range=1.0, multichannel=True))

def psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def lpips_distance(img1, img2, lpips_model):
    if lpips_model is None:
        return None
    t1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0)
    t2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0)
    t1 = t1 * 2.0 - 1.0
    t2 = t2 * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_model(t1, t2)
    return float(d.item())

def deltaE_mean(img1, img2):
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    de = deltaE_ciede2000(lab1, lab2)
    return float(np.mean(de))

def _to_prob_hist(arr, bins=256, range_=(0,256)):
    h, _ = np.histogram(arr.ravel(), bins=bins, range=range_, density=False)
    p = h.astype(np.float64)
    p += 1e-12
    p /= p.sum()
    return p

def _kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def js_brightness(img1, img2, bins=256):
    g1 = cv2.cvtColor((img1*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p = _to_prob_hist(g1, bins=bins)
    q = _to_prob_hist(g2, bins=bins)
    return js_divergence(p, q)

def js_lab(img1, img2, bins=128):
    lab1 = cv2.cvtColor((img1*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    vals = []
    for ch in range(3):
        p = _to_prob_hist(lab1[:,:,ch], bins=bins)
        q = _to_prob_hist(lab2[:,:,ch], bins=bins)
        vals.append(js_divergence(p, q))
    return float(np.mean(vals))

# ---------------------------
# 入力ローダ
# ---------------------------
def load_dir_images(dir_path):
    # 再帰で探索。画像 + npy/npz に対応
    patterns = ["**/*.png","**/*.jpg","**/*.jpeg","**/*.bmp","**/*.tif","**/*.tiff",
                "**/*.npy","**/*.npz"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(dir_path, p), recursive=True))
    files = sorted(files)
    imgs, names = [], []
    for f in files:
        img = _load_any_path(f)
        if img is not None:
            imgs.append(img)
            # 名前は相対パスの方が分かりやすい
            names.append(os.path.relpath(f, dir_path))
    return names, imgs


def to_list_from_pt(obj, image_key=None):
    """
    .ptの内容を画像orビュー辞書のリストに正規化
      - 各要素がdictで left/center/right を持つ場合: そのまま
      - 各要素がdictで image_key を持つ場合: そのキーの画像を採用
      - 各要素が画像配列/テンソル: そのまま
      - 4次元(N,H,W,C or N,C,H,W)はN枚の画像リストに分解
    返り値: (items, mode)
      mode: "multi_view" or "single"
      items: list of dict(view->img) or list of img
    """
    # dict全体（imagesキーなど）
    if isinstance(obj, dict):
        if 'images' in obj:
            obj = obj['images']
        else:
            # 最初に見つかった配列/リストを採用
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
                    obj = v
                    break

    # list/tuple
    if isinstance(obj, (list, tuple)):
        # 要素がdictで left/center/right を持つ？
        if len(obj) > 0 and isinstance(obj[0], dict):
            keys = set(obj[0].keys())
            lcr = {'left','center','right'}
            if lcr.issubset(keys):
                # 各viewは画像配列に変換
                out = []
                for d in obj:
                    out.append({k: to_rgb_float01(d[k]) for k in lcr})
                return out, "multi_view"
            # 特定キー指定
            if image_key is not None and image_key in obj[0]:
                return [to_rgb_float01(d[image_key]) for d in obj], "single"
            # dictだけど扱えない → 画像っぽい最初の値を拾う
            candidate_key = None
            for k in obj[0].keys():
                try:
                    _ = to_rgb_float01(obj[0][k])
                    candidate_key = k
                    break
                except Exception:
                    pass
            if candidate_key is not None:
                return [to_rgb_float01(d[candidate_key]) for d in obj], "single"
            raise ValueError("Unsupported dict structure in .pt")
        # 要素が配列/テンソル/PIL
        return [to_rgb_float01(x) for x in obj], "single"

    # numpy 4D or 3D
    if isinstance(obj, np.ndarray):
        if obj.ndim == 4:
            return [to_rgb_float01(obj[i]) for i in range(obj.shape[0])], "single"
        if obj.ndim == 3:
            return [to_rgb_float01(obj)], "single"
        raise ValueError("Unsupported ndarray shape for images")

    # torch tensor
    if torch.is_tensor(obj):
        if obj.ndim == 4:
            return [to_rgb_float01(obj[i]) for i in range(obj.shape[0])], "single"
        if obj.ndim == 3:
            return [to_rgb_float01(obj)], "single"
        raise ValueError("Unsupported tensor shape for images")

    raise ValueError("Could not interpret .pt content as images")

def load_pt(path, image_key=None):
    obj = torch.load(path, map_location="cpu")
    items, mode = to_list_from_pt(obj, image_key=image_key)
    names = [f"{os.path.basename(path)}#{i:06d}" for i in range(len(items))]
    return names, items, mode

def load_source(path, image_key=None):
    if os.path.isdir(path):
        names, imgs = load_dir_images(path)
        return names, imgs, "single"
    if os.path.isfile(path) and path.lower().endswith(".pt"):
        return load_pt(path, image_key=image_key)
    raise ValueError(f"Unsupported path: {path}")

# ---------------------------
# メトリクス計算
# ---------------------------
def compute_metrics(img_s, img_t, lpips_model=None):
    # サイズ違いはtをsにリサイズ
    if img_s.shape[:2] != img_t.shape[:2]:
        img_t = np.array(Image.fromarray((img_t*255).astype(np.uint8)).resize((img_s.shape[1], img_s.shape[0]), Image.BILINEAR)).astype(np.float32)/255.0

    m = {}
    m['SSIM'] = ssim_wrapper(img_s, img_t)
    m['PSNR'] = psnr(img_s, img_t)
    m['LPIPS'] = lpips_distance(img_s, img_t, lpips_model)
    m['ΔE2000'] = deltaE_mean(img_s, img_t)
    m['JS_g'] = js_brightness(img_s, img_t)
    m['JS_Lab'] = js_lab(img_s, img_t)
    return m

def metrics_to_str(m):
    def fmt(x, p=4):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "NA"
        return f"{x:.{p}f}"
    return f"SSIM {fmt(m['SSIM'])} | PSNR {fmt(m['PSNR'])} | LPIPS {fmt(m['LPIPS'])} | ΔE {fmt(m['ΔE2000'])} | JS(g) {fmt(m['JS_g'])} | JS(Lab) {fmt(m['JS_Lab'])}"

# ---------------------------
# 描画
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="元データ（ディレクトリ or .pt）")
    ap.add_argument("--tgt", required=True, help="比較対象データ（ディレクトリ or .pt）")
    ap.add_argument("--limit", type=int, default=0, help="先頭からN件に制限")
    ap.add_argument("--image-key", default=None, help="dict項目から特定キーを画像として使う（例: center）")
    ap.add_argument("--lpips-net", default="alex", choices=["alex","vgg","squeeze"], help="LPIPSのバックボーン")
    args = ap.parse_args()

    names_s, items_s, mode_s = load_source(args.src, image_key=args.image_key)
    names_t, items_t, mode_t = load_source(args.tgt, image_key=args.image_key)

    if mode_s != mode_t:
        print(f"[WARN] src({mode_s}) と tgt({mode_t}) のモードが一致していません。単一ビューにフォールバックします。", file=sys.stderr)
        # 単一ビュー化
        if mode_s == "multi_view":
            items_s = [d['center'] if 'center' in d else list(d.values())[0] for d in items_s]
        if mode_t == "multi_view":
            items_t = [d['center'] if 'center' in d else list(d.values())[0] for d in items_t]
        mode = "single"
    else:
        mode = mode_s

    n = min(len(items_s), len(items_t))
    if args.limit > 0:
        n = min(n, args.limit)
    items_s = items_s[:n]
    items_t = items_t[:n]

    # LPIPS
    loss_fn = None
    if lpips is not None:
        loss_fn = lpips.LPIPS(net=args.lpips_net)

    # figure 構成
    if mode == "multi_view":
        views = ['left','center','right']
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        plt.subplots_adjust(bottom=0.22, wspace=0.05, hspace=0.25)
        # 初期描画
        idx = 0
        src0 = items_s[idx]
        tgt0 = items_t[idx]
        img_axes = []
        title_texts = []

        for j, v in enumerate(views):
            ax_src = axes[0, j]
            ax_tgt = axes[1, j]
            im1 = ax_src.imshow(src0[v])
            ax_src.axis('off')
            m = compute_metrics(src0[v], tgt0[v], lpips_model=loss_fn)
            im2 = ax_tgt.imshow(tgt0[v])
            ax_tgt.axis('off')
            ax_src.set_title(f"{v} (src)")
            ax_tgt.set_title(f"{v} (tgt)\n{metrics_to_str(m)}", fontsize=9)
            img_axes.append((im1, im2))
            title_texts.append((ax_src.title, ax_tgt.title))

        # スライダー & ボタン
        ax_slider = plt.axes([0.20, 0.10, 0.60, 0.03])
        slider = Slider(ax_slider, 'Index', 0, n-1, valinit=0, valstep=1)

        axprev = plt.axes([0.10, 0.04, 0.10, 0.05])
        axnext = plt.axes([0.80, 0.04, 0.10, 0.05])
        btn_prev = Button(axprev, 'Previous')
        btn_next = Button(axnext, 'Next')

        info_ax = plt.axes([0.20, 0.04, 0.58, 0.04])
        info_ax.axis('off')
        info_text = info_ax.text(0, 0.5, f"{names_s[0]}  vs  {names_t[0]}", va='center', fontsize=9)

        def update(i):
            i = int(i)
            src = items_s[i]
            tgt = items_t[i]
            for j, v in enumerate(views):
                # リサイズは compute_metrics 内で実施（t->s）
                img_axes[j][0].set_data(src[v])
                img_axes[j][1].set_data(tgt[v])
                m = compute_metrics(src[v], tgt[v], lpips_model=loss_fn)
                title_texts[j][0].set_text(f"{v} (src)")
                title_texts[j][1].set_text(f"{v} (tgt)\n{metrics_to_str(m)}")
            info_text.set_text(f"{names_s[i]}  vs  {names_t[i]}")
            fig.canvas.draw_idle()

        def on_slider(val):
            update(val)

        def on_prev(event):
            i = int(slider.val)
            if i > 0:
                slider.set_val(i-1)

        def on_next(event):
            i = int(slider.val)
            if i < n-1:
                slider.set_val(i+1)

        slider.on_changed(on_slider)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        plt.show()

    else:
        # 単一ビュー: 1行2列
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.22, wspace=0.02)
        idx = 0
        src0 = items_s[idx]
        tgt0 = items_t[idx]
        # サイズ合わせは compute_metrics 内
        m0 = compute_metrics(src0, tgt0, lpips_model=loss_fn)

        im_s = axes[0].imshow(src0)
        axes[0].axis('off')
        axes[0].set_title("src")

        im_t = axes[1].imshow(tgt0)
        axes[1].axis('off')
        axes[1].set_title("tgt\n" + metrics_to_str(m0), fontsize=10)

        ax_slider = plt.axes([0.20, 0.10, 0.60, 0.03])
        slider = Slider(ax_slider, 'Index', 0, len(items_s)-1, valinit=0, valstep=1)

        axprev = plt.axes([0.10, 0.04, 0.10, 0.05])
        axnext = plt.axes([0.80, 0.04, 0.10, 0.05])
        btn_prev = Button(axprev, 'Previous')
        btn_next = Button(axnext, 'Next')

        info_ax = plt.axes([0.20, 0.04, 0.58, 0.04])
        info_ax.axis('off')
        info_text = info_ax.text(0, 0.5, f"{names_s[0]}  vs  {names_t[0]}", va='center', fontsize=9)

        def update(i):
            i = int(i)
            src = items_s[i]
            tgt = items_t[i]
            m = compute_metrics(src, tgt, lpips_model=loss_fn)
            im_s.set_data(src)
            im_t.set_data(tgt)
            axes[1].set_title("tgt\n" + metrics_to_str(m), fontsize=10)
            info_text.set_text(f"{names_s[i]}  vs  {names_t[i]}")
            fig.canvas.draw_idle()

        def on_slider(val):
            update(val)

        def on_prev(event):
            i = int(slider.val)
            if i > 0:
                slider.set_val(i-1)

        def on_next(event):
            i = int(slider.val)
            if i < len(items_s)-1:
                slider.set_val(i+1)

        slider.on_changed(on_slider)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        plt.show()

if __name__ == "__main__":
    main()
