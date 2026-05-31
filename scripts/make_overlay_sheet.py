#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def find_time_dirs(search_root: Path):
    # 01_teacher_vs_preds_overlay.png を含むディレクトリを「時刻フォルダ」とみなす
    imgs = list(search_root.rglob("01_teacher_vs_preds_overlay.png"))
    time_dirs = sorted({p.parent for p in imgs})
    return time_dirs

def infer_time_from_path(p: Path) -> str:
    # パス内の "20260112_09:49:31" を拾う（見つからなければフォルダ名）
    s = str(p)
    import re
    m = re.search(r"\d{8}_\d{2}:\d{2}:\d{2}", s)
    return m.group(0) if m else p.name

def load_img(path: Path, width: int):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if width is not None and w != width:
        nh = int(h * (width / w))
        im = im.resize((width, nh))
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_root", required=True, help="OUT_ROOT など（オーバーレイpngを含む場所）")
    ap.add_argument("--out", default="overlay_sheet.png")
    ap.add_argument("--times", nargs="*", default=None, help="並べたい eval_time を指定（省略で全て）")
    ap.add_argument("--mode", choices=["teacher_only", "all3"], default="all3",
                    help="teacher_only: 01だけ / all3: 01,02,03 を横に並べる")
    ap.add_argument("--img_width", type=int, default=520, help="各パネルの幅(px)")
    ap.add_argument("--pad", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.search_root)
    time_dirs = find_time_dirs(root)
    if not time_dirs:
        raise SystemExit(f"[ERR] not found: 01_teacher_vs_preds_overlay.png under {root}")

    rows = []
    for d in time_dirs:
        t = infer_time_from_path(d)
        if args.times is not None and len(args.times) > 0 and t not in set(args.times):
            continue
        rows.append((t, d))

    if not rows:
        raise SystemExit("[ERR] no rows after filtering --times")

    rows.sort(key=lambda x: x[0])  # 時刻順

    # 1行あたりの列数
    if args.mode == "teacher_only":
        keys = ["01_teacher_vs_preds_overlay.png"]
    else:
        keys = ["01_teacher_vs_preds_overlay.png", "02_abs_error_overlay.png", "03_residual_overlay.png"]

    # まず全画像を読み込み（欠けがあればプレースホルダ）
    loaded = []
    for t, d in rows:
        ims = []
        for k in keys:
            p = d / k
            if p.exists():
                ims.append(load_img(p, args.img_width))
            else:
                # placeholder
                ph = Image.new("RGB", (args.img_width, int(args.img_width*0.6)), (240,240,240))
                dr = ImageDraw.Draw(ph)
                dr.text((10,10), f"missing:\n{k}", fill=(0,0,0))
                ims.append(ph)
        loaded.append((t, ims))

    pad = args.pad
    label_h = 28

    # 行の高さ＝最大パネル高
    row_heights = [max(im.size[1] for im in ims) + label_h for _, ims in loaded]
    H = pad + sum(h + pad for h in row_heights)

    # 列幅
    col_w = args.img_width
    W = pad + len(keys) * (col_w + pad)

    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for (t, ims), rh in zip(loaded, row_heights):
        # time label
        draw.text((pad, y), t, fill=(0,0,0))
        y_img = y + label_h
        x = pad
        for im in ims:
            canvas.paste(im, (x, y_img))
            x += col_w + pad
        y += rh + pad

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[OK] saved: {out}")

if __name__ == "__main__":
    main()

