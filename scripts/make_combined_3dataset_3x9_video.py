#!/usr/bin/env python3
import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIEWS = ["left", "center", "right"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create a combined 3-dataset video with 3 time rows x 9 columns (dataset x camera)."
    )
    ap.add_argument("--data_root", default="/home/shin/challenge_ws/src/nav_cloning/data")
    ap.add_argument("--datasets", nargs=3, required=True)
    ap.add_argument("--limits", nargs=3, type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--tile_scale", type=int, default=2)
    ap.add_argument("--step_stride", type=int, default=3, help="advance this many steps per frame")
    return ap.parse_args()


def safe_episode(ep) -> str:
    s = str(ep)
    return s[:-2] if s.endswith('.0') else s


def load_rgb_u8(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode='r')
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    return (arr * 255.0).round().astype(np.uint8)


def put_global_header(canvas, datasets, base_step, end_step, usable_steps):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 86), (24, 24, 24), thickness=-1)
    title = "combined 3 datasets | rows=time(3) | cols=dataset x camera(3x3=9)"
    steps = f"steps {base_step} - {end_step} / {usable_steps - 1}"
    names = " | ".join(datasets)
    cv2.putText(canvas, title, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, steps, (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, names, (18, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)


def put_dataset_headers(canvas, datasets, tile_w, pad, header_h):
    for ds_idx, ds in enumerate(datasets):
        x0 = pad + ds_idx * 3 * (tile_w + pad)
        x1 = x0 + 3 * tile_w + 2 * pad
        cx = x0 + (x1 - x0) // 2
        cv2.putText(canvas, ds, (x0 + 8, header_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2, cv2.LINE_AA)
        for view_idx, view in enumerate(VIEWS):
            vx = x0 + view_idx * (tile_w + pad) + 8
            cv2.putText(canvas, view, (vx, header_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for dataset, limit in zip(args.datasets, args.limits):
        dataset_dir = Path(args.data_root) / dataset / 'dataset'
        img_dir = dataset_dir / 'img'
        csv_path = dataset_dir / 'vel' / 'data.csv'
        if not img_dir.is_dir():
            raise SystemExit(f'missing img dir: {img_dir}')
        if not csv_path.is_file():
            raise SystemExit(f'missing csv: {csv_path}')
        df = pd.read_csv(csv_path).head(limit).reset_index(drop=True)
        if len(df) < 3:
            raise SystemExit(f'not enough rows in {csv_path}: {len(df)}')
        episodes = [safe_episode(ep) for ep in df['episode'].tolist()]
        data.append({'dataset': dataset, 'img_dir': img_dir, 'episodes': episodes, 'limit': len(df)})

    usable_steps = min(item['limit'] for item in data)
    sample = load_rgb_u8(data[0]['img_dir'] / f"{data[0]['episodes'][0]}_center.npy")
    tile_h, tile_w = sample.shape[:2]
    tile_h *= args.tile_scale
    tile_w *= args.tile_scale
    pad = 2
    header_h = 86
    rows = 3
    cols = 9

    frame_h = header_h + rows * tile_h + (rows + 1) * pad
    frame_w = cols * tile_w + (cols + 1) * pad

    stem = '__'.join(args.datasets)
    normal_path = out_dir / f"{stem}__combined3x9.mp4"
    fast_path = out_dir / f"{stem}__combined3x9__x{args.speed:g}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(normal_path), fourcc, args.fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise SystemExit(f'failed to open writer: {normal_path}')

    starts = list(range(0, usable_steps - 2, args.step_stride))
    for base_step in starts:
        canvas = np.full((frame_h, frame_w, 3), 248, dtype=np.uint8)
        put_global_header(canvas, args.datasets, base_step, base_step + 2, usable_steps)
        put_dataset_headers(canvas, args.datasets, tile_w, pad, header_h)

        for row_idx in range(3):
            step_idx = base_step + row_idx
            y0 = header_h + pad + row_idx * (tile_h + pad)
            cv2.putText(canvas, f"t+{row_idx}  step {step_idx}", (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)

            for ds_idx, item in enumerate(data):
                ep = item['episodes'][step_idx]
                for view_idx, view in enumerate(VIEWS):
                    col = ds_idx * 3 + view_idx
                    x0 = pad + col * (tile_w + pad)
                    img_path = item['img_dir'] / f"{ep}_{view}.npy"
                    if img_path.is_file():
                        tile = load_rgb_u8(img_path)
                        tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        tile = np.full((tile_h, tile_w, 3), 32, dtype=np.uint8)
                        cv2.putText(tile, 'missing', (8, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                    cv2.putText(tile, ep, (6, tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                    canvas[y0:y0+tile_h, x0:x0+tile_w] = tile

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[DONE] normal video: {normal_path}")

    subprocess.run([
        'ffmpeg', '-y', '-i', str(normal_path), '-vf', f'setpts=PTS/{args.speed}', '-an', str(fast_path)
    ], check=True)
    print(f"[DONE] fast video  : {fast_path}")


if __name__ == '__main__':
    main()
