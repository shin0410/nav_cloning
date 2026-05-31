#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIEWS = ["left", "center", "right"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create a 9x9 grid video from 3-camera dataset chunks and an optional faster version."
    )
    ap.add_argument("--data_root", default="/home/shin/challenge_ws/src/nav_cloning/data")
    ap.add_argument("--dataset", required=True, help="dataset time id, e.g. 20260315_13:41:50")
    ap.add_argument("--limit", type=int, required=True, help="use first N rows from vel/data.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid_rows", type=int, default=9)
    ap.add_argument("--grid_cols", type=int, default=9)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=5.0, help="speed multiplier for fast version")
    ap.add_argument("--tile_scale", type=int, default=2)
    return ap.parse_args()


def safe_episode(ep) -> str:
    s = str(ep)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_rgb_u8(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., ::-1]  # BGR -> RGB
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def put_header(canvas: np.ndarray, dataset: str, start_step: int, end_step: int, total_steps: int):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (24, 24, 24), thickness=-1)
    text1 = f"{dataset}   steps {start_step} - {end_step} / {total_steps - 1}"
    text2 = "layout: left center right x 27 steps = 9x9 tiles"
    cv2.putText(canvas, text1, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, text2, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    if args.grid_cols % len(VIEWS) != 0:
        raise SystemExit("--grid_cols must be divisible by 3 for left/center/right layout")

    steps_per_row = args.grid_cols // len(VIEWS)
    steps_per_frame = args.grid_rows * steps_per_row

    dataset_dir = Path(args.data_root) / args.dataset / "dataset"
    img_dir = dataset_dir / "img"
    csv_path = dataset_dir / "vel" / "data.csv"
    if not img_dir.is_dir():
        raise SystemExit(f"missing img dir: {img_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).head(args.limit).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no rows found in {csv_path}")

    episodes = [safe_episode(ep) for ep in df["episode"].tolist()]

    sample = load_rgb_u8(img_dir / f"{episodes[0]}_center.npy")
    tile_h, tile_w = sample.shape[:2]
    tile_h *= args.tile_scale
    tile_w *= args.tile_scale
    pad = 2
    header_h = 60

    frame_h = header_h + args.grid_rows * tile_h + (args.grid_rows + 1) * pad
    frame_w = args.grid_cols * tile_w + (args.grid_cols + 1) * pad

    normal_path = out_dir / f"{args.dataset}__first{args.limit}__grid9x9.mp4"
    fast_path = out_dir / f"{args.dataset}__first{args.limit}__grid9x9__x{args.speed:g}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(normal_path), fourcc, args.fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise SystemExit(f"failed to open video writer: {normal_path}")

    n_frames = math.ceil(len(episodes) / steps_per_frame)

    for frame_idx in range(n_frames):
        start = frame_idx * steps_per_frame
        end = min(start + steps_per_frame, len(episodes))

        canvas = np.full((frame_h, frame_w, 3), 248, dtype=np.uint8)
        put_header(canvas, args.dataset, start, end - 1, len(episodes))

        for local_step_idx, step_idx in enumerate(range(start, end)):
            row = local_step_idx // steps_per_row
            step_col = local_step_idx % steps_per_row
            ep = episodes[step_idx]

            for view_idx, view in enumerate(VIEWS):
                col = step_col * len(VIEWS) + view_idx
                x0 = pad + col * (tile_w + pad)
                y0 = header_h + pad + row * (tile_h + pad)

                img_path = img_dir / f"{ep}_{view}.npy"
                if not img_path.is_file():
                    tile = np.full((tile_h, tile_w, 3), 32, dtype=np.uint8)
                    cv2.putText(tile, "missing", (8, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                else:
                    tile = load_rgb_u8(img_path)
                    tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

                cv2.putText(
                    tile,
                    f"{step_idx}:{view[0].upper()}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[DONE] normal video: {normal_path}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(normal_path),
            "-vf",
            f"setpts=PTS/{args.speed}",
            "-an",
            str(fast_path),
        ],
        check=True,
    )
    print(f"[DONE] fast video  : {fast_path}")


if __name__ == "__main__":
    main()
