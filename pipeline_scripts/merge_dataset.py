#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, shutil
import pandas as pd

def norm_episode(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # "123.0" → "123"（元CSVがfloatで読まれる場合対策）
    s = s.str.replace(r"\.0$", "", regex=True)
    return s

def link_all(src_dir: str, dst_dir: str, mode: str):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if not name.endswith(".npy"):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError("mode must be symlink or copy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_img", required=True)
    ap.add_argument("--orig_csv", required=True)
    ap.add_argument("--aug_img", default=None)
    ap.add_argument("--aug_csv", default=None)
    ap.add_argument("--out_img", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--mode", default="symlink", choices=["symlink","copy"])
    args = ap.parse_args()

    # 画像（元）
    link_all(args.orig_img, args.out_img, args.mode)

    # CSV（元）
    df_o = pd.read_csv(args.orig_csv)
    if "episode" not in df_o.columns:
        raise ValueError(f"orig_csvにepisode列がありません: {args.orig_csv}")
    df_o["episode"] = norm_episode(df_o["episode"])

    # 増強があるなら追加
    if args.aug_img and args.aug_csv and os.path.exists(args.aug_img) and os.path.exists(args.aug_csv):
        link_all(args.aug_img, args.out_img, args.mode)
        df_a = pd.read_csv(args.aug_csv)
        if "episode" not in df_a.columns:
            raise ValueError(f"aug_csvにepisode列がありません: {args.aug_csv}")
        df_a["episode"] = norm_episode(df_a["episode"])
        # 列がズレても残す（lux等があっても壊さない）
        df = pd.concat([df_o, df_a], ignore_index=True, sort=False)
    else:
        df = df_o

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] out_img: {args.out_img}  out_csv: {args.out_csv}  rows={len(df)}")

if __name__ == "__main__":
    main()

