#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from augmix.augment_and_mix import augment_and_mix, MEAN, STD

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ---- 画像ユーティリティ（BGR <-> RGB / スケール） -----------------
def to_float01_bgr(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]

def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]

_mean = np.array(MEAN, dtype=np.float32)
_std  = np.array(STD , dtype=np.float32)

def denormalize_rgb(img_norm: np.ndarray) -> np.ndarray:
    """AugMix が返す正規化空間 → [0,1] RGB"""
    img = img_norm * _std[None, None, :] + _mean[None, None, :]
    return np.clip(img, 0.0, 1.0)

# --------------------------------------------------------------------

def augmix_variants_bgr(img_bgr: np.ndarray, K: int):
    """BGR(0..1) を受け取り、AugMix を K 回かけた BGR(0..1) の配列を返す"""
    variants = []
    img_rgb = bgr_to_rgb(img_bgr)  # AugMix は RGB 前提
    for _ in range(K):
        mixed = augment_and_mix(img_rgb.astype(np.float32))  # AugMix 実行（RGB, 0..1）
        denorm = denormalize_rgb(mixed)                      # 逆正規化して [0,1] に戻す
        out_bgr = rgb_to_bgr(denorm).astype(np.float32)
        variants.append(out_bgr)
    return variants

def main():
    cfg = load_config()

    PC_USER_NAME = cfg["pc_user_name"]
    WS_NAME      = cfg["ws_name"]
    TIME         = cfg["time"]

    # 入出力ディレクトリ名
    IN_IMG_DIRNAME  = cfg["input_aug_dataset_img"]   # 例: "img"
    IN_VEL_DIRNAME  = cfg["input_aug_dataset_vel"]   # 例: "vel"
    OUT_IMG_DIRNAME = cfg["output_aug_dataset_img"]  # 例: "augmix_img"
    OUT_VEL_DIRNAME = cfg["output_aug_dataset_vel"]  # 例: "augmix_vel"

    parser = argparse.ArgumentParser(description="AugMix 拡張（npy+CSV 版）")
    parser.add_argument("--K", type=int, default=3, help="AugMix 生成枚数（1エピソードあたり）")
    args = parser.parse_args()

    base_dir = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data/{TIME}/dataset"

    in_img_dir   = os.path.join(base_dir, IN_IMG_DIRNAME)
    in_vel_csv   = os.path.join(base_dir, IN_VEL_DIRNAME, "data.csv")
    out_img_dir  = os.path.join(base_dir, OUT_IMG_DIRNAME)
    out_vel_dir  = os.path.join(base_dir, OUT_VEL_DIRNAME)
    out_vel_csv  = os.path.join(out_vel_dir, "data.csv")

    if not os.path.exists(in_vel_csv):
        raise FileNotFoundError(f"Velocity CSV not found: {in_vel_csv}")
    if not os.path.isdir(in_img_dir):
        raise FileNotFoundError(f"Image dir not found: {in_img_dir}")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_vel_dir, exist_ok=True)

    df = pd.read_csv(in_vel_csv)
    episodes = df["episode"].unique().tolist()

    augmented_records = []
    missing_episodes = 0

    for ep in episodes:
        row = df[df["episode"] == ep].iloc[0].to_dict()

        # 3視点すべて揃っているか確認
        imgs = {}
        ok = True
        for pos in ("center", "left", "right"):
            f = os.path.join(in_img_dir, f"{ep}_{pos}.npy")
            if not os.path.exists(f):
                print(f"[WARN] missing image: {f}")
                ok = False
                break
            try:
                imgs[pos] = to_float01_bgr(np.load(f, mmap_mode="r"))
            except Exception as e:
                print(f"[ERR ] failed to load {f}: {e}")
                ok = False
                break
        if not ok:
            missing_episodes += 1
            continue

        # AugMix を K 回（3視点で同じ K 本数を作る）
        variants = {}
        for pos in ("center", "left", "right"):
            variants[pos] = augmix_variants_bgr(imgs[pos], args.K)

        # 書き出し
        for i in range(args.K):
            ep_i = f"{ep}_{i}"
            for pos in ("center", "left", "right"):
                out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                np.save(out_path, variants[pos][i])


            rec = dict(row)
            rec["episode"] = ep_i
            augmented_records.append(rec)

    # CSV 出力（上書き）
    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)

    print(f"[DONE] episodes in: {len(episodes)}, missing: {missing_episodes}")
    print(f"[DONE] images out : {len(os.listdir(out_img_dir))} files  -> {out_img_dir}")
    print(f"[DONE] labels out : {len(augmented_records)} rows       -> {out_vel_csv}")

if __name__ == "__main__":
    main()

