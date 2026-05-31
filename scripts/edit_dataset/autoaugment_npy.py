#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import argparse
import random
import numpy as np
import pandas as pd
import yaml

from typing import Optional, Union, List
from PIL import Image
from typing import Optional, Union, List

# torchvision
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import torchvision.transforms.functional as TF


def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, "r") as f:
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


def bgr01_to_pil_rgb(img_bgr01: np.ndarray) -> Image.Image:
    img_rgb01 = bgr_to_rgb(img_bgr01)
    img_u8 = (np.clip(img_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def pil_rgb_to_bgr01(img_pil: Image.Image) -> np.ndarray:
    rgb_u8 = np.asarray(img_pil).astype(np.uint8)
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    return rgb_to_bgr(rgb01).astype(np.float32)


# --------------------------------------------------------------------
# Photometric-only "AutoAugment-like" (no geometry) for regression safety
# - 2 ops, each with prob p, magnitude sampled like torchvision's ranges.
# - Ops chosen from: Brightness/Color/Contrast/Sharpness/Posterize/Solarize/AutoContrast/Equalize/Invert
#   (these exist in torchvision's augmentation space)  ※幾何系は使わない
# --------------------------------------------------------------------
_PHOTO_OPS = [
    "Brightness",
    "Color",
    "Contrast",
    "Sharpness",
    "Posterize",
    "Solarize",
    "AutoContrast",
    "Equalize",
    "Invert",
]


def _sample_magnitude(op: str, num_bins: int = 11) -> Optional[Union[float, int]]:
    # Ranges follow torchvision v2 augmentation space idea:
    # Brightness/Color/Contrast/Sharpness: [0.0 .. 0.9] (signed)
    # Posterize: bits ~ 8 .. 4
    # Solarize: [1.0 .. 0.0] (used as threshold ratio)
    level = random.randint(0, num_bins - 1)

    if op in ("Brightness", "Color", "Contrast", "Sharpness"):
        mags = np.linspace(0.0, 0.9, num_bins, dtype=np.float32)
        mag = float(mags[level])
        # signed
        if random.random() < 0.5:
            mag = -mag
        return mag

    if op == "Posterize":
        # mimic: (8 - (arange(num_bins) / ((num_bins - 1) / 4))).round().int()
        ar = np.arange(num_bins, dtype=np.float32)
        bits = 8.0 - (ar / ((num_bins - 1) / 4.0))
        bits = int(np.round(bits[level]))
        bits = max(1, min(8, bits))
        return bits

    if op == "Solarize":
        mags = np.linspace(1.0, 0.0, num_bins, dtype=np.float32)
        return float(mags[level])

    # AutoContrast/Equalize/Invert have no magnitude
    return None


def _apply_photo_op(img: Image.Image, op: str, mag) -> Image.Image:
    if op == "Brightness":
        return TF.adjust_brightness(img, 1.0 + float(mag))
    if op == "Color":
        return TF.adjust_saturation(img, 1.0 + float(mag))
    if op == "Contrast":
        return TF.adjust_contrast(img, 1.0 + float(mag))
    if op == "Sharpness":
        return TF.adjust_sharpness(img, 1.0 + float(mag))
    if op == "Posterize":
        return TF.posterize(img, int(mag))
    if op == "Solarize":
        # mag is ratio in [0..1], threshold=255*ratio
        return TF.solarize(img, threshold=255.0 * float(mag))
    if op == "AutoContrast":
        return TF.autocontrast(img)
    if op == "Equalize":
        return TF.equalize(img)
    if op == "Invert":
        return TF.invert(img)
    return img


def photo_autoaugment_pil(img: Image.Image, num_ops: int = 2, p: float = 0.5, num_bins: int = 11) -> Image.Image:
    out = img
    ops = random.sample(_PHOTO_OPS, k=min(num_ops, len(_PHOTO_OPS)))
    for op in ops:
        if random.random() <= p:
            mag = _sample_magnitude(op, num_bins=num_bins)
            out = _apply_photo_op(out, op, mag)
    return out


# --------------------------------------------------------------------
def autoaugment_variants_bgr(img_bgr01: np.ndarray, K: int, policy: str, no_geometry: bool) -> List[np.ndarray]:
    """
    BGR(0..1) -> AutoAugment（既存ポリシー）or Photometric-only版 を K 回
    返り値: BGR(0..1) のlist
    """
    pil = bgr01_to_pil_rgb(img_bgr01)

    if not no_geometry:
        pol = {
            "IMAGENET": AutoAugmentPolicy.IMAGENET,
            "CIFAR10": AutoAugmentPolicy.CIFAR10,
            "SVHN": AutoAugmentPolicy.SVHN,
        }[policy.upper()]
        augmenter = AutoAugment(policy=pol)
        outs = [augmenter(pil) for _ in range(K)]
    else:
        # regression-safety mode
        outs = [photo_autoaugment_pil(pil) for _ in range(K)]

    return [pil_rgb_to_bgr01(o) for o in outs]


def main():
    cfg = load_config()

    PC_USER_NAME = cfg["pc_user_name"]
    WS_NAME = cfg["ws_name"]
    TIME = cfg["time"]

    # 入出力ディレクトリ名（configを基本にしつつ、CLIで上書きも可能にする）
    IN_IMG_DIRNAME = cfg["input_aug_dataset_img"]   # 例: "img"
    IN_VEL_DIRNAME = cfg["input_aug_dataset_vel"]   # 例: "vel"
    OUT_IMG_DIRNAME = cfg.get("output_aug_dataset_img", "autoaugment_img")
    OUT_VEL_DIRNAME = cfg.get("output_aug_dataset_vel", "autoaugment_vel")

    parser = argparse.ArgumentParser(description="AutoAugment 拡張（npy+CSV 版）")
    parser.add_argument("--K", type=int, default=3, help="生成枚数（1エピソードあたり）")
    parser.add_argument("--policy", type=str, default="IMAGENET", choices=["IMAGENET", "CIFAR10", "SVHN"])
    parser.add_argument("--no-geometry", action="store_true", help="幾何変換を避ける（フォトメトリック限定AutoAugment風）")
    parser.add_argument("--out-img-dirname", type=str, default=None, help="出力imgディレクトリ名を上書き")
    parser.add_argument("--out-vel-dirname", type=str, default=None, help="出力velディレクトリ名を上書き")
    parser.add_argument("--seed", type=int, default=None, help="乱数seed（再現性用）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.out_img_dirname:
        OUT_IMG_DIRNAME = args.out_img_dirname
    if args.out_vel_dirname:
        OUT_VEL_DIRNAME = args.out_vel_dirname

    base_dir = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data/{TIME}/dataset"

    in_img_dir = os.path.join(base_dir, IN_IMG_DIRNAME)
    in_vel_csv = os.path.join(base_dir, IN_VEL_DIRNAME, "data.csv")
    out_img_dir = os.path.join(base_dir, OUT_IMG_DIRNAME)
    out_vel_dir = os.path.join(base_dir, OUT_VEL_DIRNAME)
    out_vel_csv = os.path.join(out_vel_dir, "data.csv")

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

        # AutoAugment を K 回（3視点で同じ K 本数を作る）
        variants = {}
        for pos in ("center", "left", "right"):
            variants[pos] = autoaugment_variants_bgr(
                imgs[pos],
                args.K,
                policy=args.policy,
                no_geometry=args.no_geometry,
            )

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

