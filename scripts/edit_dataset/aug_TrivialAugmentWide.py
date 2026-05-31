#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
import pandas as pd
import yaml
import random
import hashlib

from typing import Optional
from PIL import Image
import torch
import torchvision.transforms as TV
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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

# --------------------------------------------------------------------

# ---- TrivialAugmentWide から “幾何変換” を全部 제거した版 --------------
# 幾何変換 = Rotate / ShearX/Y / TranslateX/Y を抽選対象から外す
DISALLOW_GEOM = {"Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"}

class TrivialAugmentWideNoGeom(TV.TrivialAugmentWide):
    def __init__(self, *args, allowed_ops=None, max_severity_ratio=None, max_severity_level=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_ops = set(allowed_ops) if allowed_ops else None
        self.max_severity_ratio = max_severity_ratio
        self.max_severity_level = max_severity_level

    def _augmentation_space(self, num_bins):
        space = super()._augmentation_space(num_bins)
        for k in DISALLOW_GEOM:
            space.pop(k, None)
        if self.allowed_ops is not None:
            space = {k: v for k, v in space.items() if k in self.allowed_ops}
        if not space:
            raise ValueError("No operations available after filtering.")
        if self.max_severity_ratio is None and self.max_severity_level is None:
            return space

        max_idx = num_bins - 1
        if self.max_severity_ratio is not None:
            max_idx = min(max_idx, int(np.floor((num_bins - 1) * float(self.max_severity_ratio))))
        if self.max_severity_level is not None:
            max_idx = min(max_idx, int(self.max_severity_level))
        max_idx = max(0, min(num_bins - 1, max_idx))

        capped = {}
        for k, (magnitudes, signed) in space.items():
            if getattr(magnitudes, "ndim", 0) > 0 and len(magnitudes) > 1:
                capped[k] = (magnitudes[: max_idx + 1], signed)
            else:
                capped[k] = (magnitudes, signed)
        return capped

def get_available_ops(num_magnitude_bins: int):
    t = TrivialAugmentWideNoGeom(
        num_magnitude_bins=num_magnitude_bins,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    return sorted(t._augmentation_space(num_magnitude_bins).keys())

def make_trivial_transform(num_magnitude_bins: int, allowed_ops=None, max_severity_ratio=None, max_severity_level=None):
    return TrivialAugmentWideNoGeom(
        num_magnitude_bins=num_magnitude_bins,
        allowed_ops=allowed_ops,
        max_severity_ratio=max_severity_ratio,
        max_severity_level=max_severity_level,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,  # 幾何変換を消してるので基本使われないが、念のため残す
    )
# --------------------------------------------------------------------

def _stable_seed_from_episode(ep: str) -> int:
    # Pythonのhash()は実行ごとに変わることがあるので、安定なseedを作る
    h = hashlib.md5(str(ep).encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32bit相当

def trivialaug_variants_bgr(
    img_bgr: np.ndarray,
    K: int,
    trivial_transform,
    *,
    same_across_views_seed: Optional[int] = None,
):
    """
    BGR(0..1) -> TrivialAugmentWide(NoGeom) を K 回 -> BGR(0..1) を返す
    same_across_views_seed を使うと、center/left/right で同一拡張にできる
    """
    variants = []

    # PILはRGB uint8前提にするのが安全
    img_rgb_u8 = (bgr_to_rgb(img_bgr) * 255.0).round().clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_rgb_u8, mode="RGB")

    for i in range(K):
        if same_across_views_seed is not None:
            seed = same_across_views_seed + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        out_pil = trivial_transform(pil)  # 毎回ランダムに1操作+強度（ただし幾何変換は除外）
        out_rgb = np.asarray(out_pil).astype(np.float32) / 255.0
        out_bgr = rgb_to_bgr(out_rgb).astype(np.float32)
        variants.append(out_bgr)

    return variants

def main():
    cfg = load_config()

    PC_USER_NAME = cfg["pc_user_name"]
    WS_NAME      = cfg["ws_name"]
    TIME         = cfg["time"]

    IN_IMG_DIRNAME  = cfg["input_aug_dataset_img"]   # 例: "img"
    IN_VEL_DIRNAME  = cfg["input_aug_dataset_vel"]   # 例: "vel"
    OUT_IMG_DIRNAME = cfg["output_aug_dataset_img"]  # 例: "trivialaug_img"
    OUT_VEL_DIRNAME = cfg["output_aug_dataset_vel"]  # 例: "trivialaug_vel"

    parser = argparse.ArgumentParser(description="TrivialAugmentWide(NoGeom) 拡張（npy+CSV 版）")
    parser.add_argument("--K", type=int, default=3, help="生成枚数（1エピソードあたり）")
    parser.add_argument("--same3view", action="store_true", help="center/left/rightを同一拡張にする")
    parser.add_argument("--num-magnitude-bins", type=int, default=31, help="TrivialAugmentWide の強度ビン数")
    parser.add_argument("--max-severity-ratio", type=float, default=1.0, help="使う severity の上限比率 (0.0-1.0, default: 1.0)")
    parser.add_argument("--max-severity-level", type=int, default=None, help="使う severity level の最大インデックス (0-based, overrides part of full range)")
    parser.add_argument("--allowed-ops", type=str, default="", help="使う処理名をカンマ区切りで指定（例: Color,Contrast）")
    parser.add_argument("--list-ops", action="store_true", help="利用可能な処理名を表示して終了")
    args = parser.parse_args()
    if args.num_magnitude_bins <= 1:
        raise ValueError("--num-magnitude-bins must be >= 2")
    if not (0.0 <= args.max_severity_ratio <= 1.0):
        raise ValueError("--max-severity-ratio must be in [0.0, 1.0]")
    if args.max_severity_level is not None and args.max_severity_level < 0:
        raise ValueError("--max-severity-level must be >= 0")
    available_ops = get_available_ops(args.num_magnitude_bins)
    if args.list_ops:
        print(",".join(available_ops))
        return

    allowed_ops = []
    if args.allowed_ops.strip():
        allowed_ops = [x.strip() for x in args.allowed_ops.split(",") if x.strip()]
        invalid = [x for x in allowed_ops if x not in available_ops]
        if invalid:
            raise ValueError(
                "Invalid --allowed-ops: "
                + ",".join(invalid)
                + " / available: "
                + ",".join(available_ops)
            )

    trivial_transform = make_trivial_transform(
        args.num_magnitude_bins,
        allowed_ops=allowed_ops if allowed_ops else None,
        max_severity_ratio=args.max_severity_ratio,
        max_severity_level=args.max_severity_level,
    )

    effective_max_level = int(np.floor((args.num_magnitude_bins - 1) * args.max_severity_ratio))
    if args.max_severity_level is not None:
        effective_max_level = min(effective_max_level, args.max_severity_level)
    effective_max_level = max(0, min(args.num_magnitude_bins - 1, effective_max_level))

    data_root = os.environ.get(
        "NAV_DATA_DIR",
        f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data",
    )
    base_dir = os.path.join(data_root, TIME, "dataset")

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

        variants = {}
        # same3view のときだけ、episodeごとの安定seedを作る
        base_seed = _stable_seed_from_episode(ep) if args.same3view else None

        for pos in ("center", "left", "right"):
            variants[pos] = trivialaug_variants_bgr(
                imgs[pos], args.K, trivial_transform, same_across_views_seed=base_seed
            )

        for i in range(args.K):
            ep_i = f"{ep}_{i}"
            for pos in ("center", "left", "right"):
                out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                np.save(out_path, variants[pos][i])

            rec = dict(row)
            rec["episode"] = ep_i
            augmented_records.append(rec)

    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)

    print(f"[DONE] episodes in: {len(episodes)}, missing: {missing_episodes}")
    print(f"[DONE] num_magnitude_bins: {args.num_magnitude_bins}")
    print(f"[DONE] max_severity_ratio: {args.max_severity_ratio}")
    print(f"[DONE] effective_max_severity_level: {effective_max_level}")
    if allowed_ops:
        print(f"[DONE] allowed_ops: {','.join(allowed_ops)}")
    print(f"[DONE] images out : {len(os.listdir(out_img_dir))} files  -> {out_img_dir}")
    print(f"[DONE] labels out : {len(augmented_records)} rows       -> {out_vel_csv}")

if __name__ == "__main__":
    main()
