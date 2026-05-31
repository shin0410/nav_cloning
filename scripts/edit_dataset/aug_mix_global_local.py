
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AugMix batch augmentor (global + local in one run).

What this does
--------------
- For each episode, it can create:
  * K_global images with "global AugMix" (full-image)
  * K_local  images with "local AugMix"  (random soft-rect mask)
- Both sets are saved in the same output image dir, but the episode IDs are
  suffixed as:
    - Global: {episode}_G{idx}
    - Local : {episode}_L{idx}
- A single CSV (out_vel_csv) is written that contains all new rows.

Assumptions
-----------
- Dataset layout and paths are defined in ../../config/config.yaml
- Input images are .npy (BGR, 0..1 or 0..255), and there is a data.csv.
- You already have augmix.augment_and_mix. (This script only orchestrates it.)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Requires your existing implementation
from augmix.augment_and_mix import augment_and_mix, MEAN, STD

# -------------------------
# Config
# -------------------------
def load_config(filename: str = "config.yaml") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------
# Image helpers
# -------------------------
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
    """AugMix returns normalized RGB; convert back to [0,1] RGB."""
    img = img_norm * _std[None, None, :] + _mean[None, None, :]
    return np.clip(img, 0.0, 1.0)

# -------------------------
# Local mask utilities
# -------------------------
def random_soft_rect_mask(
    H: int,
    W: int,
    rng: np.random.RandomState,
    area_min: float = 0.10,
    area_max: float = 0.50,
    aspect_min: float = 0.5,
    aspect_max: float = 2.0,
    feather_px_range: Tuple[int, int] = (3, 9),
) -> np.ndarray:
    """
    Create a soft rectangular mask shaped [H, W, 1], values in [0,1].
    """
    A = H * W
    target_ratio = float(rng.uniform(area_min, area_max))
    target_area = max(1, int(A * target_ratio))

    r = float(rng.uniform(aspect_min, aspect_max))  # h / w
    h = int(np.sqrt(target_area * r))
    w = int(np.sqrt(target_area / r))
    h = max(1, min(H, h))
    w = max(1, min(W, w))

    y0 = int(rng.randint(0, max(1, H - h + 1)))
    x0 = int(rng.randint(0, max(1, W - w + 1)))

    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y0 + h, x0:x0 + w] = 1.0

    fmin, fmax = feather_px_range
    f = int(rng.randint(int(fmin), int(fmax) + 1))
    if f > 0:
        # Efficient box blur (separable) as fallback without cv2
        k = max(1, 2 * f + 1)
        pad = k // 2
        # horizontal
        mh = np.pad(m, ((0,0),(pad,pad)), mode="reflect")
        mh = np.cumsum(mh, axis=1)
        mh = (mh[:, k:] - mh[:, :-k]) / k
        # vertical
        mv = np.pad(mh, ((pad,pad),(0,0)), mode="reflect")
        mv = np.cumsum(mv, axis=0)
        mv = (mv[k:, :] - mv[:-k, :]) / k
        m = mv.astype(np.float32)
        m = np.clip(m, 0.0, 1.0)

    return m[..., None]  # [H, W, 1]

def local_blend(
    src_rgb: np.ndarray,
    aug_rgb: np.ndarray,
    mask_hw1: np.ndarray,
) -> np.ndarray:
    """
    Blend src and aug with mask in [0,1]. Shapes:
      src_rgb, aug_rgb: [H, W, 3], float32 0..1
      mask_hw1: [H, W, 1], float32 0..1
    """
    return mask_hw1 * aug_rgb + (1.0 - mask_hw1) * src_rgb

# -------------------------
# AugMix runners
# -------------------------
def augmix_global_variants_bgr(img_bgr: np.ndarray, K: int) -> List[np.ndarray]:
    """Full-image AugMix, returns K BGR images in [0,1]."""
    variants = []
    img_rgb = bgr_to_rgb(img_bgr)
    for _ in range(K):
        mixed = augment_and_mix(img_rgb.astype(np.float32))  # normalized RGB
        denorm_rgb = denormalize_rgb(mixed)
        out_bgr = rgb_to_bgr(denorm_rgb).astype(np.float32)
        variants.append(out_bgr)
    return variants

def augmix_local_variants_bgr(
    img_bgr: np.ndarray,
    K: int,
    masks: List[np.ndarray],
) -> List[np.ndarray]:
    """Local AugMix using provided masks (each [H,W,1])."""
    variants = []
    src_rgb = bgr_to_rgb(img_bgr)  # [0,1]
    for i in range(K):
        mixed = augment_and_mix(src_rgb.astype(np.float32))   # normalized RGB
        aug_rgb = denormalize_rgb(mixed)                      # back to [0,1]
        out_rgb = local_blend(src_rgb, aug_rgb, masks[i])     # local fusion
        out_bgr = rgb_to_bgr(out_rgb).astype(np.float32)
        variants.append(out_bgr)
    return variants

# -------------------------
# Main
# -------------------------
def main():
    cfg = load_config()

    PC_USER_NAME = cfg.get("pc_user_name")
    WS_NAME      = cfg.get("ws_name")
    TIME         = cfg.get("time")

    # Directories in config
    IN_IMG_DIRNAME  = cfg.get("input_aug_dataset_img", "img")          # e.g., "img"
    IN_VEL_DIRNAME  = cfg.get("input_aug_dataset_vel", "vel")          # e.g., "vel"
    OUT_IMG_DIRNAME = cfg.get("output_aug_dataset_img", "augmix_img")  # e.g., "augmix_img"
    OUT_VEL_DIRNAME = cfg.get("output_aug_dataset_vel", "augmix_vel")  # e.g., "augmix_vel"

    parser = argparse.ArgumentParser(description="AugMix augmenter (global + local in one run).")
    parser.add_argument("--K_global", type=int, default=0, help="Number of GLOBAL AugMix variants per episode")
    parser.add_argument("--K_local",  type=int, default=0, help="Number of LOCAL  AugMix variants per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")

    # Local augmentation options
    parser.add_argument("--local_area_min", type=float, default=0.10, help="Min area ratio for local mask")
    parser.add_argument("--local_area_max", type=float, default=0.50, help="Max area ratio for local mask")
    parser.add_argument("--local_feather_min", type=int, default=3, help="Min feather pixels")
    parser.add_argument("--local_feather_max", type=int, default=9, help="Max feather pixels")

    args = parser.parse_args()

    if args.K_global <= 0 and args.K_local <= 0:
        raise SystemExit("Specify at least one of --K_global or --K_local (>0).")

    base_dir = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data/{TIME}/dataset"

    in_img_dir   = os.path.join(base_dir, IN_IMG_DIRNAME)
    in_vel_csv   = os.path.join(base_dir, IN_VEL_DIRNAME, "data.csv")
    out_img_dir  = os.path.join(base_dir, OUT_IMG_DIRNAME)
    out_vel_dir  = os.path.join(base_dir, OUT_VEL_DIRNAME)
    out_vel_csv  = os.path.join(out_vel_dir, "data.csv")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_vel_dir, exist_ok=True)

    if not os.path.exists(in_vel_csv):
        raise FileNotFoundError(f"CSV not found: {in_vel_csv}")
    df = pd.read_csv(in_vel_csv)

    episodes = list(df["episode"].astype(str).values)
    augmented_records: List[dict] = []

    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random.RandomState()

    missing_episodes = 0

    for _, row in df.iterrows():
        ep = str(row["episode"])

        # Load center/left/right npy as BGR [0,1]
        imgs: Dict[str, np.ndarray] = {}
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

        # ---------------- GLOBAL ----------------
        if args.K_global > 0:
            global_variants: Dict[str, List[np.ndarray]] = {}
            for pos in ("center", "left", "right"):
                global_variants[pos] = augmix_global_variants_bgr(imgs[pos], args.K_global)

            for i in range(args.K_global):
                ep_i = f"{ep}_G{i}"  # <- G suffix
                for pos in ("center", "left", "right"):
                    out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                    np.save(out_path, global_variants[pos][i])

                rec = dict(row)
                rec["episode"] = ep_i
                augmented_records.append(rec)

        # ---------------- LOCAL -----------------
        if args.K_local > 0:
            H, W = imgs["center"].shape[:2]
            masks = [
                random_soft_rect_mask(
                    H, W, rng,
                    area_min=args.local_area_min,
                    area_max=args.local_area_max,
                    feather_px_range=(args.local_feather_min, args.local_feather_max),
                )
                for _ in range(args.K_local)
            ]
            local_variants: Dict[str, List[np.ndarray]] = {}
            for pos in ("center", "left", "right"):
                local_variants[pos] = augmix_local_variants_bgr(imgs[pos], args.K_local, masks=masks)

            for i in range(args.K_local):
                ep_i = f"{ep}_L{i}"  # <- L suffix
                for pos in ("center", "left", "right"):
                    out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                    np.save(out_path, local_variants[pos][i])

                rec = dict(row)
                rec["episode"] = ep_i
                augmented_records.append(rec)

    # Write new CSV (overwrite)
    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)

    print(f"[DONE] episodes in: {len(episodes)}, missing: {missing_episodes}")
    print(f"[DONE] images out : {len(os.listdir(out_img_dir))} files  -> {out_img_dir}")
    print(f"[DONE] labels out : {len(augmented_records)} rows       -> {out_vel_csv}")

if __name__ == "__main__":
    main()
