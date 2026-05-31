
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AugMix batch augmentor (global + local in one run) — robust episode handling + per-view mask resize

Fixes vs previous:
- All references to argparse 'args' live INSIDE main() (avoids NameError).
- Normalizes CSV 'episode' so 1139.0 → "1139", with fallback loader that tries both names.
- For LOCAL AugMix, masks are generated on center view and resized per view (center/left/right) to avoid shape mismatch.
- No OpenCV required; uses separable box-blur to feather masks.
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional
from PIL import Image  # for mask resize

# add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# your existing AugMix impl
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
    if arr.size > 0 and float(arr.max()) > 1.0:
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
# Episode normalize + loader
# -------------------------
def normalize_episode_id(val) -> str:
    """
    Normalize episode to match filenames:
      1139.0   -> "1139"
      "1139.0" -> "1139"
      0012     -> "12"
      12.5     -> "12.5"
    """
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    s = str(val).strip()
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".")[0]
    return s

def candidate_img_paths(img_dir: str, ep: str, pos: str) -> List[str]:
    cands = [os.path.join(img_dir, f"{ep}_{pos}.npy")]
    if re.fullmatch(r"\d+", ep):
        cands.append(os.path.join(img_dir, f"{ep}.0_{pos}.npy"))
    if re.fullmatch(r"\d+\.0+", ep):
        ep_int = ep.split(".")[0]
        cands.append(os.path.join(img_dir, f"{ep_int}_{pos}.npy"))
    cands.append(os.path.join(img_dir, f"{str(ep)}_{pos}.npy"))
    # dedupe
    uniq = []
    for p in cands:
        if p not in uniq:
            uniq.append(p)
    return uniq

def try_load_img(img_dir: str, ep: str, pos: str) -> Optional[np.ndarray]:
    for p in candidate_img_paths(img_dir, ep, pos):
        if os.path.exists(p):
            try:
                return np.load(p, mmap_mode="r")
            except Exception:
                pass
    return None

# -------------------------
# Local mask utils
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
        # separable box blur (approx Gaussian)
        k = max(1, 2 * f + 1)
        pad = k // 2
        mh = np.pad(m, ((0,0),(pad,pad)), mode="reflect")
        mh = np.cumsum(mh, axis=1)
        mh = (mh[:, k:] - mh[:, :-k]) / k
        mv = np.pad(mh, ((pad,pad),(0,0)), mode="reflect")
        mv = np.cumsum(mv, axis=0)
        mv = (mv[k:, :] - mv[:-k, :]) / k
        m = mv.astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
    return m[..., None]  # [H,W,1]

def resize_mask_hw1(mask_hw1: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    if mask_hw1.shape[0] == target_H and mask_hw1.shape[1] == target_W:
        return mask_hw1
    m = (mask_hw1[..., 0] * 255.0).astype(np.uint8)
    im = Image.fromarray(m, mode="L")
    im_r = im.resize((target_W, target_H), resample=Image.BILINEAR)
    mr = np.asarray(im_r).astype(np.float32) / 255.0
    return mr[..., None]

def local_blend(src_rgb: np.ndarray, aug_rgb: np.ndarray, mask_hw1: np.ndarray) -> np.ndarray:
    return mask_hw1 * aug_rgb + (1.0 - mask_hw1) * src_rgb

# -------------------------
# AugMix runners
# -------------------------
def augmix_global_variants_bgr(img_bgr: np.ndarray, K: int) -> List[np.ndarray]:
    variants = []
    img_rgb = bgr_to_rgb(img_bgr)
    for _ in range(K):
        mixed = augment_and_mix(img_rgb.astype(np.float32))  # normalized RGB
        denorm_rgb = denormalize_rgb(mixed)
        out_bgr = rgb_to_bgr(denorm_rgb).astype(np.float32)
        variants.append(out_bgr)
    return variants

def augmix_local_variants_bgr(img_bgr: np.ndarray, K: int, masks: List[np.ndarray]) -> List[np.ndarray]:
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

    IN_IMG_DIRNAME  = cfg.get("input_aug_dataset_img", "img")
    IN_VEL_DIRNAME  = cfg.get("input_aug_dataset_vel", "vel")
    OUT_IMG_DIRNAME = cfg.get("output_aug_dataset_img", "augmix_img")
    OUT_VEL_DIRNAME = cfg.get("output_aug_dataset_vel", "augmix_vel")

    parser = argparse.ArgumentParser(description="AugMix augmenter (global + local) with robust episode handling.")
    parser.add_argument("--K_global", type=int, default=0, help="Number of GLOBAL AugMix variants per episode")
    parser.add_argument("--K_local",  type=int, default=0, help="Number of LOCAL  AugMix variants per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
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

    if "episode" not in df.columns:
        raise KeyError("'episode' column not found in CSV")
    df["episode"] = df["episode"].apply(normalize_episode_id)
    episodes = list(df["episode"].astype(str).values)

    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random.RandomState()

    augmented_records: List[dict] = []
    missing_episodes = 0
    processed_episodes = 0

    for _, row in df.iterrows():
        ep = str(row["episode"])

        imgs: Dict[str, np.ndarray] = {}
        ok = True
        for pos in ("center", "left", "right"):
            arr = try_load_img(in_img_dir, ep, pos)
            if arr is None:
                print(f"[WARN] missing image (tried variants): {ep}_{pos}.npy / {ep}.0_{pos}.npy / raw")
                ok = False
                break
            try:
                imgs[pos] = to_float01_bgr(arr)
            except Exception as e:
                print(f"[ERR ] failed to convert {ep}_{pos}: {e}")
                ok = False
                break
        if not ok:
            missing_episodes += 1
            continue

        processed_episodes += 1

        # GLOBAL
        if args.K_global > 0:
            global_variants: Dict[str, List[np.ndarray]] = {}
            for pos in ("center", "left", "right"):
                global_variants[pos] = augmix_global_variants_bgr(imgs[pos], args.K_global)
            for i in range(args.K_global):
                ep_i = f"{ep}_G{i}"
                for pos in ("center", "left", "right"):
                    np.save(os.path.join(out_img_dir, f"{ep_i}_{pos}.npy"), global_variants[pos][i])
                rec = dict(row); rec["episode"] = ep_i; augmented_records.append(rec)

        # LOCAL
        if args.K_local > 0:
            Hc, Wc = imgs["center"].shape[:2]
            masks_center = [
                random_soft_rect_mask(
                    Hc, Wc, rng,
                    area_min=args.local_area_min,
                    area_max=args.local_area_max,
                    feather_px_range=(args.local_feather_min, args.local_feather_max),
                )
                for _ in range(args.K_local)
            ]
            local_variants: Dict[str, List[np.ndarray]] = {}
            for pos in ("center", "left", "right"):
                Hp, Wp = imgs[pos].shape[:2]
                masks_pos = [resize_mask_hw1(m, Hp, Wp) for m in masks_center]
                local_variants[pos] = augmix_local_variants_bgr(imgs[pos], args.K_local, masks=masks_pos)
            for i in range(args.K_local):
                ep_i = f"{ep}_L{i}"
                for pos in ("center", "left", "right"):
                    np.save(os.path.join(out_img_dir, f"{ep_i}_{pos}.npy"), local_variants[pos][i])
                rec = dict(row); rec["episode"] = ep_i; augmented_records.append(rec)

    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)

    print(f"[DONE] episodes in : {len(episodes)} (after normalize)")
    print(f"[DONE] processed    : {processed_episodes}")
    print(f"[DONE] missing      : {missing_episodes}")
    print(f"[DONE] images out   : {len(os.listdir(out_img_dir))} files  -> {out_img_dir}")
    print(f"[DONE] labels out   : {len(augmented_records)} rows       -> {out_vel_csv}")

if __name__ == "__main__":
    main()
