#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import argparse
import random
import hashlib
import numpy as np
import pandas as pd
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from augmix import augmentations as augmix_augmentations
from augmix.augment_and_mix import augment_and_mix, MEAN, STD


AUGMIX_OPS = {
    "AutoContrast": augmix_augmentations.autocontrast,
    "Brightness": augmix_augmentations.brightness,
    "Color": augmix_augmentations.color,
    "Contrast": augmix_augmentations.contrast,
    "Equalize": augmix_augmentations.equalize,
    "Posterize": augmix_augmentations.posterize,
    "Sharpness": augmix_augmentations.sharpness,
    "Solarize": augmix_augmentations.solarize,
}


def parse_allowed_ops(raw: str):
    if not raw.strip():
        return []
    allowed = []
    seen = set()
    for token in raw.split(","):
        op = token.strip()
        if not op or op in seen:
            continue
        if op not in AUGMIX_OPS:
            raise ValueError(
                f"Invalid --allowed-ops value: {op} / available: {','.join(sorted(AUGMIX_OPS))}"
            )
        seen.add(op)
        allowed.append(op)
    return allowed


def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
_std = np.array(STD, dtype=np.float32)


def denormalize_rgb(img_norm: np.ndarray) -> np.ndarray:
    img = img_norm * _std[None, None, :] + _mean[None, None, :]
    return np.clip(img, 0.0, 1.0)


def stable_seed_u32(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=4).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def set_all_seeds(seed: int):
    seed = int(seed) & 0xFFFFFFFF
    np.random.seed(seed)
    random.seed(seed)


def augmix_once_bgr(img_bgr01: np.ndarray, seed: int) -> np.ndarray:
    set_all_seeds(seed)
    img_rgb = bgr_to_rgb(img_bgr01).astype(np.float32)
    mixed = augment_and_mix(img_rgb)
    denorm = denormalize_rgb(mixed)
    out_bgr = rgb_to_bgr(denorm).astype(np.float32)
    return out_bgr


def main():
    cfg = load_config()

    pc_user_name = cfg["pc_user_name"]
    ws_name = cfg["ws_name"]
    time_id = cfg["time"]

    in_img_dirname = cfg["input_aug_dataset_img"]
    in_vel_dirname = cfg["input_aug_dataset_vel"]
    out_img_dirname = cfg["output_aug_dataset_img"]
    out_vel_dirname = cfg["output_aug_dataset_vel"]

    parser = argparse.ArgumentParser(description="AugMix 拡張（npy+CSV 版）: multi-view整合性あり")
    parser.add_argument("--K", type=int, default=3, help="AugMix 生成枚数（1エピソードあたり）")
    parser.add_argument("--seed", type=int, default=1234, help="全体seed（再現性用）")
    parser.add_argument("--map_csv", type=str, default="augmix_map.csv", help="対応表CSVのファイル名（out_vel_dir配下に出す）")
    parser.add_argument("--allowed-ops", type=str, default="", help="使う処理名をカンマ区切りで指定（例: Equalize,Brightness,AutoContrast）")
    parser.add_argument("--list-ops", action="store_true", help="利用可能な処理名を表示して終了")
    args = parser.parse_args()

    if args.list_ops:
        print(",".join(sorted(AUGMIX_OPS)))
        return

    allowed_ops = parse_allowed_ops(args.allowed_ops)
    if allowed_ops:
        augmix_augmentations.augmentations = [AUGMIX_OPS[op] for op in allowed_ops]

    data_root = os.environ.get(
        "NAV_DATA_DIR",
        f"/home/{pc_user_name}/{ws_name}/src/nav_cloning/data",
    )
    base_dir = os.path.join(data_root, time_id, "dataset")

    in_img_dir = os.path.join(base_dir, in_img_dirname)
    in_vel_csv = os.path.join(base_dir, in_vel_dirname, "data.csv")
    out_img_dir = os.path.join(base_dir, out_img_dirname)
    out_vel_dir = os.path.join(base_dir, out_vel_dirname)
    out_vel_csv = os.path.join(out_vel_dir, "data.csv")
    out_map_csv = os.path.join(out_vel_dir, args.map_csv)

    if not os.path.exists(in_vel_csv):
        raise FileNotFoundError(f"Velocity CSV not found: {in_vel_csv}")
    if not os.path.isdir(in_img_dir):
        raise FileNotFoundError(f"Image dir not found: {in_img_dir}")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_vel_dir, exist_ok=True)

    df = pd.read_csv(in_vel_csv)
    episodes = df["episode"].unique().tolist()

    augmented_records = []
    map_records = []
    missing_episodes = 0
    reused_variants = 0
    generated_variants = 0

    for ep in tqdm(episodes, desc="AugMix episodes"):
        row = df[df["episode"] == ep].iloc[0].to_dict()

        # First inspect which variants already exist so reruns can resume quickly.
        done_variants = {}
        need_any_generation = False
        for i in range(args.K):
            ep_i = f"{ep}_{i}"
            out_paths = [os.path.join(out_img_dir, f"{ep_i}_{pos}.npy") for pos in ("center", "left", "right")]
            done = all(os.path.exists(p) for p in out_paths)
            done_variants[i] = done
            if done:
                reused_variants += 1
            else:
                need_any_generation = True

            seed_i = stable_seed_u32(args.seed, ep, i)
            rec = dict(row)
            rec["episode"] = ep_i
            augmented_records.append(rec)
            map_records.append({
                "base_episode": ep,
                "aug_episode": ep_i,
                "aug_index": i,
                "seed_u32": seed_i,
            })

        if not need_any_generation:
            continue

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

        for i in range(args.K):
            if done_variants[i]:
                continue
            seed_i = stable_seed_u32(args.seed, ep, i)
            ep_i = f"{ep}_{i}"
            for pos in ("center", "left", "right"):
                out_img = augmix_once_bgr(imgs[pos], seed=seed_i)
                out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                np.save(out_path, out_img)
            generated_variants += 1

    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)
    pd.DataFrame(map_records).to_csv(out_map_csv, index=False)

    n_files = len([n for n in os.listdir(out_img_dir) if n.endswith('.npy')])
    print(f"[DONE] episodes in : {len(episodes)}, missing: {missing_episodes}")
    print(f"[DONE] variants    : generated={generated_variants}, reused={reused_variants}")
    print(f"[DONE] images out  : {n_files} files  -> {out_img_dir}")
    print(f"[DONE] labels out  : {len(augmented_records)} rows -> {out_vel_csv}")
    print(f"[DONE] map out     : {len(map_records)} rows -> {out_map_csv}")


if __name__ == "__main__":
    main()
