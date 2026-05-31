#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from PIL import Image, ImageOps, ImageEnhance, ImageDraw
except ImportError as e:
    raise ImportError(
        "Pillow が必要です。`pip install pillow` してください。"
    ) from e


# ---------------------- config ----------------------
def load_config(filename: str = "config.yaml") -> Dict[str, Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------- image utils ----------------------
def to_float01_bgr(arr: np.ndarray) -> np.ndarray:
    """BGR画像を float32 [0,1] に揃える。入力が uint8(0..255) でもOK。"""
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
    img_u8 = (np.clip(img_rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def pil_rgb_to_bgr01(pil: Image.Image) -> np.ndarray:
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr_bgr = rgb_to_bgr(arr)
    return np.clip(arr_bgr, 0.0, 1.0).astype(np.float32)


# ---------------------- RandAugment core ----------------------
def _float_parameter(m: int, max_v: float) -> float:
    """m: 0..30 を想定。0..max_v にスケール"""
    m = int(np.clip(m, 0, 30))
    return float(m) * max_v / 30.0


def _int_parameter(m: int, max_v: int) -> int:
    m = int(np.clip(m, 0, 30))
    return int(round(float(m) * max_v / 30.0))


@dataclass
class OpPlan:
    name: str
    params: Tuple[Any, ...]
    do_apply: bool


def _safe_fillcolor(pil: Image.Image, fill: Tuple[int, int, int] = (128, 128, 128)) -> Dict[str, Any]:
    """
    Pillowのバージョン差で rotate/transform の fillcolor が使えない場合があるので吸収。
    """
    # rotate は fillcolor を受けるが古いとダメなことがある
    return {"fillcolor": fill}  # 失敗したら呼び出し側で except する


# ---- photometric ops ----
def op_autocontrast(img: Image.Image, *_):
    return ImageOps.autocontrast(img)


def op_equalize(img: Image.Image, *_):
    return ImageOps.equalize(img)


def op_invert(img: Image.Image, *_):
    return ImageOps.invert(img)


def op_solarize(img: Image.Image, m: int):
    # threshold: 256 - magnitude
    thr = 256 - _int_parameter(m, 256)
    thr = int(np.clip(thr, 0, 256))
    return ImageOps.solarize(img, threshold=thr)


def op_posterize(img: Image.Image, m: int):
    # bits: 8 - (0..4)  -> 8..4
    v = _int_parameter(m, 4)
    bits = int(np.clip(8 - v, 1, 8))
    return ImageOps.posterize(img, bits=bits)


def op_color(img: Image.Image, m: int):
    # 1 +/- 0.9
    v = _float_parameter(m, 0.9)
    # torchvision系は [0.1, 1.9] 付近。ここでは 1+v を基本に、たまに減衰も入れる
    factor = 1.0 + v
    return ImageEnhance.Color(img).enhance(factor)


def op_contrast(img: Image.Image, m: int):
    v = _float_parameter(m, 0.9)
    factor = 1.0 + v
    return ImageEnhance.Contrast(img).enhance(factor)


def op_brightness(img: Image.Image, m: int):
    v = _float_parameter(m, 0.9)
    factor = 1.0 + v
    return ImageEnhance.Brightness(img).enhance(factor)


def op_sharpness(img: Image.Image, m: int):
    v = _float_parameter(m, 0.9)
    factor = 1.0 + v
    return ImageEnhance.Sharpness(img).enhance(factor)


def op_cutout(img: Image.Image, m: int, rng: random.Random):
    # cutoutサイズ：画像短辺の最大50%程度まで（Mで増える）
    w, h = img.size
    max_frac = 0.5
    frac = _float_parameter(m, max_frac)
    cut = int(round(min(w, h) * frac))
    if cut <= 0:
        return img

    x0 = rng.randint(0, max(0, w - 1))
    y0 = rng.randint(0, max(0, h - 1))
    x1 = int(np.clip(x0 + cut, 0, w))
    y1 = int(np.clip(y0 + cut, 0, h))

    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return out


# ---- geometry ops (optional) ----
def op_rotate(img: Image.Image, m: int, rng: random.Random):
    deg = _float_parameter(m, 30.0)  # 0..30deg
    if rng.random() < 0.5:
        deg = -deg
    try:
        return img.rotate(deg, resample=Image.BILINEAR, **_safe_fillcolor(img))
    except TypeError:
        # fillcolor 非対応
        return img.rotate(deg, resample=Image.BILINEAR)


def op_shear_x(img: Image.Image, m: int, rng: random.Random):
    v = _float_parameter(m, 0.3)  # 0..0.3
    if rng.random() < 0.5:
        v = -v
    w, h = img.size
    # affine matrix: (1, v, 0, 0, 1, 0)
    try:
        return img.transform((w, h), Image.AFFINE, (1, v, 0, 0, 1, 0),
                             resample=Image.BILINEAR, **_safe_fillcolor(img))
    except TypeError:
        return img.transform((w, h), Image.AFFINE, (1, v, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def op_shear_y(img: Image.Image, m: int, rng: random.Random):
    v = _float_parameter(m, 0.3)
    if rng.random() < 0.5:
        v = -v
    w, h = img.size
    # affine matrix: (1, 0, 0, v, 1, 0)
    try:
        return img.transform((w, h), Image.AFFINE, (1, 0, 0, v, 1, 0),
                             resample=Image.BILINEAR, **_safe_fillcolor(img))
    except TypeError:
        return img.transform((w, h), Image.AFFINE, (1, 0, 0, v, 1, 0),
                             resample=Image.BILINEAR)


def op_translate_x(img: Image.Image, m: int, rng: random.Random):
    w, h = img.size
    # 最大10%程度（Mで増える）
    max_frac = 0.10
    dx = int(round(_float_parameter(m, max_frac) * w))
    if rng.random() < 0.5:
        dx = -dx
    try:
        return img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, 0),
                             resample=Image.BILINEAR, **_safe_fillcolor(img))
    except TypeError:
        return img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, 0),
                             resample=Image.BILINEAR)


def op_translate_y(img: Image.Image, m: int, rng: random.Random):
    w, h = img.size
    max_frac = 0.10
    dy = int(round(_float_parameter(m, max_frac) * h))
    if rng.random() < 0.5:
        dy = -dy
    try:
        return img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, dy),
                             resample=Image.BILINEAR, **_safe_fillcolor(img))
    except TypeError:
        return img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, dy),
                             resample=Image.BILINEAR)


def build_op_pool(include_geometry: bool, allowed_ops: List[str] = None) -> Dict[str, Callable]:
    # photometric-only（デフォルト）
    ops: Dict[str, Callable] = {
        "AutoContrast": op_autocontrast,
        "Equalize": op_equalize,
        "Invert": op_invert,
        "Solarize": op_solarize,
        "Posterize": op_posterize,
        "Color": op_color,
        "Contrast": op_contrast,
        "Brightness": op_brightness,
        "Sharpness": op_sharpness,
        "Cutout": op_cutout,
    }
    if include_geometry:
        ops.update({
            "Rotate": op_rotate,
            "ShearX": op_shear_x,
            "ShearY": op_shear_y,
            "TranslateX": op_translate_x,
            "TranslateY": op_translate_y,
        })
    if allowed_ops:
        ops = {name: fn for name, fn in ops.items() if name in set(allowed_ops)}
        if not ops:
            raise ValueError("No operations available after --allowed-ops filtering.")
    return ops


def parse_allowed_ops(raw: str, include_geometry: bool) -> List[str]:
    if not raw.strip():
        return []
    available = build_op_pool(include_geometry)
    allowed: List[str] = []
    seen = set()
    for token in raw.split(","):
        op = token.strip()
        if not op or op in seen:
            continue
        if op not in available:
            raise ValueError(
                f"Invalid --allowed-ops value: {op} / available: {','.join(sorted(available))}"
            )
        seen.add(op)
        allowed.append(op)
    return allowed


def sample_randaugment_plan(
    N: int,
    M: int,
    p: float,
    include_geometry: bool,
    rng: random.Random,
    allowed_ops: List[str] = None,
) -> List[OpPlan]:
    """
    RandAugment: ops集合から N 個ランダムに選び、各opは確率 p で適用。
    3視点を同期したいので「計画」を先に作って使い回す。
    """
    N = int(max(0, N))
    M = int(np.clip(M, 0, 30))
    p = float(np.clip(p, 0.0, 1.0))

    pool = build_op_pool(include_geometry, allowed_ops)
    names = list(pool.keys())
    if len(names) == 0 or N == 0:
        return []

    plan: List[OpPlan] = []
    for _ in range(N):
        name = rng.choice(names)
        do_apply = (rng.random() < p)

        # opごとに必要なパラメータを固定しておく（視点同期のため）
        if name in ("Solarize", "Posterize", "Color", "Contrast", "Brightness", "Sharpness"):
            params = (M,)
        elif name in ("Cutout",):
            # Cutoutは位置も固定したいので、rngを渡して同じrngを使って適用する方式にする
            # ここではパラメータとして M だけ持つ（位置は apply 時に rng から出るが、
            # 同じ順で同じrngを使えば同期できる）
            params = (M,)
        elif name in ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"):
            params = (M,)
        else:
            params = tuple()

        plan.append(OpPlan(name=name, params=params, do_apply=do_apply))

    return plan


def apply_plan(
    img: Image.Image,
    plan: List[OpPlan],
    include_geometry: bool,
    rng: random.Random,
    allowed_ops: List[str] = None,
) -> Image.Image:
    pool = build_op_pool(include_geometry, allowed_ops)

    out = img
    for step in plan:
        if not step.do_apply:
            continue
        fn = pool[step.name]

        # opによって rng が必要
        if step.name in ("Cutout", "Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"):
            out = fn(out, *step.params, rng)  # type: ignore
        else:
            out = fn(out, *step.params)       # type: ignore

    return out


# ---------------------- main ----------------------
def main():
    cfg = load_config()

    PC_USER_NAME = cfg["pc_user_name"]
    WS_NAME = cfg["ws_name"]
    TIME = cfg["time"]

    IN_IMG_DIRNAME = cfg["input_aug_dataset_img"]    # 例: "img"
    IN_VEL_DIRNAME = cfg["input_aug_dataset_vel"]    # 例: "vel"
    OUT_IMG_DIRNAME = cfg["output_aug_dataset_img"]  # 例: "randaug_img" などにしてもOK
    OUT_VEL_DIRNAME = cfg["output_aug_dataset_vel"]  # 例: "randaug_vel"

    parser = argparse.ArgumentParser(description="RandAugment 拡張（npy+CSV 版）")
    parser.add_argument("--K", type=int, default=3, help="生成枚数（1エピソードあたり）")
    parser.add_argument("--N", type=int, default=2, help="RandAugment: 適用するop数")
    parser.add_argument("--M", type=int, default=9, help="RandAugment: magnitude (0..30)")
    parser.add_argument("--p", type=float, default=0.5, help="各opを適用する確率 (0..1)")
    parser.add_argument("--seed", type=int, default=None, help="再現性用seed（未指定ならランダム）")
    parser.add_argument("--sync_views", action="store_true",
                        help="3視点(center/left/right)に同じ変換を適用（推奨）")
    parser.add_argument("--include_geometry", action="store_true",
                        help="Rotate/Shear/Translate も混ぜる（※回帰ラベルずれ注意）")
    parser.add_argument("--allowed-ops", type=str, default="",
                        help="使う処理名をカンマ区切りで指定（例: Equalize,Brightness,AutoContrast）")
    parser.add_argument("--list-ops", action="store_true", help="利用可能な処理名を表示して終了")
    args = parser.parse_args()

    if args.list_ops:
        print(",".join(sorted(build_op_pool(args.include_geometry))))
        return

    allowed_ops = parse_allowed_ops(args.allowed_ops, args.include_geometry)

    data_root = os.environ.get(
        "NAV_DATA_DIR",
        f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data",
    )
    base_dir = os.path.join(data_root, TIME, "dataset")

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
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found in {in_vel_csv}")

    episodes = df["episode"].unique().tolist()

    augmented_records: List[Dict[str, Any]] = []
    missing_episodes = 0

    for ep in episodes:
        # 1 episode 1 row 前提（元コード踏襲）
        row = df[df["episode"] == ep].iloc[0].to_dict()

        # 3視点すべて揃っているか確認
        imgs_bgr01: Dict[str, np.ndarray] = {}
        ok = True
        for pos in ("center", "left", "right"):
            fpath = os.path.join(in_img_dir, f"{ep}_{pos}.npy")
            if not os.path.exists(fpath):
                print(f"[WARN] missing image: {fpath}")
                ok = False
                break
            try:
                imgs_bgr01[pos] = to_float01_bgr(np.load(fpath, mmap_mode="r"))
            except Exception as e:
                print(f"[ERR ] failed to load {fpath}: {e}")
                ok = False
                break

        if not ok:
            missing_episodes += 1
            continue

        # PIL化（RGB）
        imgs_pil: Dict[str, Image.Image] = {
            pos: bgr01_to_pil_rgb(imgs_bgr01[pos]) for pos in ("center", "left", "right")
        }

        # K回生成
        for i in range(int(max(0, args.K))):
            ep_i = f"{ep}_{i}"

            # 1サンプルごとにrngを作る（順序が変わっても再現しやすい）
            if args.seed is None:
                rng = random.Random()
            else:
                # 32bitに落として安定化
                key = (args.seed, ep, i)
                seed_i = (hash(key) & 0xFFFFFFFF)
                rng = random.Random(seed_i)

            # plan を先に作って、sync_viewsなら3視点で共通適用
            plan = sample_randaugment_plan(
                N=args.N,
                M=args.M,
                p=args.p,
                include_geometry=args.include_geometry,
                rng=rng,
                allowed_ops=allowed_ops,
            )

            # sync_views の場合、Cutout等も揃えるため「同じ順で同じrng」を使って適用する
            if args.sync_views:
                # ここで rng を使い切ると順序で差が出るので、視点ごとに「同じseedのrng」を作り直す
                # ただし plan 生成にも rng を使ったので、同じ seed_i から派生seedで固定する
                if args.seed is None:
                    # sync_views だけど seed 未指定 → 一つの乱数列を使い回す（順序固定で同期）
                    apply_rng = rng
                    outs_pil = {pos: apply_plan(imgs_pil[pos], plan, args.include_geometry, apply_rng, allowed_ops)
                                for pos in ("center", "left", "right")}
                else:
                    # 固定seedなら視点ごとに同じ乱数列を用意して完全同期
                    seed_apply = (hash(("apply", args.seed, ep, i)) & 0xFFFFFFFF)
                    outs_pil = {}
                    for pos in ("center", "left", "right"):
                        apply_rng = random.Random(seed_apply)
                        outs_pil[pos] = apply_plan(imgs_pil[pos], plan, args.include_geometry, apply_rng, allowed_ops)
            else:
                # 視点ごとに別変換（非推奨だが選べる）
                outs_pil = {}
                for pos in ("center", "left", "right"):
                    rng_pos = random.Random(rng.randint(0, 2**32 - 1))
                    plan_pos = sample_randaugment_plan(
                        N=args.N, M=args.M, p=args.p,
                        include_geometry=args.include_geometry,
                        rng=rng_pos,
                        allowed_ops=allowed_ops,
                    )
                    outs_pil[pos] = apply_plan(imgs_pil[pos], plan_pos, args.include_geometry, rng_pos, allowed_ops)

            # 書き出し（BGR float01 npy）
            for pos in ("center", "left", "right"):
                out_bgr01 = pil_rgb_to_bgr01(outs_pil[pos])
                out_path = os.path.join(out_img_dir, f"{ep_i}_{pos}.npy")
                np.save(out_path, out_bgr01.astype(np.float32))

            rec = dict(row)
            rec["episode"] = ep_i
            augmented_records.append(rec)

    pd.DataFrame(augmented_records).to_csv(out_vel_csv, index=False)

    print(f"[DONE] episodes in: {len(episodes)}, missing: {missing_episodes}")
    print(f"[DONE] images out : {len(os.listdir(out_img_dir))} files  -> {out_img_dir}")
    print(f"[DONE] labels out : {len(augmented_records)} rows       -> {out_vel_csv}")
    if allowed_ops:
        print(f"[DONE] allowed_ops: {','.join(allowed_ops)}")

    if args.include_geometry:
        print("[NOTE] include_geometry=True です。回帰ラベル（steering等）をそのままコピーしている場合、"
              "Rotate/Translate/Shear でラベルずれが起きる可能性があります。まずは M を小さめにしてください。")


if __name__ == "__main__":
    main()
