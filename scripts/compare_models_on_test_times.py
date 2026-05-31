#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


DEFAULT_TEST_TIMES = [
    "20260112_10:05:46",
    "20260112_11:09:30",
    "20260112_12:06:28",
    "20260112_13:09:38",
    "20260112_14:11:47",
    "20260112_15:07:57",
    "20260112_16:19:24",
    "20260112_17:15:12",
    "20260112_18:14:00",
]


class Net(nn.Module):
    # Must match nav_cloning/scripts/learning_surprise.py
    def __init__(self, n_channel: int = 3, n_out: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

        self.flatten = nn.Flatten()
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.flatten,
        )
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

    def forward(self, x):
        return self.fc_layer(self.cnn_layer(x))


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def public_data_dir(cfg: dict, nav_dir: Path) -> Path:
    env_data_dir = os.environ.get("NAV_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser().resolve()

    pc = cfg.get("pc_user_name")
    ws = cfg.get("ws_name")
    if pc and ws:
        return Path(f"/home/{pc}/{ws}/src/nav_cloning/data")
    return nav_dir / "data"


def safe_episode(ep) -> str:
    if isinstance(ep, (int, np.integer)):
        return str(int(ep))
    if isinstance(ep, float):
        if abs(ep - int(ep)) < 1e-9:
            return str(int(ep))
    return re.sub(r"\.0$", "", str(ep))


def parse_k_list(s: str) -> List[int]:
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return sorted(set(out))


def resolve_model_dir(
    data_root: Path,
    target_time: str,
    epoch: str,
    explicit_model_dir: str = None,
) -> Tuple[Path, str]:
    if explicit_model_dir:
        p = Path(explicit_model_dir)
        if not p.is_dir():
            raise SystemExit(f"[ERR] --model_dir not found: {p}")
        return p, p.name

    model_root = data_root / target_time / "model"
    if not model_root.is_dir():
        raise SystemExit(f"[ERR] model root not found: {model_root}")

    preferred = model_root / epoch
    if preferred.is_dir():
        return preferred, epoch

    # Fallback: choose latest epoch-like dir that has .pt files.
    cands = []
    for d in model_root.iterdir():
        if not d.is_dir():
            continue
        if not any(d.glob("*.pt")):
            continue
        try:
            key = int(d.name)
        except Exception:
            key = -1
        cands.append((key, d.stat().st_mtime, d))

    if not cands:
        raise SystemExit(
            f"[ERR] no model checkpoint directory under: {model_root}\n"
            f"expected epoch dir: {preferred}"
        )

    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = cands[0][2]
    return chosen, chosen.name


def find_latest(model_dir: Path, regex: str) -> Path:
    pat = re.compile(regex)
    cands = [p for p in model_dir.glob("*.pt") if pat.match(p.name)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def discover_models(model_dir: Path, ks: List[int], allow_missing: bool) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    missing: List[str] = []

    baseline = find_latest(model_dir, r"^model_gpu_baseline(?:_.*)?\.pt$")
    if baseline is None:
        missing.append("baseline")
    else:
        found["baseline"] = baseline

    for k in ks:
        aug = find_latest(model_dir, rf"^model_gpu_augmix_K{k}(?:_.*)?\.pt$")
        taw = find_latest(model_dir, rf"^model_gpu_trivialaugwide_K{k}(?:_.*)?\.pt$")
        rnd = find_latest(model_dir, rf"^model_gpu_randaugment_K{k}(?:_.*)?\.pt$")

        if aug is None:
            missing.append(f"augmix_K{k}")
        else:
            found[f"augmix_K{k}"] = aug

        if taw is None:
            missing.append(f"TrivialAugmentWide_K{k}")
        else:
            found[f"TrivialAugmentWide_K{k}"] = taw

        if rnd is None:
            missing.append(f"rand_augment_K{k}")
        else:
            found[f"rand_augment_K{k}"] = rnd

    if missing and not allow_missing:
        raise SystemExit(
            "[ERR] required models are missing under "
            f"{model_dir}\n  missing: {', '.join(missing)}\n"
            "Use --allow-missing-models to skip missing entries."
        )
    if missing:
        print(f"[WARN] skip missing models: {', '.join(missing)}")
    if not found:
        raise SystemExit(f"[ERR] no models discovered under: {model_dir}")
    return found


def prep_image(path: Path) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"invalid image ndim={arr.ndim}: {path}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    sd = torch.load(model_path, map_location=device)
    if not isinstance(sd, dict):
        raise ValueError(f"unsupported checkpoint format: {model_path}")
    in_ch = 3
    if "conv1.weight" in sd and hasattr(sd["conv1.weight"], "shape"):
        in_ch = int(sd["conv1.weight"].shape[1])
    model = Net(n_channel=in_ch, n_out=1).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def calc_metrics(teacher: np.ndarray, pred: np.ndarray, thr1: float, thr2: float) -> dict:
    abs_err = np.abs(pred - teacher)
    resid = pred - teacher
    sign_flip = (pred * teacher) < 0
    disagree1 = (np.abs(pred) >= thr1) ^ (np.abs(teacher) >= thr1)
    disagree2 = (np.abs(pred) >= thr2) ^ (np.abs(teacher) >= thr2)
    return {
        "count": int(len(pred)),
        "mae": float(abs_err.mean()),
        "mse": float(np.mean(resid ** 2)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "median": float(np.median(abs_err)),
        "p80": float(np.quantile(abs_err, 0.80)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "sign_flip_rate": float(sign_flip.mean()),
        f"thr_disagree@{thr1}": float(disagree1.mean()),
        f"thr_disagree@{thr2}": float(disagree2.mean()),
    }


def infer_one_time(
    model: nn.Module,
    data_root: Path,
    test_time: str,
    test_img_dir: str,
    test_vel_dir: str,
    view: str,
    batch_size: int,
    device: torch.device,
    thr1: float,
    thr2: float,
    test_first_n: int = None,
) -> Tuple[pd.DataFrame, dict]:
    img_dir = data_root / test_time / "dataset" / test_img_dir
    csv_path = data_root / test_time / "dataset" / test_vel_dir / "data.csv"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing img dir: {img_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing csv: {csv_path}")

    df = pd.read_csv(csv_path)
    original_total_rows = len(df)
    if test_first_n is not None:
        df = df.head(test_first_n).copy()
    eval_total_rows = len(df)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found: {csv_path}")
    if view not in df.columns:
        raise KeyError(f"'{view}' column not found: {csv_path}")

    samples = []
    missing_images = 0
    invalid_teacher = 0

    for _, row in df.iterrows():
        ep = safe_episode(row["episode"])
        try:
            teacher = float(row[view])
        except Exception:
            invalid_teacher += 1
            continue
        if not np.isfinite(teacher):
            invalid_teacher += 1
            continue

        img_path = img_dir / f"{ep}_{view}.npy"
        if not img_path.is_file():
            missing_images += 1
            continue

        samples.append((ep, teacher, img_path))

    if not samples:
        raise RuntimeError(f"no valid samples for test_time={test_time}")

    preds: List[float] = []
    episodes: List[str] = []
    teachers: List[float] = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            xs = []
            for ep, teacher, path in chunk:
                xs.append(prep_image(path))
                episodes.append(ep)
                teachers.append(float(teacher))
            x = torch.stack(xs, dim=0).to(device, dtype=torch.float32)
            y = model(x).squeeze(1).detach().cpu().numpy().astype(np.float32)
            preds.extend(y.tolist())

    teacher_np = np.asarray(teachers, dtype=np.float32)
    pred_np = np.asarray(preds, dtype=np.float32)
    metrics = calc_metrics(teacher_np, pred_np, thr1, thr2)
    metrics["test_time"] = test_time
    metrics["original_total_rows"] = int(original_total_rows)
    metrics["eval_total_rows"] = int(eval_total_rows)
    metrics["test_first_n"] = int(test_first_n) if test_first_n is not None else -1
    metrics["kept_rows"] = int(len(pred_np))
    metrics["missing_images"] = int(missing_images)
    metrics["invalid_teacher"] = int(invalid_teacher)

    abs_err = np.abs(pred_np - teacher_np)
    resid = pred_np - teacher_np
    sign_flip = ((pred_np * teacher_np) < 0).astype(int)
    disagree1 = ((np.abs(pred_np) >= thr1) ^ (np.abs(teacher_np) >= thr1)).astype(int)
    disagree2 = ((np.abs(pred_np) >= thr2) ^ (np.abs(teacher_np) >= thr2)).astype(int)

    detail = pd.DataFrame(
        {
            "test_time": test_time,
            "episode": episodes,
            "teacher": teacher_np,
            "pred": pred_np,
            "abs_error": abs_err,
            "residual": resid,
            "sign_flip": sign_flip,
            f"disagree@{thr1}": disagree1,
            f"disagree@{thr2}": disagree2,
        }
    )
    return detail, metrics


def aggregate_detail_by_model(detail_df: pd.DataFrame, thr1: float, thr2: float) -> pd.DataFrame:
    rows = []
    for model_name, g in detail_df.groupby("model"):
        teacher = g["teacher"].to_numpy(dtype=np.float32)
        pred = g["pred"].to_numpy(dtype=np.float32)
        m = calc_metrics(teacher, pred, thr1, thr2)
        m["model"] = model_name
        m["n_times"] = int(g["test_time"].nunique())
        m["count"] = int(len(g))
        rows.append(m)
    cols = [
        "model",
        "n_times",
        "count",
        "mae",
        "mse",
        "rmse",
        "median",
        "p80",
        "p95",
        "sign_flip_rate",
        f"thr_disagree@{thr1}",
        f"thr_disagree@{thr2}",
    ]
    return pd.DataFrame(rows)[cols].sort_values("model")


def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline + augmix/taw/rand_augment (K list) models on test times and export CSV."
    )
    ap.add_argument("--nav_dir", default=None, help="default: this script's nav_cloning dir")
    ap.add_argument("--config", default=None, help="default: <nav_dir>/config/config.yaml")
    ap.add_argument("--target_time", default="20260112_14:08:03")
    ap.add_argument("--epoch", default=None, help="default: read from config")
    ap.add_argument("--model_dir", default=None, help="optional explicit model directory (overrides --target_time/--epoch)")
    ap.add_argument("--ks", default="3,5,7,9", help="comma separated, e.g. 3,5,7,9")
    ap.add_argument("--test_times", nargs="*", default=DEFAULT_TEST_TIMES)
    ap.add_argument("--test-img-dir", default="img", help="dataset image dir name under each test_time/dataset/")
    ap.add_argument("--test-vel-dir", default="vel", help="dataset label dir name under each test_time/dataset/")
    ap.add_argument("--test-first-n", type=int, default=None, help="optional: evaluate only the first N rows from each test dataset csv")
    ap.add_argument("--view", default="center", choices=["center", "left", "right"])
    ap.add_argument("--thr1", type=float, default=0.1)
    ap.add_argument("--thr2", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default=None, help="default: <data_root>/_compare_models_<target>_<stamp>")
    ap.add_argument("--allow-missing-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="model discovery only (no inference)")
    args = ap.parse_args()

    nav_dir = Path(args.nav_dir) if args.nav_dir else Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else (nav_dir / "config" / "config.yaml")
    cfg = load_yaml(cfg_path)

    data_root = public_data_dir(cfg, nav_dir)
    epoch = str(args.epoch if args.epoch is not None else cfg.get("epoch", "100"))
    model_dir, epoch_used = resolve_model_dir(
        data_root=data_root,
        target_time=args.target_time,
        epoch=epoch,
        explicit_model_dir=args.model_dir,
    )

    ks = parse_k_list(args.ks)
    model_map = discover_models(model_dir, ks, allow_missing=args.allow_missing_models)

    target_safe = args.target_time.replace(":", "-")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (data_root / f"_compare_models_{target_safe}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = [{"model": k, "model_path": str(v)} for k, v in model_map.items()]
    model_df = pd.DataFrame(model_rows).sort_values("model")
    model_df.to_csv(out_dir / "models_used.csv", index=False)
    print("[INFO] models:")
    print(f"[INFO] model_dir: {model_dir} (epoch={epoch_used})")
    for _, r in model_df.iterrows():
        print(f"  {r['model']}: {r['model_path']}")

    if args.dry_run:
        print(f"[DONE] dry-run only. model list: {out_dir / 'models_used.csv'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    summary_rows = []
    detail_frames = []

    for model_name, model_path in model_map.items():
        print(f"[RUN] model={model_name}")
        model = load_model(model_path, device)

        for t in args.test_times:
            try:
                detail, met = infer_one_time(
                    model=model,
                    data_root=data_root,
                    test_time=t,
                    test_img_dir=args.test_img_dir,
                    test_vel_dir=args.test_vel_dir,
                    view=args.view,
                    batch_size=args.batch_size,
                    device=device,
                    thr1=args.thr1,
                    thr2=args.thr2,
                    test_first_n=args.test_first_n,
                )
            except Exception as e:
                print(f"[WARN] skip test_time={t} model={model_name}: {e}")
                continue

            detail["model"] = model_name
            detail["model_path"] = str(model_path)
            detail_frames.append(detail)

            met["model"] = model_name
            met["model_path"] = str(model_path)
            summary_rows.append(met)
            print(
                f"  [OK] test_time={t} count={met['count']} "
                f"mae={met['mae']:.6f} missing={met['missing_images']}"
            )

    if not summary_rows:
        raise SystemExit("[ERR] no evaluation results were produced.")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    col_summary = [
        "test_time",
        "model",
        "count",
        "mae",
        "mse",
        "rmse",
        "median",
        "p80",
        "p95",
        "sign_flip_rate",
        f"thr_disagree@{args.thr1}",
        f"thr_disagree@{args.thr2}",
        "original_total_rows",
        "eval_total_rows",
        "test_first_n",
        "kept_rows",
        "missing_images",
        "invalid_teacher",
        "model_path",
    ]
    summary_df = summary_df[col_summary].sort_values(["test_time", "model"])
    summary_df.to_csv(out_dir / "summary_by_time_model.csv", index=False)

    detail_df = detail_df.sort_values(["test_time", "model", "episode"])
    detail_df.to_csv(out_dir / "detail_per_sample.csv", index=False)

    model_agg = aggregate_detail_by_model(detail_df, args.thr1, args.thr2)
    model_agg = model_agg.merge(model_df, on="model", how="left")
    model_agg.to_csv(out_dir / "summary_by_model.csv", index=False)

    pivot = summary_df.pivot_table(index="test_time", columns="model", values="mae", aggfunc="first")
    pivot.to_csv(out_dir / "mae_pivot.csv")

    print(f"[DONE] summary_by_time_model: {out_dir / 'summary_by_time_model.csv'}")
    print(f"[DONE] summary_by_model     : {out_dir / 'summary_by_model.csv'}")
    print(f"[DONE] detail_per_sample   : {out_dir / 'detail_per_sample.csv'}")
    print(f"[DONE] models_used         : {out_dir / 'models_used.csv'}")
    print(f"[DONE] mae_pivot           : {out_dir / 'mae_pivot.csv'}")


if __name__ == "__main__":
    main()
