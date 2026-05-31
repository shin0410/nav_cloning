#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_episode(ep) -> str:
    return str(ep).split(".")[0]


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found: {csv_path}")
    return df


def image_mean_luma(npy_path: Path) -> float:
    arr = np.load(npy_path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"invalid image shape: {npy_path} {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    b = arr[:, :, 0]
    g = arr[:, :, 1]
    r = arr[:, :, 2]
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return float(np.mean(y))


def compute_split_stats(data_root: Path, time_id: str, img_dir_name: str, vel_dir_name: str) -> dict:
    ds_root = data_root / time_id / "dataset"
    img_root = ds_root / img_dir_name
    csv_path = ds_root / vel_dir_name / "data.csv"
    if not img_root.is_dir():
        raise FileNotFoundError(f"img dir not found: {img_root}")
    df = load_csv(csv_path)

    lux_series = pd.to_numeric(df["lux"], errors="coerce") if "lux" in df.columns else pd.Series(dtype=float)
    brightness = []
    missing_images = 0

    for ep in df["episode"]:
        ep_s = safe_episode(ep)
        img_path = img_root / f"{ep_s}_center.npy"
        if not img_path.is_file():
            missing_images += 1
            continue
        brightness.append(image_mean_luma(img_path))

    brightness_arr = np.asarray(brightness, dtype=float)
    lux_arr = lux_series.dropna().to_numpy(dtype=float) if len(lux_series) else np.asarray([], dtype=float)

    return {
        "time": time_id,
        "rows": int(len(df)),
        "lux_count": int(np.isfinite(lux_arr).sum()),
        "mean_lux": float(np.nanmean(lux_arr)) if lux_arr.size else float("nan"),
        "median_lux": float(np.nanmedian(lux_arr)) if lux_arr.size else float("nan"),
        "std_lux": float(np.nanstd(lux_arr)) if lux_arr.size else float("nan"),
        "brightness_count": int(np.isfinite(brightness_arr).sum()),
        "mean_brightness": float(np.nanmean(brightness_arr)) if brightness_arr.size else float("nan"),
        "median_brightness": float(np.nanmedian(brightness_arr)) if brightness_arr.size else float("nan"),
        "std_brightness": float(np.nanstd(brightness_arr)) if brightness_arr.size else float("nan"),
        "missing_center_images": int(missing_images),
        "img_dir": str(img_root),
        "csv_path": str(csv_path),
    }


def ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
        return float("nan")
    return float(b / a)


def qcut_or_cut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    valid = clean.dropna()
    if valid.nunique() <= 1:
        return pd.Series(["all"] * len(clean), index=series.index)
    try:
        bins = pd.qcut(clean, q=min(q, valid.nunique()), duplicates="drop")
    except Exception:
        bins = pd.cut(clean, bins=min(q, valid.nunique()))
    return bins.astype(str).fillna(f"{prefix}_nan")


def scatter_with_reg(x, y, xlabel: str, ylabel: str, title: str, out_png: Path) -> None:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=35, alpha=0.8)
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 100)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, color="tab:red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def bar_plot(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[xcol].astype(str), df[ycol].to_numpy())
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze relation between baseline MAE and train/test lux or image brightness gaps."
    )
    ap.add_argument("--summary-csv", required=True, help="summary_by_train_test.csv from baseline hour matrix eval")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--train-img-dir", default="train4000_img")
    ap.add_argument("--train-vel-dir", default="train4000_vel")
    ap.add_argument("--test-img-dir", default="test_rest_img")
    ap.add_argument("--test-vel-dir", default="test_rest_vel")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    req = {"train_time", "test_time", "mae", "same_hour", "hour_gap"}
    miss = req - set(summary_df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in summary_csv: {sorted(miss)}")

    train_rows = []
    test_rows = []
    for t in args.times:
        train_rows.append(compute_split_stats(data_root, t, args.train_img_dir, args.train_vel_dir))
        test_rows.append(compute_split_stats(data_root, t, args.test_img_dir, args.test_vel_dir))

    train_stats = pd.DataFrame(train_rows).add_prefix("train_").rename(columns={"train_time": "train_time"})
    test_stats = pd.DataFrame(test_rows).add_prefix("test_").rename(columns={"test_time": "test_time"})
    train_stats.to_csv(out_dir / "train_split_lux_brightness_stats.csv", index=False)
    test_stats.to_csv(out_dir / "test_split_lux_brightness_stats.csv", index=False)

    merged = summary_df.merge(train_stats, on="train_time", how="left").merge(test_stats, on="test_time", how="left")
    merged["diff_mean_lux"] = merged["test_mean_lux"] - merged["train_mean_lux"]
    merged["abs_diff_mean_lux"] = np.abs(merged["diff_mean_lux"])
    merged["ratio_mean_lux"] = [ratio(a, b) for a, b in zip(merged["train_mean_lux"], merged["test_mean_lux"])]
    merged["diff_mean_brightness"] = merged["test_mean_brightness"] - merged["train_mean_brightness"]
    merged["abs_diff_mean_brightness"] = np.abs(merged["diff_mean_brightness"])
    merged["ratio_mean_brightness"] = [ratio(a, b) for a, b in zip(merged["train_mean_brightness"], merged["test_mean_brightness"])]
    merged.to_csv(out_dir / "mae_with_lux_brightness_gap.csv", index=False)

    same_mean_mae = float(merged.loc[merged["same_hour"] == 1, "mae"].mean())
    tol_rows = []
    for factor in [1.05, 1.10, 1.20, 1.30]:
        thr = same_mean_mae * factor
        sub = merged[merged["mae"] <= thr]
        tol_rows.append(
            {
                "mae_threshold_factor_vs_same_hour_mean": factor,
                "mae_threshold": thr,
                "n_pairs": int(len(sub)),
                "max_abs_diff_mean_lux": float(sub["abs_diff_mean_lux"].max()) if len(sub) else float("nan"),
                "max_abs_diff_mean_brightness": float(sub["abs_diff_mean_brightness"].max()) if len(sub) else float("nan"),
            }
        )
    tol_df = pd.DataFrame(tol_rows)
    tol_df.to_csv(out_dir / "tolerance_candidates.csv", index=False)

    gap_lux = merged.copy()
    gap_lux["lux_gap_bin"] = qcut_or_cut(gap_lux["abs_diff_mean_lux"], q=6, prefix="lux")
    gap_lux_summary = (
        gap_lux.groupby("lux_gap_bin", as_index=False)
        .agg(
            n_pairs=("mae", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_abs_diff_mean_lux=("abs_diff_mean_lux", "mean"),
        )
    )
    gap_lux_summary.to_csv(out_dir / "summary_by_lux_gap_bin.csv", index=False)

    gap_bri = merged.copy()
    gap_bri["brightness_gap_bin"] = qcut_or_cut(gap_bri["abs_diff_mean_brightness"], q=6, prefix="brightness")
    gap_bri_summary = (
        gap_bri.groupby("brightness_gap_bin", as_index=False)
        .agg(
            n_pairs=("mae", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_abs_diff_mean_brightness=("abs_diff_mean_brightness", "mean"),
        )
    )
    gap_bri_summary.to_csv(out_dir / "summary_by_brightness_gap_bin.csv", index=False)

    scatter_with_reg(
        merged["abs_diff_mean_lux"],
        merged["mae"],
        xlabel="abs(train_mean_lux - test_mean_lux)",
        ylabel="MAE",
        title="Baseline MAE vs Lux Gap",
        out_png=out_dir / "scatter_mae_vs_lux_gap.png",
    )
    scatter_with_reg(
        merged["abs_diff_mean_brightness"],
        merged["mae"],
        xlabel="abs(train_mean_brightness - test_mean_brightness)",
        ylabel="MAE",
        title="Baseline MAE vs Brightness Gap",
        out_png=out_dir / "scatter_mae_vs_brightness_gap.png",
    )
    scatter_with_reg(
        merged["test_mean_lux"],
        merged["mae"],
        xlabel="test_mean_lux",
        ylabel="MAE",
        title="Baseline MAE vs Test Lux",
        out_png=out_dir / "scatter_mae_vs_test_mean_lux.png",
    )
    scatter_with_reg(
        merged["test_mean_brightness"],
        merged["mae"],
        xlabel="test_mean_brightness",
        ylabel="MAE",
        title="Baseline MAE vs Test Brightness",
        out_png=out_dir / "scatter_mae_vs_test_mean_brightness.png",
    )

    bar_plot(
        gap_lux_summary,
        xcol="lux_gap_bin",
        ycol="mean_mae",
        title="Mean MAE by Lux Gap Bin",
        out_png=out_dir / "bar_mean_mae_by_lux_gap_bin.png",
    )
    bar_plot(
        gap_bri_summary,
        xcol="brightness_gap_bin",
        ycol="mean_mae",
        title="Mean MAE by Brightness Gap Bin",
        out_png=out_dir / "bar_mean_mae_by_brightness_gap_bin.png",
    )

    print(f"[DONE] train stats           : {out_dir / 'train_split_lux_brightness_stats.csv'}")
    print(f"[DONE] test stats            : {out_dir / 'test_split_lux_brightness_stats.csv'}")
    print(f"[DONE] merged gap summary    : {out_dir / 'mae_with_lux_brightness_gap.csv'}")
    print(f"[DONE] tolerance candidates  : {out_dir / 'tolerance_candidates.csv'}")
    print(f"[DONE] lux gap bins          : {out_dir / 'summary_by_lux_gap_bin.csv'}")
    print(f"[DONE] brightness gap bins   : {out_dir / 'summary_by_brightness_gap_bin.csv'}")


if __name__ == "__main__":
    main()
