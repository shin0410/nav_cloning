#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_TEST_TIMES = [
    "20260315_11:06:52",
    "20260316_13:44:20",
    "20260319_17:02:53",
]


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_compare_cmd(args, compare_py: Path, out_dir: Path):
    cmd = [
        sys.executable,
        str(compare_py),
        "--nav_dir",
        str(args.nav_dir),
        "--config",
        str(args.config),
        "--target_time",
        args.target_time,
        "--epoch",
        str(args.epoch),
        "--ks",
        args.ks,
        "--view",
        args.view,
        "--thr1",
        str(args.thr1),
        "--thr2",
        str(args.thr2),
        "--batch_size",
        str(args.batch_size),
        "--out_dir",
        str(out_dir),
        "--test-img-dir",
        args.test_img_dir,
        "--test-vel-dir",
        args.test_vel_dir,
        "--test-first-n",
        str(args.test_first_n),
        "--test_times",
        *args.test_times,
    ]
    if args.model_dir:
        cmd.extend(["--model_dir", str(args.model_dir)])
    if args.allow_missing_models:
        cmd.append("--allow-missing-models")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def compute_effect_tables(summary_csv: Path, out_dir: Path):
    df = pd.read_csv(summary_csv)
    req = {"test_time", "model", "mae"}
    miss = req - set(df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in summary csv: {sorted(miss)}")

    base = df[df["model"] == "baseline"][["test_time", "mae"]].rename(columns={"mae": "baseline_mae"})
    if base.empty:
        raise SystemExit("[ERR] baseline model was not evaluated; cannot compute effect vs baseline")

    merged = df.merge(base, on="test_time", how="left")
    merged["delta_mae"] = merged["mae"] - merged["baseline_mae"]
    merged["improve_mae"] = merged["baseline_mae"] - merged["mae"]
    merged["improve_pct"] = merged["improve_mae"] / merged["baseline_mae"] * 100.0
    merged.loc[merged["baseline_mae"] == 0.0, "improve_pct"] = 0.0
    merged = merged.sort_values(["test_time", "model"])

    cols = [
        "test_time",
        "model",
        "mae",
        "baseline_mae",
        "delta_mae",
        "improve_mae",
        "improve_pct",
        "count",
        "missing_images",
        "model_path",
    ]
    keep_cols = [c for c in cols if c in merged.columns]
    merged[keep_cols].to_csv(out_dir / "effect_vs_baseline_by_time.csv", index=False)

    agg = (
        merged.groupby("model", as_index=False)
        .agg(
            n_times=("test_time", "nunique"),
            mean_mae=("mae", "mean"),
            mean_baseline_mae=("baseline_mae", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_improve_mae=("improve_mae", "mean"),
            mean_improve_pct=("improve_pct", "mean"),
            best_improve_mae=("improve_mae", "max"),
            worst_improve_mae=("improve_mae", "min"),
        )
        .sort_values(["mean_improve_mae", "model"], ascending=[False, True])
    )
    agg.to_csv(out_dir / "effect_vs_baseline_by_model.csv", index=False)

    pivot = merged.pivot_table(index="test_time", columns="model", values="improve_mae", aggfunc="first")
    pivot.to_csv(out_dir / "improve_mae_pivot.csv")

    best = (
        merged.sort_values(["test_time", "mae", "model"])
        .groupby("test_time", as_index=False)
        .first()[["test_time", "model", "mae", "baseline_mae", "improve_mae", "improve_pct"]]
        .rename(columns={"model": "best_model"})
    )
    best.to_csv(out_dir / "best_model_by_test_time.csv", index=False)

    return agg


def plot_effect_bars(effect_by_model_csv: Path, out_png: Path):
    df = pd.read_csv(effect_by_model_csv)
    if df.empty:
        return
    plt.figure(figsize=(12, 5))
    colors = ["#4c78a8" if m != "baseline" else "#999999" for m in df["model"]]
    plt.bar(df["model"], df["mean_improve_mae"], color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("mean MAE improvement vs baseline (+ better)")
    plt.title("Augmentation effect over 3 test datasets (first N rows)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    default_nav_dir = script_dir.parent
    default_config = default_nav_dir / "config" / "config.yaml"

    ap = argparse.ArgumentParser(
        description="Compare baseline vs augmentation models on the 3 fixed test datasets and summarize effect vs baseline."
    )
    ap.add_argument("--nav_dir", type=Path, default=default_nav_dir)
    ap.add_argument("--config", type=Path, default=default_config)
    ap.add_argument("--target_time", default="20260112_14:08:03")
    ap.add_argument("--epoch", default="100")
    ap.add_argument("--model_dir", type=Path, default=None, help="optional explicit model directory")
    ap.add_argument("--ks", default="3,5,7,9")
    ap.add_argument("--test_times", nargs="*", default=DEFAULT_TEST_TIMES)
    ap.add_argument("--test_img_dir", default="img")
    ap.add_argument("--test_vel_dir", default="vel")
    ap.add_argument("--test_first_n", type=int, default=6200)
    ap.add_argument("--view", default="center", choices=["center", "left", "right"])
    ap.add_argument("--thr1", type=float, default=0.1)
    ap.add_argument("--thr2", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--allow-missing-models", action="store_true")
    ap.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    compare_py = script_dir / "compare_models_on_test_times.py"
    plot_py = script_dir / "plot_compare_mae_lines.py"

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_dir:
        model_root_tag = args.model_dir.parent.parent.name if len(args.model_dir.parents) >= 2 else args.model_dir.parent.name
        model_tag = f"{model_root_tag}_{args.model_dir.name}"
    else:
        model_tag = args.target_time.replace(":", "-")
    out_dir = args.out_dir if args.out_dir else (args.nav_dir / "data" / f"_aug_effects_3tests_{model_tag}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_cmd = build_compare_cmd(args, compare_py, out_dir)
    run_cmd(compare_cmd)

    if args.dry_run:
        print(f"[DONE] dry-run only: {out_dir}")
        return

    summary_csv = out_dir / "summary_by_time_model.csv"
    if not summary_csv.is_file():
        raise SystemExit(f"[ERR] summary not found: {summary_csv}")

    run_cmd([
        sys.executable,
        str(plot_py),
        "--summary_csv",
        str(summary_csv),
        "--out_dir",
        str(out_dir / "plots"),
    ])

    compute_effect_tables(summary_csv, out_dir)
    plot_effect_bars(out_dir / "effect_vs_baseline_by_model.csv", out_dir / "plots" / "10_effect_vs_baseline_mean_improve.png")

    print(f"[DONE] out_dir                   : {out_dir}")
    print(f"[DONE] raw summary              : {out_dir / 'summary_by_time_model.csv'}")
    print(f"[DONE] test_first_n             : {args.test_first_n}")
    print(f"[DONE] effect by time           : {out_dir / 'effect_vs_baseline_by_time.csv'}")
    print(f"[DONE] effect by model          : {out_dir / 'effect_vs_baseline_by_model.csv'}")
    print(f"[DONE] best model by test time  : {out_dir / 'best_model_by_test_time.csv'}")
    print(f"[DONE] plots                    : {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
