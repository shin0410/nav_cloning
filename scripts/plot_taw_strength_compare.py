#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def extract_magnitude(model_name: str):
    m = re.match(r"^TrivialAugmentWide_K(\d+)$", str(model_name))
    if not m:
        return None
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(
        description="Plot MAE comparison for TrivialAugmentWide strength sweep."
    )
    ap.add_argument("--summary_csv", required=True, help="summary_by_time_model.csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    req = {"test_time", "model", "mae"}
    miss = req - set(df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in summary_csv: {sorted(miss)}")

    df = df[["test_time", "model", "mae"]].copy()
    df["magnitude_bins"] = df["model"].map(extract_magnitude)
    df = df[df["magnitude_bins"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] no TrivialAugmentWide_K* rows found in summary_csv")

    df["magnitude_bins"] = df["magnitude_bins"].astype(int)
    df["model_label"] = df["magnitude_bins"].map(lambda x: f"M{x}")
    df = df.sort_values(["test_time", "magnitude_bins"], ascending=[True, False])
    df.to_csv(out_dir / "summary_taw_strength_by_time.csv", index=False)

    wide = df.pivot_table(
        index="test_time",
        columns="model_label",
        values="mae",
        aggfunc="first",
    )
    ordered_cols = sorted(list(wide.columns), key=lambda x: int(x[1:]), reverse=True)
    wide = wide.reindex(columns=ordered_cols)
    wide.to_csv(out_dir / "mae_table_taw_strength.csv")

    overall = (
        df.groupby("magnitude_bins")["mae"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("magnitude_bins", ascending=False)
        .rename(
            columns={
                "count": "n_test_times",
                "mean": "mae_mean",
                "median": "mae_median",
                "std": "mae_std",
            }
        )
    )
    overall.to_csv(out_dir / "summary_taw_strength_overall.csv", index=False)

    plt.figure(figsize=(14, 6))
    x = wide.index.tolist()
    for c in wide.columns:
        plt.plot(x, wide[c].to_numpy(), marker="o", label=c)
    plt.xlabel("test dataset time (folder name)")
    plt.ylabel("MAE")
    plt.title("MAE over test datasets (TrivialAugmentWide strength sweep, K=5)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_taw_strength_compare.png", dpi=180)
    plt.close()

    print(f"[DONE] table  : {out_dir / 'mae_table_taw_strength.csv'}")
    print(f"[DONE] plot   : {out_dir / 'mae_taw_strength_compare.png'}")
    print(f"[DONE] overall: {out_dir / 'summary_taw_strength_overall.csv'}")


if __name__ == "__main__":
    main()

