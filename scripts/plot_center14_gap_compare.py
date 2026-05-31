#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = ["augmix", "TrivialAugmentWide", "rand_augment"]
KS = [3, 5, 7, 9]


def plot_lines(df_wide: pd.DataFrame, cols, title: str, out_png: Path) -> None:
    x = df_wide.index.tolist()
    plt.figure(figsize=(14, 6))
    for c in cols:
        if c not in df_wide.columns:
            continue
        plt.plot(x, df_wide[c].to_numpy(), marker="o", label=c)
    plt.xlabel("test dataset time (folder name)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export per-method baseline+K(3/5/7/9) comparison tables/plots for center-14 gap experiments."
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

    df = df.sort_values(["test_time", "model"]).copy()
    wide = df.pivot_table(index="test_time", columns="model", values="mae", aggfunc="first")

    for method in METHODS:
        cols = ["baseline"] + [f"{method}_K{k}" for k in KS]
        cols = [c for c in cols if c in wide.columns]
        if len(cols) <= 1:
            continue

        sub = wide[cols].copy()
        sub.to_csv(out_dir / f"mae_table_center14_gap_{method}.csv")

        filtered = df[df["model"].isin(cols)].copy()
        filtered.to_csv(out_dir / f"summary_by_time_model_center14_gap_{method}.csv", index=False)

        mean_rows = []
        for model_name in cols:
            g = filtered[filtered["model"] == model_name]
            mean_rows.append(
                {
                    "model": model_name,
                    "mean_mae": float(g["mae"].mean()),
                    "median_mae": float(g["mae"].median()),
                    "n_test_times": int(g["test_time"].nunique()),
                }
            )
        mean_df = pd.DataFrame(mean_rows)
        mean_df["sort_k"] = mean_df["model"].map(
            lambda s: 0 if s == "baseline" else int(re.search(r"_K(\d+)$", s).group(1))
        )
        mean_df = mean_df.sort_values(["sort_k", "model"]).drop(columns=["sort_k"])
        mean_df.to_csv(out_dir / f"summary_mean_mae_center14_gap_{method}.csv", index=False)

        plot_lines(
            sub,
            cols,
            f"MAE over test datasets (center-14 replacement: {method})",
            out_dir / f"mae_center14_gap_{method}.png",
        )

    print(f"[DONE] saved method-wise tables/plots under: {out_dir}")


if __name__ == "__main__":
    main()
