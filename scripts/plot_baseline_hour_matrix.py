#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(df: pd.DataFrame, title: str, out_png: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "MAE"})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_bar(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_png: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df[xcol].astype(str), df[ycol].to_numpy())
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot heatmaps and summaries for baseline hour matrix evaluation."
    )
    ap.add_argument("--eval-dir", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        raise SystemExit(f"[ERR] eval dir not found: {eval_dir}")

    time_csv = eval_dir / "mae_matrix_by_time.csv"
    hour_csv = eval_dir / "mae_matrix_by_hour.csv"
    same_diff_csv = eval_dir / "summary_same_vs_diff.csv"
    hour_gap_csv = eval_dir / "summary_by_hour_gap.csv"

    df_time = pd.read_csv(time_csv, index_col=0)
    df_hour = pd.read_csv(hour_csv, index_col=0)
    df_same_diff = pd.read_csv(same_diff_csv)
    df_gap = pd.read_csv(hour_gap_csv)

    plot_heatmap(
        df_time,
        "Baseline MAE Matrix by Exact Time",
        eval_dir / "heatmap_mae_by_time.png",
        xlabel="test time",
        ylabel="train time",
    )
    plot_heatmap(
        df_hour,
        "Baseline MAE Matrix by Hour",
        eval_dir / "heatmap_mae_by_hour.png",
        xlabel="test hour",
        ylabel="train hour",
    )

    plot_bar(
        df_same_diff,
        xcol="same_or_diff",
        ycol="mean_mae",
        title="Baseline Mean MAE: same hour vs different hour",
        out_png=eval_dir / "bar_same_vs_diff_mean_mae.png",
    )
    plot_bar(
        df_gap,
        xcol="hour_gap",
        ycol="mean_mae",
        title="Baseline Mean MAE by Hour Gap",
        out_png=eval_dir / "bar_hour_gap_mean_mae.png",
    )

    print(f"[DONE] plots saved under: {eval_dir}")


if __name__ == "__main__":
    main()
