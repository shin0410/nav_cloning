#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PAT = re.compile(r"^(default|surprise)_(clean|augmix_k\d+|taw_k\d+_m\d+|randaugment_k\d+_n\d+_m\d+_p\w+)$")


def parse_model_name(name: str):
    m = PAT.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def grouped_bar(df: pd.DataFrame, out_png: Path):
    if df.empty:
        return
    augs = df["aug_condition"].tolist()
    default_vals = df["default"].tolist() if "default" in df.columns else [float("nan")] * len(augs)
    surprise_vals = df["surprise"].tolist() if "surprise" in df.columns else [float("nan")] * len(augs)

    x = list(range(len(augs)))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], default_vals, width=width, label="default", color="#4c78a8")
    plt.bar([i + width / 2 for i in x], surprise_vals, width=width, label="surprise", color="#f58518")
    plt.ylabel("mean MAE")
    plt.title("Mean MAE by augmentation and training method")
    plt.xticks(x, augs, rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def single_bar(df: pd.DataFrame, xcol: str, ycol: str, title: str, ylabel: str, out_png: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    colors = ["#4c78a8" if v >= 0 else "#e45756" for v in df[ycol]]
    plt.bar(df[xcol], df[ycol], color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Summarize 8-way default/surprise x augmentation comparison.")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    req = {"test_time", "model", "mae"}
    miss = req - set(df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in summary_csv: {sorted(miss)}")

    parsed = df["model"].astype(str).apply(parse_model_name)
    df["training_method"] = [x[0] for x in parsed]
    df["aug_condition"] = [x[1] for x in parsed]
    df_known = df.dropna(subset=["training_method", "aug_condition"]).copy()
    if df_known.empty:
        raise SystemExit("[ERR] no models matched expected 8-way naming pattern")

    mean_mae = (
        df_known.groupby(["aug_condition", "training_method"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae"})
    )
    mean_pivot = mean_mae.pivot(index="aug_condition", columns="training_method", values="mean_mae")
    mean_pivot.to_csv(out_dir / "matrix_mean_mae.csv")

    mean_pivot_reset = mean_pivot.reset_index()
    grouped_bar(mean_pivot_reset, out_dir / "11_mean_mae_default_vs_surprise.png")

    clean_ref = (
        df_known[df_known["aug_condition"] == "clean"][["test_time", "training_method", "mae"]]
        .rename(columns={"mae": "clean_mae_same_method"})
    )
    effect_vs_clean_by_time = df_known.merge(clean_ref, on=["test_time", "training_method"], how="left")
    effect_vs_clean_by_time["improve_mae_vs_clean"] = (
        effect_vs_clean_by_time["clean_mae_same_method"] - effect_vs_clean_by_time["mae"]
    )
    effect_vs_clean_by_time["improve_pct_vs_clean"] = (
        effect_vs_clean_by_time["improve_mae_vs_clean"] / effect_vs_clean_by_time["clean_mae_same_method"] * 100.0
    )
    effect_vs_clean_by_time.loc[
        effect_vs_clean_by_time["clean_mae_same_method"] == 0.0, "improve_pct_vs_clean"
    ] = 0.0
    effect_vs_clean_by_time = effect_vs_clean_by_time.sort_values(["training_method", "test_time", "aug_condition"])
    effect_vs_clean_by_time.to_csv(out_dir / "effect_vs_clean_within_method_by_time.csv", index=False)

    effect_vs_clean_by_model = (
        effect_vs_clean_by_time.groupby(["training_method", "aug_condition"], as_index=False)
        .agg(
            n_times=("test_time", "nunique"),
            mean_mae=("mae", "mean"),
            mean_clean_mae_same_method=("clean_mae_same_method", "mean"),
            mean_improve_mae_vs_clean=("improve_mae_vs_clean", "mean"),
            mean_improve_pct_vs_clean=("improve_pct_vs_clean", "mean"),
        )
        .sort_values(["training_method", "mean_improve_mae_vs_clean"], ascending=[True, False])
    )
    effect_vs_clean_by_model.to_csv(out_dir / "effect_vs_clean_within_method_by_model.csv", index=False)

    for method in sorted(effect_vs_clean_by_model["training_method"].unique()):
        sub = effect_vs_clean_by_model[effect_vs_clean_by_model["training_method"] == method].copy()
        sub = sub[sub["aug_condition"] != "clean"]
        single_bar(
            sub,
            xcol="aug_condition",
            ycol="mean_improve_mae_vs_clean",
            title=f"Augmentation effect vs clean within {method}",
            ylabel="mean MAE improvement vs clean (+ better)",
            out_png=out_dir / f"12_effect_vs_clean_{method}.png",
        )

    default_df = (
        df_known[df_known["training_method"] == "default"][["test_time", "aug_condition", "mae"]]
        .rename(columns={"mae": "default_mae"})
    )
    surprise_df = (
        df_known[df_known["training_method"] == "surprise"][["test_time", "aug_condition", "mae"]]
        .rename(columns={"mae": "surprise_mae"})
    )
    method_effect_by_time = default_df.merge(surprise_df, on=["test_time", "aug_condition"], how="inner")
    method_effect_by_time["surprise_better_by_mae"] = (
        method_effect_by_time["default_mae"] - method_effect_by_time["surprise_mae"]
    )
    method_effect_by_time["surprise_better_pct"] = (
        method_effect_by_time["surprise_better_by_mae"] / method_effect_by_time["default_mae"] * 100.0
    )
    method_effect_by_time.loc[method_effect_by_time["default_mae"] == 0.0, "surprise_better_pct"] = 0.0
    method_effect_by_time = method_effect_by_time.sort_values(["test_time", "aug_condition"])
    method_effect_by_time.to_csv(out_dir / "effect_surprise_vs_default_by_time.csv", index=False)

    method_effect_by_aug = (
        method_effect_by_time.groupby("aug_condition", as_index=False)
        .agg(
            n_times=("test_time", "nunique"),
            mean_default_mae=("default_mae", "mean"),
            mean_surprise_mae=("surprise_mae", "mean"),
            mean_surprise_better_by_mae=("surprise_better_by_mae", "mean"),
            mean_surprise_better_pct=("surprise_better_pct", "mean"),
        )
        .sort_values("mean_surprise_better_by_mae", ascending=False)
    )
    method_effect_by_aug.to_csv(out_dir / "effect_surprise_vs_default_by_aug.csv", index=False)

    single_bar(
        method_effect_by_aug,
        xcol="aug_condition",
        ycol="mean_surprise_better_by_mae",
        title="Surprise vs default by augmentation",
        ylabel="mean MAE improvement of surprise over default (+ better)",
        out_png=out_dir / "13_effect_surprise_vs_default.png",
    )

    print(f"[DONE] summary dir: {out_dir}")
    print(f"[DONE] matrix_mean_mae.csv: {out_dir / 'matrix_mean_mae.csv'}")
    print(f"[DONE] effect_vs_clean_within_method_by_model.csv: {out_dir / 'effect_vs_clean_within_method_by_model.csv'}")
    print(f"[DONE] effect_surprise_vs_default_by_aug.csv: {out_dir / 'effect_surprise_vs_default_by_aug.csv'}")


if __name__ == "__main__":
    main()
