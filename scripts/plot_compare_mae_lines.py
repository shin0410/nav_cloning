#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def sort_models(models):
    def key(name: str):
        if name == "baseline":
            return (0, 0, name)
        m = re.match(r"^(augmix|TrivialAugmentWide|rand_augment)_K(\d+)$", name)
        if m:
            method = m.group(1)
            k = int(m.group(2))
            order = {"augmix": 1, "TrivialAugmentWide": 2, "rand_augment": 3}.get(method, 9)
            return (order, k, name)
        return (99, 0, name)

    return sorted(models, key=key)


def sort_k_cols(cols, method):
    out = []
    for c in cols:
        m = re.match(rf"^{re.escape(method)}_K(\d+)$", c)
        if m:
            out.append((int(m.group(1)), c))
    out.sort(key=lambda x: x[0])
    return [c for _, c in out]


def plot_lines(df_wide, cols, title, out_png):
    if len(cols) == 0:
        return False

    x = df_wide.index.tolist()
    plt.figure(figsize=(14, 6))
    for c in cols:
        y = df_wide[c].to_numpy()
        plt.plot(x, y, marker="o", label=c)

    plt.xlabel("test dataset time (folder name)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
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
    df = df.sort_values(["test_time", "model"])

    wide = df.pivot_table(index="test_time", columns="model", values="mae", aggfunc="first")
    wide = wide.reindex(columns=sort_models(list(wide.columns)))
    wide.to_csv(out_dir / "mae_table_all_models.csv")

    all_cols = list(wide.columns)
    ok_all = plot_lines(
        wide,
        all_cols,
        "MAE over test datasets (all models)",
        out_dir / "01_mae_all_models.png",
    )
    if not ok_all:
        raise SystemExit("[ERR] no model columns available to plot")

    for idx, method in enumerate(["augmix", "TrivialAugmentWide", "rand_augment"], start=2):
        cols = sort_k_cols(list(wide.columns), method)
        tbl_name = f"mae_table_{method}_k_compare.csv"
        png_name = f"{idx:02d}_mae_{method}_k_compare.png"
        if cols:
            wide[cols].to_csv(out_dir / tbl_name)
            plot_lines(
                wide,
                cols,
                f"MAE over test datasets ({method}: K=3/5/7/9)",
                out_dir / png_name,
            )

    print(f"[DONE] saved tables/plots under: {out_dir}")
    print(f"[DONE] table(all): {out_dir / 'mae_table_all_models.csv'}")
    print(f"[DONE] plot(all) : {out_dir / '01_mae_all_models.png'}")


if __name__ == "__main__":
    main()
