#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def extract_k_alias(model_name: str):
    m = re.match(r"^TrivialAugmentWide_K(\d+)$", str(model_name))
    if not m:
        return None
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(
        description="Plot MAE comparison for TrivialAugmentWide single-op sweep."
    )
    ap.add_argument("--summary_csv", required=True, help="summary_by_time_model.csv")
    ap.add_argument("--op_map_csv", required=True, help="operation mapping CSV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--title",
        default="MAE over test datasets (TrivialAugmentWide single-op, K=5, M=31)",
        help="plot title",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    req = {"test_time", "model", "mae"}
    miss = req - set(summary.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in summary_csv: {sorted(miss)}")

    op_map = pd.read_csv(args.op_map_csv)
    req_map = {"k_alias", "op_name"}
    miss_map = req_map - set(op_map.columns)
    if miss_map:
        raise SystemExit(f"[ERR] missing columns in op_map_csv: {sorted(miss_map)}")

    map_dict = {
        int(r["k_alias"]): str(r["op_name"])
        for _, r in op_map.iterrows()
    }

    df = summary[["test_time", "model", "mae"]].copy()
    df["k_alias"] = df["model"].map(extract_k_alias)
    df = df[df["k_alias"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] no TrivialAugmentWide_K* rows found in summary_csv")
    df["k_alias"] = df["k_alias"].astype(int)
    df["op_name"] = df["k_alias"].map(map_dict)
    df = df[df["op_name"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] model labels do not match op_map_csv k_alias values")

    order_df = op_map[["k_alias", "op_name"]].drop_duplicates().sort_values("k_alias")
    op_order = order_df["op_name"].tolist()

    df = df.sort_values(["test_time", "k_alias"])
    df.to_csv(out_dir / "summary_taw_op_by_time.csv", index=False)

    wide = df.pivot_table(
        index="test_time",
        columns="op_name",
        values="mae",
        aggfunc="first",
    )
    wide = wide.reindex(columns=[c for c in op_order if c in wide.columns])
    wide.to_csv(out_dir / "mae_table_taw_op_compare.csv")

    overall = (
        df.groupby("op_name")["mae"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_test_times",
                "mean": "mae_mean",
                "median": "mae_median",
                "std": "mae_std",
            }
        )
    )
    overall["op_order"] = overall["op_name"].apply(
        lambda x: op_order.index(x) if x in op_order else 10**9
    )
    overall = overall.sort_values("op_order").drop(columns=["op_order"])
    overall.to_csv(out_dir / "summary_taw_op_overall.csv", index=False)

    plt.figure(figsize=(14, 6))
    x = wide.index.tolist()
    for c in wide.columns:
        plt.plot(x, wide[c].to_numpy(), marker="o", label=c)
    plt.xlabel("test dataset time (folder name)")
    plt.ylabel("MAE")
    plt.title(args.title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_taw_op_compare.png", dpi=180)
    plt.close()

    print(f"[DONE] table  : {out_dir / 'mae_table_taw_op_compare.csv'}")
    print(f"[DONE] plot   : {out_dir / 'mae_taw_op_compare.png'}")
    print(f"[DONE] overall: {out_dir / 'summary_taw_op_overall.csv'}")


if __name__ == "__main__":
    main()
