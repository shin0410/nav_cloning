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
        description="Plot MAE comparison for TrivialAugmentWide max-severity cap sweep."
    )
    ap.add_argument("--summary_csv", required=True, help="summary_by_time_model.csv")
    ap.add_argument("--cap_map_csv", required=True, help="cap mapping CSV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--title",
        default="MAE over test datasets (TrivialAugmentWide max-severity cap sweep)",
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

    cap_map = pd.read_csv(args.cap_map_csv)
    req_map = {"k_alias", "cap_ratio", "max_severity_level", "label"}
    miss_map = req_map - set(cap_map.columns)
    if miss_map:
        raise SystemExit(f"[ERR] missing columns in cap_map_csv: {sorted(miss_map)}")

    map_dict = {
        int(r["k_alias"]): {
            "label": str(r["label"]),
            "cap_ratio": float(r["cap_ratio"]),
            "max_severity_level": int(r["max_severity_level"]),
        }
        for _, r in cap_map.iterrows()
    }

    df = summary[["test_time", "model", "mae"]].copy()
    df["k_alias"] = df["model"].map(extract_k_alias)
    df = df[df["k_alias"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] no TrivialAugmentWide_K* rows found in summary_csv")
    df["k_alias"] = df["k_alias"].astype(int)
    df["label"] = df["k_alias"].map(lambda x: map_dict.get(x, {}).get("label"))
    df["cap_ratio"] = df["k_alias"].map(lambda x: map_dict.get(x, {}).get("cap_ratio"))
    df["max_severity_level"] = df["k_alias"].map(lambda x: map_dict.get(x, {}).get("max_severity_level"))
    df = df[df["label"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] model labels do not match cap_map_csv k_alias values")

    order_df = (
        cap_map[["k_alias", "label", "cap_ratio", "max_severity_level"]]
        .drop_duplicates()
        .sort_values(["cap_ratio", "max_severity_level"], ascending=[False, False])
    )
    label_order = order_df["label"].tolist()

    df = df.sort_values(["test_time", "cap_ratio", "max_severity_level"], ascending=[True, False, False])
    df.to_csv(out_dir / "summary_taw_cap_by_time.csv", index=False)

    wide = df.pivot_table(
        index="test_time",
        columns="label",
        values="mae",
        aggfunc="first",
    )
    wide = wide.reindex(columns=[c for c in label_order if c in wide.columns])
    wide.to_csv(out_dir / "mae_table_taw_cap_compare.csv")

    overall = (
        df.groupby(["label", "cap_ratio", "max_severity_level"])["mae"]
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
        .sort_values(["cap_ratio", "max_severity_level"], ascending=[False, False])
    )
    overall.to_csv(out_dir / "summary_taw_cap_overall.csv", index=False)

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
    plt.savefig(out_dir / "mae_taw_cap_compare.png", dpi=180)
    plt.close()

    print(f"[DONE] table  : {out_dir / 'mae_table_taw_cap_compare.csv'}")
    print(f"[DONE] plot   : {out_dir / 'mae_taw_cap_compare.png'}")
    print(f"[DONE] overall: {out_dir / 'summary_taw_cap_overall.csv'}")


if __name__ == "__main__":
    main()
