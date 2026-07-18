#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A案: results_long.csv からデータ効率曲線・交互作用プロット・必要データ量表を作る。

出力:
  01_data_efficiency_<gap_type>.png  encoder×aug 別の MAE vs データ量 (シードで誤差棒)
  02_aug_effect.png                  拡張の改善量 (none−taw) を encoder×gap_type 別に
  summary_mean.csv                   セル平均のロング表
  data_requirement.csv               目標MAE到達に必要な行数 (線形補間)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"scratch": "#4c78a8", "resnet18": "#72b7b2", "vit_b_16": "#54a24b",
          "dinov2_vits14": "#f58518", "dinov2_vitb14": "#e45756",
          "clip_vitb16": "#b279a2"}
LSTYLE = {"none": "--", "taw3op": "-"}


def rows_to_num(df: pd.DataFrame) -> pd.DataFrame:
    max_named = pd.to_numeric(df.loc[df["rows"] != "all", "rows"], errors="coerce").max()
    all_val = (max_named * 1.5) if np.isfinite(max_named) else 1.0
    df["rows_num"] = df["rows"].apply(
        lambda r: float(all_val) if r == "all" else float(r))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_mae", type=float, default=None,
                    help="必要データ量の目標MAE。省略時は scratch+none 最大データ時の平均MAE")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    df["mae"] = pd.to_numeric(df["mae"])
    df["rows"] = df["rows"].astype(str)
    df = rows_to_num(df)

    # seed 内で test_time を平均してから seed 間の分散を取る
    per_seed = (df.groupby(["encoder", "aug", "rows", "rows_num", "gap_type", "seed"],
                           as_index=False)["mae"].mean())
    mean_tbl = (per_seed.groupby(["encoder", "aug", "rows", "rows_num", "gap_type"],
                                 as_index=False)
                .agg(mean_mae=("mae", "mean"), std_mae=("mae", "std"),
                     n_seeds=("seed", "nunique")))
    mean_tbl.sort_values(["gap_type", "encoder", "aug", "rows_num"]).to_csv(
        out / "summary_mean.csv", index=False)

    # 01: データ効率曲線
    for gap_type, sub in mean_tbl.groupby("gap_type"):
        plt.figure(figsize=(8, 5))
        for (enc, aug), g in sub.groupby(["encoder", "aug"]):
            g = g.sort_values("rows_num")
            plt.errorbar(g["rows_num"], g["mean_mae"], yerr=g["std_mae"].fillna(0),
                         marker="o", capsize=3,
                         color=COLORS.get(enc, "#888888"), linestyle=LSTYLE.get(aug, ":"),
                         label="%s / %s" % (enc, aug))
        plt.xlabel("training rows per dataset")
        plt.ylabel("MAE (mean over test times, err = seed std)")
        plt.title("Data efficiency (%s-condition test)" % gap_type)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / ("01_data_efficiency_%s.png" % gap_type), dpi=180)
        plt.close()

    # 02: 拡張効果 (同一 encoder/rows/seed/gap_type で none - taw3op)
    pv = per_seed.pivot_table(index=["encoder", "rows", "rows_num", "gap_type", "seed"],
                              columns="aug", values="mae").reset_index()
    if {"none", "taw3op"} <= set(pv.columns):
        pv["aug_improve"] = pv["none"] - pv["taw3op"]
        eff = (pv.groupby(["encoder", "gap_type"], as_index=False)
               .agg(mean_improve=("aug_improve", "mean"),
                    std_improve=("aug_improve", "std")))
        eff.to_csv(out / "aug_effect.csv", index=False)
        plt.figure(figsize=(7, 4.5))
        gaps = sorted(eff["gap_type"].unique())
        encs = sorted(eff["encoder"].unique())
        width = 0.8 / max(1, len(gaps))
        x = np.arange(len(encs))
        for j, gt in enumerate(gaps):
            e = eff[eff["gap_type"] == gt].set_index("encoder").reindex(encs)
            plt.bar(x + j * width, e["mean_improve"], width=width,
                    yerr=e["std_improve"].fillna(0), capsize=3, label="test=%s" % gt)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x + width * (len(gaps) - 1) / 2, encs)
        plt.ylabel("MAE improvement by augmentation (+ = helps)")
        plt.title("Augmentation effect vs encoder\n(does pretraining absorb augmentation?)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "02_aug_effect.png", dpi=180)
        plt.close()

    # 必要データ量: 各曲線が目標MAEを下回る最小 rows (線形補間)
    req_rows = []
    for gap_type, sub in mean_tbl.groupby("gap_type"):
        base = sub[(sub["encoder"] == "scratch") & (sub["aug"] == "none")]
        if args.target_mae is not None:
            target = args.target_mae
        elif len(base):
            target = float(base.loc[base["rows_num"].idxmax(), "mean_mae"])
        else:
            target = float(sub["mean_mae"].min())
        for (enc, aug), g in sub.groupby(["encoder", "aug"]):
            g = g.sort_values("rows_num")
            xs, ys = g["rows_num"].to_numpy(), g["mean_mae"].to_numpy()
            need = np.nan
            for i in range(len(xs)):
                if ys[i] <= target:
                    if i == 0:
                        need = xs[0]
                    else:
                        f = (ys[i - 1] - target) / max(1e-12, ys[i - 1] - ys[i])
                        need = xs[i - 1] + f * (xs[i] - xs[i - 1])
                    break
            req_rows.append({"gap_type": gap_type, "encoder": enc, "aug": aug,
                             "target_mae": target, "rows_needed": need,
                             "best_mae": float(ys.min())})
    req = pd.DataFrame(req_rows)
    base_need = req[(req["encoder"] == "scratch") & (req["aug"] == "none")] \
        .set_index("gap_type")["rows_needed"]
    req["data_reduction_vs_scratch"] = req.apply(
        lambda r: base_need.get(r["gap_type"], np.nan) / r["rows_needed"]
        if r["rows_needed"] and np.isfinite(r["rows_needed"]) else np.nan, axis=1)
    req.to_csv(out / "data_requirement.csv", index=False)

    print("[DONE] plots/tables in %s" % out)


if __name__ == "__main__":
    main()
