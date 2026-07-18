#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B案: 照度(lux)計測付き時間帯ベンチマーク。

複数モデルを時間帯の異なるテストデータ群で評価し、
「MAE を照度ギャップの関数として」定量化する。

- lux は data.csv の lux 列を使用。無い場合は center 画像の平均輝度で代用
  (lux_source 列で区別される)。
- models_csv 形式 (train_time は劣化曲線のフィットに必要、省略可):
    model,model_path,train_time
    baseline_scratch,/path/to/model_gpu.pt,20260308_120428
    dinov2_head,/path/to/dinov2__none__rall__s1.pt,20260308_120428
  ※ .meta.json が隣にあれば encoder を自動判別、無ければ既存 scratch Net とみなす。

出力:
  mae_by_time.csv        モデル×テスト時刻の MAE と lux
  leaderboard.csv        モデル別: 最小ギャップMAE / 劣化傾き / 最大ギャップMAE
  degradation_fit.csv    MAE = a + b*|Δlux| の線形フィット (R^2 付き)
  01_mae_vs_lux_gap.png  散布図+フィット直線
  02_mae_by_time.png     時刻別 MAE 折れ線
  benchmark_manifest.json  評価プロトコル (再現用)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import dataset_lux_stats, evaluate_model_on_time, pick_device


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--models_csv", required=True)
    ap.add_argument("--test_times", nargs="+", required=True)
    ap.add_argument("--view", default="center")
    ap.add_argument("--first_n", type=int, default=6100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def linfit(x: np.ndarray, y: np.ndarray):
    """y = a + b*x の最小二乗。返り値 (a, b, r2)"""
    if len(x) < 2 or np.allclose(x.std(), 0):
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    models = pd.read_csv(args.models_csv)
    if not {"model", "model_path"} <= set(models.columns):
        raise SystemExit("[ERR] models_csv には model,model_path 列が必要です")
    has_train_time = "train_time" in models.columns

    # テスト側 lux
    lux_rows = []
    for t in args.test_times:
        st = dataset_lux_stats(args.data_root, t, first_n=args.first_n)
        lux_rows.append(st)
        print("[LUX] %s mean=%.1f (%s)" % (t, st["mean_lux"], st["lux_source"]))
    lux_df = pd.DataFrame(lux_rows).rename(columns={"time_id": "test_time"})

    # 学習側 lux (train_time があるモデルのみ)
    train_lux = {}
    if has_train_time:
        for t in models["train_time"].dropna().unique():
            train_lux[t] = dataset_lux_stats(args.data_root, str(t))["mean_lux"]

    backbone_cache = {}
    rows = []
    for _, m in models.iterrows():
        for t in args.test_times:
            met = evaluate_model_on_time(
                str(m["model_path"]), args.data_root, t, device,
                view=args.view, first_n=args.first_n,
                batch_size=args.batch_size, backbone_cache=backbone_cache)
            rec = {"model": m["model"], "test_time": t,
                   "mae": met["mae"], "rmse": met["rmse"], "count": met["count"]}
            if has_train_time and pd.notna(m.get("train_time")):
                rec["train_time"] = str(m["train_time"])
            rows.append(rec)
            print("[EVAL] %s on %s: mae=%.4f" % (m["model"], t, met["mae"]))

    df = pd.DataFrame(rows).merge(lux_df[["test_time", "mean_lux", "lux_source"]],
                                  on="test_time")
    df = df.rename(columns={"mean_lux": "test_lux"})
    if has_train_time and "train_time" in df.columns:
        df["train_lux"] = df["train_time"].map(train_lux)
        df["lux_gap"] = (df["test_lux"] - df["train_lux"]).abs()
        df["log_lux_ratio"] = np.abs(np.log(
            df["test_lux"].clip(lower=1e-6) / df["train_lux"].clip(lower=1e-6)))
    df.to_csv(out / "mae_by_time.csv", index=False)

    # 劣化曲線フィット + リーダーボード
    fits, board = [], []
    for name, g in df.groupby("model"):
        entry = {"model": name, "n_times": len(g), "mae_mean": float(g["mae"].mean()),
                 "mae_worst": float(g["mae"].max())}
        if "lux_gap" in g.columns and g["lux_gap"].notna().sum() >= 2:
            gg = g.dropna(subset=["lux_gap"])
            a, b, r2 = linfit(gg["lux_gap"].to_numpy(float), gg["mae"].to_numpy(float))
            fits.append({"model": name, "intercept_mae": a,
                         "slope_mae_per_100lux": b * 100.0, "r2": r2})
            i_min = gg["lux_gap"].idxmin()
            i_max = gg["lux_gap"].idxmax()
            entry.update(mae_at_min_gap=float(gg.loc[i_min, "mae"]),
                         mae_at_max_gap=float(gg.loc[i_max, "mae"]),
                         min_gap_lux=float(gg.loc[i_min, "lux_gap"]),
                         max_gap_lux=float(gg.loc[i_max, "lux_gap"]),
                         slope_mae_per_100lux=b * 100.0)
        board.append(entry)
    pd.DataFrame(fits).to_csv(out / "degradation_fit.csv", index=False)
    lb = pd.DataFrame(board).sort_values("mae_mean")
    lb.insert(0, "rank", range(1, len(lb) + 1))
    lb.to_csv(out / "leaderboard.csv", index=False)

    # 01: MAE vs lux gap
    if "lux_gap" in df.columns and df["lux_gap"].notna().any():
        plt.figure(figsize=(8, 5))
        for name, g in df.dropna(subset=["lux_gap"]).groupby("model"):
            g = g.sort_values("lux_gap")
            plt.scatter(g["lux_gap"], g["mae"], label=name, s=35)
            if len(g) >= 2:
                a, b, _ = linfit(g["lux_gap"].to_numpy(float), g["mae"].to_numpy(float))
                xs = np.linspace(0, g["lux_gap"].max(), 20)
                plt.plot(xs, a + b * xs, alpha=0.6)
        plt.xlabel("|train lux - test lux|")
        plt.ylabel("MAE")
        plt.title("Degradation vs illumination gap")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "01_mae_vs_lux_gap.png", dpi=180)
        plt.close()

    # 02: 時刻別 MAE
    plt.figure(figsize=(9, 5))
    pv = df.pivot_table(index="test_time", columns="model", values="mae", aggfunc="first")
    for c in pv.columns:
        plt.plot(pv.index, pv[c], marker="o", label=c)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("MAE")
    plt.title("MAE per test time")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "02_mae_by_time.png", dpi=180)
    plt.close()

    manifest = {
        "protocol": {"view": args.view, "first_n": args.first_n,
                     "metric": "MAE of angular velocity regression"},
        "test_times": args.test_times,
        "test_lux": {r["time_id"]: {"mean_lux": r["mean_lux"], "source": r["lux_source"]}
                     for r in lux_rows},
        "models": models.to_dict(orient="records"),
    }
    with open(out / "benchmark_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] benchmark outputs in %s" % out)
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
