#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_overlay_teacher_preds.py
- deta.csv の教師(center)と、複数の推定列(pred_*)を同一グラフに重ね描き
- CSVごとに1枚の重ね折れ線（teacher + 複数pred）
- ついでに各predの MAE/中央値/p80/p95・符号反転率・しきい値不一致率も表示

使い方（例）:
python3 plot_overlay_teacher_preds.py \
  --csv1 /home/shin/challenge_ws/src/nav_cloning/data/20250903_12:21:08/dataset/vel/deta.csv \
  --preds1 pred_m12_base,pred_m12_aug \
  --csv2 /home/shin/challenge_ws/src/nav_cloning/data/20250902_17:25:30/dataset/vel/deta.csv \
  --preds2 pred_m17_base,pred_m17_aug \
  --title1 "12:21:08" \
  --title2 "17:25:30"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_df(csv_path, preds, teacher="center", xcol="episode"):
    df = pd.read_csv(csv_path)
    need = [xcol, teacher] + preds
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に {c} 列がありません。")
    df = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=xcol)
    return df

def metrics(df, pred_col, teacher="center", thresholds=(0.1, 0.2)):
    y = df[teacher]
    yhat = df[pred_col]
    e = (yhat - y).abs()
    q50, q80, q95 = np.percentile(e, [50, 80, 95])
    sign_flip = (np.sign(yhat) != np.sign(y)).mean()
    out = {
        "count": int(e.shape[0]),
        "mae": float(e.mean()),
        "median": float(q50),
        "p80": float(q80),
        "p95": float(q95),
        "sign_flip_rate": float(sign_flip),
    }
    for t in thresholds:
        disagree = (((yhat.abs() <= t) & (y.abs() > t)) | ((y.abs() <= t) & (yhat.abs() > t))).mean()
        out[f"thr_disagree@{t}"] = float(disagree)
    return out

def plot_overlay(df, preds, teacher="center", title="", smooth=0):
    x = df["episode"].to_numpy()
    plt.figure()
    y = df[teacher].to_numpy()
    if smooth > 1:
        y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    plt.plot(x, y, label=f"teacher({teacher})")
    for pc in preds:
        yy = df[pc].to_numpy()
        if smooth > 1:
            yy = pd.Series(yy).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
        plt.plot(x, yy, label=pc)
    plt.xlabel("step (episode)")
    plt.ylabel("yaw rate [rad/s]")
    plt.title(f"{title} — overlay")
    plt.legend()
    plt.grid(True, alpha=0.3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--preds1", required=True, help="カンマ区切り（例: pred_m12_base,pred_m12_aug）")
    ap.add_argument("--csv2", default=None)
    ap.add_argument("--preds2", default=None, help="カンマ区切り（例: pred_m17_base,pred_m17_aug）")
    ap.add_argument("--teacher", default="center")
    ap.add_argument("--title1", default="CSV1")
    ap.add_argument("--title2", default="CSV2")
    ap.add_argument("--thresholds", default="0.1,0.2")
    ap.add_argument("--smooth", type=int, default=0, help="移動平均ウィンドウ（0/1で無効）")
    args = ap.parse_args()

    thr = tuple(float(x) for x in args.thresholds.split(",")) if args.thresholds else (0.1, 0.2)

    # CSV1
    preds1 = [s.strip() for s in args.preds1.split(",") if s.strip()]
    df1 = load_df(args.csv1, preds1, teacher=args.teacher)
    print(f"[{args.title1}] teacher={args.teacher}")
    for pc in preds1:
        m = metrics(df1, pc, teacher=args.teacher, thresholds=thr)
        print(f"  {pc}: {m}")
    plot_overlay(df1, preds1, teacher=args.teacher, title=args.title1, smooth=args.smooth)

    # CSV2（任意）
    if args.csv2 and args.preds2:
        preds2 = [s.strip() for s in args.preds2.split(",") if s.strip()]
        df2 = load_df(args.csv2, preds2, teacher=args.teacher)
        print(f"[{args.title2}] teacher={args.teacher}")
        for pc in preds2:
            m = metrics(df2, pc, teacher=args.teacher, thresholds=thr)
            print(f"  {pc}: {m}")
        plot_overlay(df2, preds2, teacher=args.teacher, title=args.title2, smooth=args.smooth)

    plt.show()

if __name__ == "__main__":
    main()

