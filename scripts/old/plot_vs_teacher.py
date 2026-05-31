#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_vs_teacher.py
- deta.csv の教師(center) と推定列(pred_*)を比較して折れ線グラフを描画
- 1つ目CSVの図: Teacher vs Pred、残差、残差ヒスト
- 2つ目CSVも同様（指定した場合）
- コンソールに MAE/percentile、符号反転率、しきい値不一致率(0.1/0.2)を表示

使い方（あなたの環境の例）:
python3 plot_vs_teacher.py \
  --csv1 /home/shin/challenge_ws/src/nav_cloning/data/20250903_12:21:08/dataset/vel/deta.csv \
  --pred1 pred_m12_base \
  --csv2 /home/shin/challenge_ws/src/nav_cloning/data/20250902_17:25:30/dataset/vel/deta.csv \
  --pred2 pred_m17_base \
  --title1 "12:21:08 (M12)" \
  --title2 "17:25:30 (M17)"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_df(csv_path, pred_col, teacher_col="center", xcol="episode"):
    df = pd.read_csv(csv_path)
    need = [xcol, teacher_col, pred_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に {c} 列がありません。")
    df = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=xcol)
    return df

def metrics(df, pred_col, teacher_col="center", thresholds=(0.1, 0.2)):
    y = df[teacher_col]
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

def one_panel(df, pred_col, teacher_col="center", title=""):
    x = df["episode"].to_numpy()
    y = df[teacher_col].to_numpy()
    yhat = df[pred_col].to_numpy()
    # 図1: Teacher vs Pred
    plt.figure()
    plt.plot(x, y, label="teacher(center)")
    plt.plot(x, yhat, label=pred_col)
    plt.xlabel("step (episode)")
    plt.ylabel("yaw rate [rad/s]")
    plt.title(f"{title} — teacher vs {pred_col}")
    plt.legend()
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.3)
    # 図2: 残差
    plt.figure()
    plt.plot(x, (yhat - y), label=f"residual({pred_col}-center)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step (episode)")
    plt.ylabel("residual [rad/s]")
    plt.title(f"{title} — residual")
    # 固定レンジ: residual を [-1, 1] に固定
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    # 図3: 残差ヒストグラム
    plt.figure()
    # 固定ビン: [-1, 1] を 50分割（幅 0.04）
    edges = np.linspace(-1, 1, 51)
    plt.hist((yhat - y), bins=edges)
    plt.xlabel("residual [rad/s]")
    plt.ylabel("count")
    plt.title(f"{title} — residual histogram")
    # 固定レンジ: ヒストグラムの縦軸を 0〜600 に固定
    plt.xlim(-1, 1)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--pred1", required=True)
    ap.add_argument("--csv2", default=None)
    ap.add_argument("--pred2", default=None)
    ap.add_argument("--teacher", default="center")
    ap.add_argument("--title1", default="CSV1")
    ap.add_argument("--title2", default="CSV2")
    ap.add_argument("--thresholds", default="0.1,0.2")
    args = ap.parse_args()

    thr = tuple(float(x) for x in args.thresholds.split(",")) if args.thresholds else (0.1, 0.2)

    # CSV1
    df1 = load_df(args.csv1, args.pred1, teacher_col=args.teacher)
    m1 = metrics(df1, args.pred1, teacher_col=args.teacher, thresholds=thr)
    print(f"[{args.title1}] {args.pred1} vs {args.teacher}: {m1}")
    one_panel(df1, args.pred1, teacher_col=args.teacher, title=args.title1)

    # CSV2（任意）
    if args.csv2 and args.pred2:
        df2 = load_df(args.csv2, args.pred2, teacher_col=args.teacher)
        m2 = metrics(df2, args.pred2, teacher_col=args.teacher, thresholds=thr)
        print(f"[{args.title2}] {args.pred2} vs {args.teacher}: {m2}")
        one_panel(df2, args.pred2, teacher_col=args.teacher, title=args.title2)

    plt.show()

if __name__ == "__main__":
    main()

