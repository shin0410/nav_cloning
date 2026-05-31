#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_wz_by_step.py
- 2つの deta.csv を読み込み、pred列の折れ線グラフを描画
- 図1: CSV1 単独
- 図2: CSV2 単独
- 図3: CSV1/CSV2 を重ね描き（episode を横軸）

使い方（例）:
python3 plot_wz_by_step.py \
  --csv1 /home/shin/challenge_ws/src/nav_cloning/data/20250903_12:21:08/dataset/vel/deta.csv \
  --col1 pred_m12_base \
  --csv2 /home/shin/challenge_ws/src/nav_cloning/data/20250902_17:25:30/dataset/vel/deta.csv \
  --col2 pred_m17_base
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_xy(csv_path: str, ycol: str, xcol: str = "episode"):
    df = pd.read_csv(csv_path)
    if xcol not in df or ycol not in df:
        raise ValueError(f"{csv_path} に必要な列がありません: {xcol}, {ycol}")
    # episode順に並べる & 欠損/inf を除去
    df = df[[xcol, ycol]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=xcol)
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True, help="1つ目の deta.csv")
    ap.add_argument("--col1", required=True, help="1つ目で描く予測列名（例: pred_m12_base）")
    ap.add_argument("--csv2", required=True, help="2つ目の deta.csv")
    ap.add_argument("--col2", required=True, help="2つ目で描く予測列名（例: pred_m17_base）")
    ap.add_argument("--title1", default="CSV1", help="図1タイトル")
    ap.add_argument("--title2", default="CSV2", help="図2タイトル")
    ap.add_argument("--title12", default="Overlay (CSV1 vs CSV2)", help="図3タイトル")
    args = ap.parse_args()

    x1, y1 = load_xy(args.csv1, args.col1)
    x2, y2 = load_xy(args.csv2, args.col2)

    # 図1: CSV1 単独
    plt.figure()
    plt.plot(x1, y1)
    plt.xlabel("step (episode)")
    plt.ylabel("predicted yaw rate [rad/s]")
    plt.title(f"{args.title1}: {args.col1}")
    plt.grid(True, alpha=0.3)

    # 図2: CSV2 単独
    plt.figure()
    plt.plot(x2, y2)
    plt.xlabel("step (episode)")
    plt.ylabel("predicted yaw rate [rad/s]")
    plt.title(f"{args.title2}: {args.col2}")
    plt.grid(True, alpha=0.3)

    # 図3: 重ね描き（episode を横軸）
    plt.figure()
    plt.plot(x1, y1, label=f"{args.title1}:{args.col1}")
    plt.plot(x2, y2, label=f"{args.title2}:{args.col2}")
    plt.xlabel("step (episode)")
    plt.ylabel("predicted yaw rate [rad/s]")
    plt.title(args.title12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    main()

