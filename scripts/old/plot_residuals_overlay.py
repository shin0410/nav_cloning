#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_residuals_overlay.py  (plot_vs_teacher.py準拠のカウント)
- teacher(center) 対 Baseline / AugMix の残差を同図に重ねて可視化
- 前処理/カウント方針は plot_vs_teacher.py と同じ：
  * episode/center/pred が有限な行だけに dropna
  * ヒストは bins=50, range 指定なし（= 全サンプルをカウント）
- Ctrl+Cで詰まる場合は --no-show や --save を使ってください
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_df_pvt_style(csv_path, pred_col, teacher_col="center", xcol="episode"):
    """plot_vs_teacher.py と同じ前処理（inf→NaN→dropna, ソート）"""
    df = pd.read_csv(csv_path)
    need = [xcol, teacher_col, pred_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に {c} 列がありません。")
    df = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=xcol)
    return df  # この df は pred ごとに独立（= サンプル数が違ってもOK）

def residual_series(df, pred_col, teacher_col="center"):
    return (df[pred_col].to_numpy() - df[teacher_col].to_numpy()), df["episode"].to_numpy()

def panel_one_csv(csv_path, base_col, aug_col, teacher_col="center",
                  title="", bins=50, smooth=0, save=None, no_show=False):
    # pred ごとに plot_vs_teacher.py と同じ条件で個別に前処理
    df_b = load_df_pvt_style(csv_path, base_col, teacher_col=teacher_col)
    df_a = load_df_pvt_style(csv_path, aug_col,  teacher_col=teacher_col)

    r_base, x_b = residual_series(df_b, base_col, teacher_col)
    r_aug,  x_a = residual_series(df_a, aug_col,  teacher_col)

    # 参考: 指標（端末出力）
    def metrics(r):
        e = np.abs(r)
        q50, q80, q95 = np.percentile(e, [50, 80, 95])
        return dict(count=int(e.size), mae=float(e.mean()),
                    median=float(q50), p80=float(q80), p95=float(q95))
    print(f"[{title}] vs {teacher_col}")
    print(f"  {base_col}: {metrics(r_base)}")
    print(f"  {aug_col} : {metrics(r_aug)}")

    # 残差の折れ線（各シリーズ独立の episode 軸でOK）
    if smooth and smooth > 1:
        r_base = pd.Series(r_base).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
        r_aug  = pd.Series(r_aug ).rolling(smooth, center=True, min_periods=1).mean().to_numpy()

    plt.figure()
    plt.plot(x_b, r_base, label=f"residual({base_col}-center)")
    plt.plot(x_a, r_aug,  label=f"residual({aug_col}-center)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step (episode)")
    plt.ylabel("residual [rad/s]")
    plt.title(f"{title} — residual lines")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 残差ヒスト：bins=50, range指定なし → 各ヒストの総カウントは各シリーズのサンプル数そのもの
    edges = np.histogram_bin_edges(np.concatenate([r_base, r_aug]), bins=bins)
    plt.figure()
    plt.hist(r_base, bins=edges, histtype="step", label=base_col)
    plt.hist(r_aug,  bins=edges, histtype="step", label=aug_col)
    plt.xlabel("residual [rad/s]")
    plt.ylabel("count")
    plt.title(f"{title} — residual histogram (pvt-style counting)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存/表示
    if save:
        outdir = Path(save); outdir.mkdir(parents=True, exist_ok=True)
        t = title.replace(" ", "_").replace("/", "-")
        plt.savefig(outdir / f"{t}_residual_lines.png", dpi=160, bbox_inches="tight")
        plt.savefig(outdir / f"{t}_residual_hist.png",  dpi=160, bbox_inches="tight")
    if not no_show:
        plt.show()
    else:
        plt.close("all")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--base1", required=True)
    ap.add_argument("--aug1",  required=True)
    ap.add_argument("--csv2", default=None)
    ap.add_argument("--base2", default=None)
    ap.add_argument("--aug2",  default=None)
    ap.add_argument("--teacher", default="center")
    ap.add_argument("--title1", default="CSV1")
    ap.add_argument("--title2", default="CSV2")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    panel_one_csv(args.csv1, args.base1, args.aug1, teacher_col=args.teacher,
                  title=args.title1, bins=args.bins, smooth=args.smooth,
                  save=args.save, no_show=args.no_show)

    if args.csv2 and args.base2 and args.aug2:
        panel_one_csv(args.csv2, args.base2, args.aug2, teacher_col=args.teacher,
                      title=args.title2, bins=args.bins, smooth=args.smooth,
                      save=args.save, no_show=args.no_show)

if __name__ == "__main__":
    main()

