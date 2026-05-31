#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["baseline", "augmix", "taw"]

def load_one(csv_path: str, pred_col: str, teacher_col: str, xcol: str):
    df = pd.read_csv(csv_path)
    need = [xcol, teacher_col, pred_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に {c} 列がありません")
    df = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=xcol)
    df = df.rename(columns={pred_col: f"pred_{pred_col}"})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help=".../_pipeline_out_XXXX")
    ap.add_argument("--eval_time", required=True, help="20250903_12:10:21 など")
    ap.add_argument("--teacher", default="center")
    ap.add_argument("--xcol", default="episode")
    args = ap.parse_args()

    outdir = os.path.join(args.out_root, "overlay", f"eval_{args.eval_time}")
    os.makedirs(outdir, exist_ok=True)

    # 読む
    dfs = []
    for m in METHODS:
        csvp = os.path.join(args.out_root, m, f"eval_{args.eval_time}", "with_pred.csv")
        if not os.path.exists(csvp):
            raise FileNotFoundError(csvp)
        pred_col = f"pred_{m}"
        d = pd.read_csv(csvp)
        need = [args.xcol, args.teacher, pred_col]
        for c in need:
            if c not in d.columns:
                raise ValueError(f"{csvp} に {c} 列がありません")
        d = d[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
        d = d.sort_values(by=args.xcol)
        d = d.rename(columns={pred_col: m})
        dfs.append(d)

    # episodeで揃える（inner join）
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on=[args.xcol, args.teacher], how="inner")

    x = base[args.xcol].to_numpy()
    y = base[args.teacher].to_numpy()

    # 1) teacher + 3pred（重ね描き）
    plt.figure()
    plt.plot(x, y, label="teacher")
    for m in METHODS:
        plt.plot(x, base[m].to_numpy(), label=m)
    plt.xlabel("step (episode)")
    plt.ylabel("yaw rate [rad/s]")
    plt.title(f"teacher vs preds (overlay) — {args.eval_time}")
    plt.legend()
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_teacher_vs_preds_overlay.png"), dpi=170)
    plt.close()

    # 2) Fig5.9風：|error| over time（3本重ね）
    plt.figure()
    for m in METHODS:
        e = np.abs(base[m].to_numpy() - y)
        plt.plot(x, e, label=f"|{m}-teacher|")
    plt.xlabel("step (episode)")
    plt.ylabel("|pred-teacher| [rad/s]")
    plt.title(f"abs error over time (overlay) — {args.eval_time}")
    plt.legend()
    plt.ylim(0.0, 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_abs_error_overlay.png"), dpi=170)
    plt.close()

    # 3) residual over time（3本重ね）
    plt.figure()
    for m in METHODS:
        r = base[m].to_numpy() - y
        plt.plot(x, r, label=f"{m}-teacher")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step (episode)")
    plt.ylabel("residual [rad/s]")
    plt.title(f"residual over time (overlay) — {args.eval_time}")
    plt.legend()
    plt.ylim(-1.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_residual_overlay.png"), dpi=170)
    plt.close()

    print("[DONE] overlay plots:", outdir)

if __name__ == "__main__":
    main()
