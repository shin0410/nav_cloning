#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def metrics(df, pred, teacher="center", thresholds=(0.1,0.2)):
    y = df[teacher].to_numpy()
    yhat = df[pred].to_numpy()
    e = np.abs(yhat - y)
    q50,q80,q95 = np.percentile(e, [50,80,95])
    sign_flip = (np.sign(yhat) != np.sign(y)).mean()
    out = {
        "count": int(len(df)),
        "mae": float(e.mean()),
        "median": float(q50),
        "p80": float(q80),
        "p95": float(q95),
        "sign_flip_rate": float(sign_flip),
    }
    for t in thresholds:
        disagree = (((np.abs(yhat) <= t) & (np.abs(y) > t)) | ((np.abs(y) <= t) & (np.abs(yhat) > t))).mean()
        out[f"thr_disagree@{t}"] = float(disagree)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--teacher", default="center")
    ap.add_argument("--xcol", default="episode")
    ap.add_argument("--title", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", default="0.1,0.2")
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    thr = tuple(float(x) for x in args.thresholds.split(","))

    df = pd.read_csv(args.csv)
    need = [args.xcol, args.teacher, args.pred]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{args.csv} に {c} 列がありません")
    df = df[need].copy().replace([np.inf,-np.inf], np.nan).dropna()
    df = df.sort_values(by=args.xcol)

    x = df[args.xcol].to_numpy()
    y = df[args.teacher].to_numpy()
    yhat = df[args.pred].to_numpy()
    resid = yhat - y
    abse = np.abs(resid)

    os.makedirs(args.outdir, exist_ok=True)

    # 1) teacher vs pred
    plt.figure()
    plt.plot(x, y, label="teacher")
    plt.plot(x, yhat, label=args.pred)
    plt.xlabel("step (episode)")
    plt.ylabel("yaw rate [rad/s]")
    plt.title(f"{args.title} — teacher vs pred")
    plt.legend()
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "01_teacher_vs_pred.png"), dpi=160)
    plt.close()

    # 2) residual
    plt.figure()
    plt.plot(x, resid)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step (episode)")
    plt.ylabel("residual [rad/s]")
    plt.title(f"{args.title} — residual")
    plt.ylim(-1.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "02_residual.png"), dpi=160)
    plt.close()

    # 3) residual hist
    plt.figure()
    edges = np.linspace(-1, 1, 51)
    plt.hist(resid, bins=edges)
    plt.xlabel("residual [rad/s]")
    plt.ylabel("count")
    plt.title(f"{args.title} — residual hist")
    plt.xlim(-1, 1)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "03_residual_hist.png"), dpi=160)
    plt.close()

    # 4) Fig5.9: abs error over time
    plt.figure()
    plt.plot(x, abse)
    plt.xlabel("step (episode)")
    plt.ylabel("|pred-teacher| [rad/s]")
    plt.title(f"{args.title} — abs error over time (Fig5.9 style)")
    plt.ylim(0.0, 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "04_abs_error_over_time.png"), dpi=160)
    plt.close()

    m = metrics(df, args.pred, teacher=args.teacher, thresholds=thr)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    print("[METRICS]", m)

if __name__ == "__main__":
    main()

