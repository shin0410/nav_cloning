#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_cols(p: Path):
    df = pd.read_csv(p)
    if "teacher" not in df.columns or "pred" not in df.columns:
        raise SystemExit(f"with_pred.csv must contain teacher,pred: {p}")
    x = df["idx"].to_numpy() if "idx" in df.columns else np.arange(len(df))
    t = df["teacher"].to_numpy()
    y = df["pred"].to_numpy()
    return x, t, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help=".../eval/<eval_time>")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=["augmix_cons"])
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = [p for p in eval_dir.iterdir() if p.is_dir() and p.name not in set(args.exclude)]
    method_dirs = sorted(method_dirs, key=lambda p: p.name)

    series = []
    for md in method_dirs:
        p = md / "with_pred.csv"
        if p.is_file():
            x, t, y = load_cols(p)
            series.append((md.name, x, t, y))
    if not series:
        raise SystemExit(f"no with_pred.csv found under {eval_dir}")

    teacher = series[0][2]
    x = series[0][1]

    # 1) teacher vs preds overlay
    plt.figure()
    plt.plot(x, teacher, label="teacher")
    for name, x, t, y in series:
        plt.plot(x, y, label=name)
    plt.xlabel("step"); plt.ylabel("wz")
    plt.title("teacher vs preds (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_teacher_vs_preds_overlay.png", dpi=170)
    plt.close()

    # 2) abs error overlay
    plt.figure()
    for name, x, t, y in series:
        ae = np.abs(y - t)
        plt.plot(x, ae, label=name)
    plt.xlabel("step"); plt.ylabel("|pred-teacher|")
    plt.title("abs error (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_abs_error_overlay.png", dpi=170)
    plt.close()

    # 3) residual overlay
    plt.figure()
    for name, x, t, y in series:
        r = y - t
        plt.plot(x, r, label=name)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("step"); plt.ylabel("pred-teacher")
    plt.title("residual (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_residual_overlay.png", dpi=170)
    plt.close()

    print(f"[DONE] overlays -> {out_dir}")

if __name__ == "__main__":
    main()
