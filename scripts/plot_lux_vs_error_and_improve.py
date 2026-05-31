#!/usr/bin/env python3
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXCLUDE = {"augmix_cons"}

def p(x): return Path(os.path.expanduser(x)).resolve()
def safe_episode_str(x):
    s = str(x)
    return s[:-2] if s.endswith(".0") else s

def load_lux_series(data_root: Path, t: str, error_lux: float):
    csv_path = data_root / t / "dataset" / "vel" / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "lux" not in df.columns:
        raise ValueError(f"need episode,lux: {csv_path}")

    ep = df["episode"].apply(safe_episode_str).astype(str)
    lux = pd.to_numeric(df["lux"], errors="coerce").to_numpy(dtype=float)

    sat = np.isfinite(lux) & (lux >= float(error_lux))
    lux2 = lux.copy()
    lux2[sat] = np.nan

    lux_s = pd.Series(lux2, index=ep)
    meta = {
        "mean_lux": float(np.nanmean(lux2)) if np.isfinite(lux2).any() else float("nan"),
        "median_lux": float(np.nanmedian(lux2)) if np.isfinite(lux2).any() else float("nan"),
    }
    return lux_s, meta

def read_with_pred(out_root: Path, t: str, m: str):
    wp = out_root / "eval" / t / m / "with_pred.csv"
    if not wp.is_file():
        raise FileNotFoundError(f"missing: {wp}")
    df = pd.read_csv(wp)
    df["episode"] = df["episode"].apply(safe_episode_str).astype(str)
    df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
    return df

def quantile_edges(x, n_bins):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins+1)))
    if len(edges) < 4:
        return None
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--data_root", default="~/challenge_ws/src/nav_cloning/data")
    ap.add_argument("--thr", type=float, default=0.2, help="bad-step threshold for |error|")
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--error_lux", type=float, default=2_000_000.0)
    ap.add_argument("--times", nargs="*", default=None, help="use only these eval_time(s)")
    ap.add_argument("--out_subdir", default="plots_eval_only_hourly", help="output folder under out_root")
    args = ap.parse_args()

    out_root = p(args.out_root)
    data_root = p(args.data_root)

    fig_dir = out_root / args.out_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_times = sorted([d.name for d in (out_root/"eval").iterdir() if d.is_dir()])
    if args.times and len(args.times) > 0:
        wanted = set(args.times)
        eval_times = [t for t in all_times if t in wanted]
    else:
        eval_times = all_times

    if not eval_times:
        raise SystemExit("[ERR] no eval_times matched --times")

    methods = sorted([d.name for d in (out_root/"eval"/eval_times[0]).iterdir()
                      if d.is_dir() and d.name not in EXCLUDE])

    rows = []
    step_rows = []
    for t in eval_times:
        lux_s, lux_meta = load_lux_series(data_root, t, args.error_lux)
        for m in methods:
            dfp = read_with_pred(out_root, t, m)
            dfp["lux"] = dfp["episode"].map(lux_s)
            ae = dfp["abs_error"].to_numpy(dtype=float)

            rows.append({
                "eval_time": t,
                "method": m,
                "mae": float(np.nanmean(ae)),
                "bad_rate": float(np.nanmean(ae > float(args.thr))),
                "mean_lux": lux_meta["mean_lux"],
                "median_lux": lux_meta["median_lux"],
                "count": int(np.isfinite(ae).sum()),
            })

            s = dfp[["lux","abs_error"]].copy()
            s["eval_time"] = t
            s["method"] = m
            step_rows.append(s)

    met = pd.DataFrame(rows)
    met.to_csv(fig_dir / "metrics_long.csv", index=False)

    # improvement vs baseline
    if "baseline" not in set(met["method"]):
        raise SystemExit("baseline not found (needed for improvement)")
    wide = met.pivot_table(index="eval_time", columns="method", values="mae", aggfunc="first")
    wide.to_csv(fig_dir / "mae_wide.csv")

    base = wide["baseline"]
    delta = pd.DataFrame(index=wide.index)
    for m in wide.columns:
        if m == "baseline": continue
        delta[m] = wide[m] - base
    delta.to_csv(fig_dir / "delta_mae_vs_baseline.csv")

    improve_pct = pd.DataFrame(index=wide.index)
    for m in wide.columns:
        if m == "baseline": continue
        improve_pct[m] = (base - wide[m]) / base * 100.0
    improve_pct.to_csv(fig_dir / "improve_percent_vs_baseline.csv")

    summ = []
    for m in delta.columns:
        x = delta[m].dropna().to_numpy()
        summ.append({
            "method": m,
            "mean_delta_mae": float(np.mean(x)) if len(x) else float("nan"),
            "std_delta_mae": float(np.std(x)) if len(x) else float("nan"),
            "mean_improve_%": float(np.nanmean(improve_pct[m].to_numpy())),
            "n_times": int(np.isfinite(x).sum())
        })
    pd.DataFrame(summ).to_csv(fig_dir / "improvement_summary.csv", index=False)

    # plots
    plt.figure()
    for m in sorted(set(met["method"])):
        dfm = met[met["method"]==m]
        plt.scatter(dfm["mean_lux"], dfm["mae"], label=m, s=25)
    plt.xlabel("mean lux (per test time)")
    plt.ylabel("MAE")
    plt.title("mean lux vs MAE (each point = one test time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "01_mean_lux_vs_mae.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,4))
    for m in sorted(set(met["method"])):
        dfm = met[met["method"]==m].sort_values("eval_time")
        plt.plot(dfm["eval_time"], dfm["mae"], marker="o", label=m)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("eval_time"); plt.ylabel("MAE")
    plt.title("MAE over selected test times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "02_mae_over_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,4))
    for m in sorted(set(met["method"])):
        dfm = met[met["method"]==m].sort_values("eval_time")
        plt.plot(dfm["eval_time"], dfm["bad_rate"], marker="o", label=m)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("eval_time"); plt.ylabel(f"rate(|error|>{args.thr})")
    plt.title("Bad-step rate over selected test times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "03_bad_rate_over_time.png", dpi=200)
    plt.close()

    plt.figure()
    labels = list(delta.columns)
    data = [delta[c].dropna().to_numpy() for c in labels]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("delta MAE (method - baseline)  negative=better")
    plt.title("Delta MAE distribution vs baseline (selected times)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_delta_mae_boxplot.png", dpi=200)
    plt.close()

    plt.figure()
    means = [float(np.mean(d)) if len(d) else np.nan for d in data]
    stds  = [float(np.std(d)) if len(d) else np.nan for d in data]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("mean delta MAE (±std)")
    plt.title("Average improvement vs baseline (selected times)")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_delta_mae_bar.png", dpi=200)
    plt.close()

    steps = pd.concat(step_rows, ignore_index=True)
    steps["lux"] = pd.to_numeric(steps["lux"], errors="coerce")
    steps["abs_error"] = pd.to_numeric(steps["abs_error"], errors="coerce")
    steps = steps[np.isfinite(steps["lux"]) & np.isfinite(steps["abs_error"])]

    edges = quantile_edges(steps["lux"].to_numpy(), args.bins)
    if edges is not None:
        plt.figure()
        for m in sorted(set(steps["method"])):
            dfm = steps[steps["method"]==m]
            lx = dfm["lux"].to_numpy()
            er = dfm["abs_error"].to_numpy()
            ids = np.digitize(lx, edges[1:-1], right=False)
            xs, ys = [], []
            for b in range(len(edges)-1):
                mask = (ids==b)
                if mask.sum() < 5: continue
                xs.append(0.5*(edges[b]+edges[b+1]))
                ys.append(float(np.mean(er[mask])))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("lux (quantile-bin center)")
        plt.ylabel("mean abs_error")
        plt.title("Mean abs_error vs lux (binned, pooled; selected times)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "06_lux_binned_mean_error.png", dpi=200)
        plt.close()

        plt.figure()
        for m in sorted(set(steps["method"])):
            dfm = steps[steps["method"]==m]
            lx = dfm["lux"].to_numpy()
            er = dfm["abs_error"].to_numpy()
            bad = er > float(args.thr)
            ids = np.digitize(lx, edges[1:-1], right=False)
            xs, ys = [], []
            for b in range(len(edges)-1):
                mask = (ids==b)
                if mask.sum() < 5: continue
                xs.append(0.5*(edges[b]+edges[b+1]))
                ys.append(float(np.mean(bad[mask])))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("lux (quantile-bin center)")
        plt.ylabel(f"rate(|error|>{args.thr})")
        plt.title("Bad-step rate vs lux (binned, pooled; selected times)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "07_lux_binned_bad_rate.png", dpi=200)
        plt.close()

    print("[DONE] wrote:", fig_dir)

if __name__ == "__main__":
    main()
