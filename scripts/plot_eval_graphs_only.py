#!/usr/bin/env python3
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXCLUDE_METHODS = {"augmix_cons"}  # 今後重ねない

def p(x): return Path(os.path.expanduser(x)).resolve()

def list_eval_times(out_root: Path):
    d = out_root / "eval"
    if not d.is_dir():
        raise SystemExit(f"[ERR] eval dir not found: {d}")
    times = sorted([x.name for x in d.iterdir() if x.is_dir()])
    if not times:
        raise SystemExit(f"[ERR] no eval_time dirs under: {d}")
    return times

def list_methods(out_root: Path, eval_time: str):
    d = out_root / "eval" / eval_time
    ms = sorted([x.name for x in d.iterdir() if x.is_dir()])
    ms = [m for m in ms if m not in EXCLUDE_METHODS]
    if not ms:
        raise SystemExit(f"[ERR] no methods under: {d} (after exclude)")
    return ms

def safe_episode_str(x):
    s = str(x)
    return s[:-2] if s.endswith(".0") else s

def load_lux_series(data_root: Path, eval_time: str, error_lux: float):
    # data/<eval_time>/dataset/vel/data.csv を読む
    csv_path = data_root / eval_time / "dataset" / "vel" / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing lux csv: {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError(f"'episode' not found: {csv_path}")
    if "lux" not in df.columns:
        raise ValueError(f"'lux' column not found (need illuminance): {csv_path}")

    lux = pd.to_numeric(df["lux"], errors="coerce").to_numpy(dtype=float)
    sat_mask = np.isfinite(lux) & (lux >= float(error_lux))
    lux2 = lux.copy()
    lux2[sat_mask] = np.nan

    ep = df["episode"].apply(safe_episode_str).astype(str)
    lux_s = pd.Series(lux2, index=ep)

    meta = {
        "mean_lux": float(np.nanmean(lux2)) if np.isfinite(lux2).any() else float("nan"),
        "median_lux": float(np.nanmedian(lux2)) if np.isfinite(lux2).any() else float("nan"),
        "sat_rate": float(sat_mask.mean()) if len(lux2) else float("nan"),
        "n": int(len(lux2)),
        "n_valid": int(np.isfinite(lux2).sum())
    }
    return lux_s, meta

def load_with_pred(out_root: Path, eval_time: str, method: str):
    wp = out_root / "eval" / eval_time / method / "with_pred.csv"
    if not wp.is_file():
        raise FileNotFoundError(f"missing with_pred.csv: {wp}")
    w = pd.read_csv(wp)
    if "episode" not in w.columns or "abs_error" not in w.columns:
        raise ValueError(f"with_pred.csv missing required cols: {wp}")
    w["episode"] = w["episode"].apply(safe_episode_str).astype(str)
    return w

def quantile_bins(x, n_bins=12):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="_pipeline_out_... directory")
    ap.add_argument("--data_root", default="~/challenge_ws/src/nav_cloning/data")
    ap.add_argument("--error_lux", type=float, default=2_000_000.0, help=">= this treated as invalid lux")
    ap.add_argument("--thr", type=float, default=0.2, help="|error| threshold for 'bad step' rate")
    ap.add_argument("--bins", type=int, default=12, help="lux bin count (quantile bins)")
    ap.add_argument("--scatter_step_sample", type=int, default=0, help="0=off, else sample N points per method for lux-vs-error scatter")
    args = ap.parse_args()

    out_root = p(args.out_root)
    data_root = p(args.data_root)
    fig_dir = out_root / "plots_eval_only"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_times = list_eval_times(out_root)
    methods = list_methods(out_root, eval_times[0])

    # ---- build long table (time, method) ----
    rows = []
    step_rows = []  # for lux-binned plots (pooled)

    for t in eval_times:
        lux_s, lux_meta = load_lux_series(data_root, t, args.error_lux)

        for m in methods:
            if m in EXCLUDE_METHODS:
                continue

            w = load_with_pred(out_root, t, m)
            # join lux by episode
            w["lux"] = w["episode"].map(lux_s)
            ae = pd.to_numeric(w["abs_error"], errors="coerce").to_numpy(dtype=float)

            mae = float(np.nanmean(ae))
            bad_rate = float(np.nanmean(ae > float(args.thr)))

            rows.append({
                "eval_time": t,
                "method": m,
                "count": int(len(w)),
                "mae": mae,
                "bad_rate@thr": bad_rate,
                "thr": float(args.thr),
                "mean_lux": lux_meta["mean_lux"],
                "median_lux": lux_meta["median_lux"],
                "sat_rate": lux_meta["sat_rate"],
                "lux_valid_n": lux_meta["n_valid"],
            })

            # pooled step data for lux vs error bin plot
            step = w[["lux", "abs_error"]].copy()
            step["eval_time"] = t
            step["method"] = m
            step_rows.append(step)

    met = pd.DataFrame(rows)
    met.to_csv(fig_dir / "metrics_long_with_lux.csv", index=False)

    steps = pd.concat(step_rows, ignore_index=True)
    # clean
    steps["lux"] = pd.to_numeric(steps["lux"], errors="coerce")
    steps["abs_error"] = pd.to_numeric(steps["abs_error"], errors="coerce")
    steps = steps[np.isfinite(steps["lux"]) & np.isfinite(steps["abs_error"])]

    # ---- improvement table vs baseline ----
    if "baseline" not in set(met["method"]):
        raise SystemExit("[ERR] baseline not found in methods. Need baseline to compute improvements.")

    wide = met.pivot_table(index="eval_time", columns="method", values="mae", aggfunc="first")
    wide.to_csv(fig_dir / "mae_wide.csv")

    base = wide["baseline"]
    imp = pd.DataFrame(index=wide.index)
    for m in wide.columns:
        if m == "baseline": 
            continue
        imp[m] = wide[m] - base  # negative = improved
    imp.to_csv(fig_dir / "delta_mae_vs_baseline.csv")

    imp_pct = pd.DataFrame(index=wide.index)
    for m in wide.columns:
        if m == "baseline":
            continue
        imp_pct[m] = (base - wide[m]) / base * 100.0
    imp_pct.to_csv(fig_dir / "improve_percent_vs_baseline.csv")

    # summary (mean/std)
    summ = []
    for m in imp.columns:
        x = imp[m].dropna().to_numpy()
        summ.append({
            "method": m,
            "mean_delta_mae": float(np.mean(x)) if len(x) else float("nan"),
            "std_delta_mae": float(np.std(x)) if len(x) else float("nan"),
            "mean_improve_%": float(np.nanmean(imp_pct[m].to_numpy())) if m in imp_pct.columns else float("nan"),
            "n_times": int(np.isfinite(x).sum())
        })
    pd.DataFrame(summ).to_csv(fig_dir / "improvement_summary.csv", index=False)

    # -------------------------
    # PLOTS
    # -------------------------

    # 1) mean_lux vs mae scatter (per method)
    plt.figure()
    for m in sorted(set(met["method"])):
        dfm = met[met["method"] == m]
        plt.scatter(dfm["mean_lux"], dfm["mae"], label=m, s=25)
    plt.xlabel("mean lux (per eval_time)")
    plt.ylabel("MAE (|pred-teacher| mean)")
    plt.title("mean lux vs MAE (each point = one test time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "01_mean_lux_vs_mae_scatter.png", dpi=200)
    plt.close()

    # 2) mae over time (x=eval_time index)
    plt.figure(figsize=(12, 4))
    for m in sorted(set(met["method"])):
        dfm = met[met["method"] == m].copy()
        dfm = dfm.sort_values("eval_time")
        plt.plot(dfm["eval_time"], dfm["mae"], marker="o", label=m)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("eval_time")
    plt.ylabel("MAE")
    plt.title("MAE over test times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "02_mae_over_time.png", dpi=200)
    plt.close()

    # 3) bad_rate (|error|>thr) over time
    plt.figure(figsize=(12, 4))
    for m in sorted(set(met["method"])):
        dfm = met[met["method"] == m].copy().sort_values("eval_time")
        plt.plot(dfm["eval_time"], dfm["bad_rate@thr"], marker="o", label=m)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("eval_time")
    plt.ylabel(f"rate(|error|>{args.thr})")
    plt.title("Bad-step rate over test times (proxy of 'cannot drive')")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "03_bad_rate_over_time.png", dpi=200)
    plt.close()

    # 4) delta MAE boxplot (vs baseline)
    plt.figure()
    labels = list(imp.columns)
    data = [imp[c].dropna().to_numpy() for c in labels]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("delta MAE (method - baseline)  (negative = better)")
    plt.title("Delta MAE distribution vs baseline (across test times)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_delta_mae_boxplot.png", dpi=200)
    plt.close()

    # 5) mean delta MAE bar
    plt.figure()
    means = [float(np.mean(d)) if len(d) else np.nan for d in data]
    stds  = [float(np.std(d)) if len(d) else np.nan for d in data]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("mean delta MAE (±std)")
    plt.title("Average improvement vs baseline (delta MAE)")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_delta_mae_bar.png", dpi=200)
    plt.close()

    # 6) lux-bin mean error (pooled over all times) - overlay methods
    edges = quantile_bins(steps["lux"].to_numpy(), n_bins=args.bins)
    if edges is not None:
        plt.figure()
        for m in sorted(set(steps["method"])):
            dfm = steps[steps["method"] == m]
            lx = dfm["lux"].to_numpy()
            er = dfm["abs_error"].to_numpy()
            # bin
            ids = np.digitize(lx, edges[1:-1], right=False)
            xs, ys = [], []
            for b in range(len(edges)-1):
                mask = (ids == b)
                if mask.sum() < 5:
                    continue
                xs.append(0.5*(edges[b] + edges[b+1]))
                ys.append(float(np.mean(er[mask])))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("lux (bin center, quantile bins)")
        plt.ylabel("mean abs_error")
        plt.title("Mean abs_error vs lux (binned, pooled over all test times)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "06_lux_binned_mean_error_overlay.png", dpi=200)
        plt.close()

        # 7) lux-bin bad-rate (pooled) - overlay methods
        plt.figure()
        for m in sorted(set(steps["method"])):
            dfm = steps[steps["method"] == m]
            lx = dfm["lux"].to_numpy()
            er = dfm["abs_error"].to_numpy()
            bad = er > float(args.thr)
            ids = np.digitize(lx, edges[1:-1], right=False)
            xs, ys = [], []
            for b in range(len(edges)-1):
                mask = (ids == b)
                if mask.sum() < 5:
                    continue
                xs.append(0.5*(edges[b] + edges[b+1]))
                ys.append(float(np.mean(bad[mask])))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("lux (bin center, quantile bins)")
        plt.ylabel(f"rate(|error|>{args.thr})")
        plt.title("Bad-step rate vs lux (binned, pooled)  [proxy of 'cannot drive']")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "07_lux_binned_bad_rate_overlay.png", dpi=200)
        plt.close()

    # optional: raw scatter of lux vs abs_error (sampled)
    if args.scatter_step_sample and args.scatter_step_sample > 0:
        n = int(args.scatter_step_sample)
        for m in sorted(set(steps["method"])):
            dfm = steps[steps["method"] == m]
            if len(dfm) > n:
                dfm = dfm.sample(n=n, random_state=0)
            plt.figure()
            plt.scatter(dfm["lux"], dfm["abs_error"], s=8)
            plt.xlabel("lux")
            plt.ylabel("abs_error")
            plt.title(f"lux vs abs_error scatter (sample={len(dfm)})  method={m}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"scatter_lux_vs_error__{m}.png", dpi=200)
            plt.close()

    print("=== DONE ===")
    print("OUT_ROOT:", out_root)
    print("FIG_DIR :", fig_dir)
    print("Saved:")
    print(" - metrics_long_with_lux.csv")
    print(" - delta_mae_vs_baseline.csv / improve_percent_vs_baseline.csv / improvement_summary.csv")
    print(" - PNG plots 01..07 (+ optional scatter)")
