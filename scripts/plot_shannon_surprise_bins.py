#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VIEWS = ("center", "left", "right")


def parse_csv_list(value):
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def collect_csv_paths(args):
    paths = []
    for csv_path in args.csv or []:
        paths.append(Path(csv_path).expanduser().resolve())

    data_root = Path(args.data_root).expanduser().resolve()
    for time_id in args.time or []:
        paths.append(data_root / time_id / "dataset" / args.vel_dir / "data.csv")

    if not paths:
        raise SystemExit("[ERR] pass at least one --csv or --time")

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("[ERR] missing csv:\n  " + "\n  ".join(missing))

    return paths


def load_angles(csv_paths, views, max_steps):
    value_rows = []
    source_summaries = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if max_steps and max_steps > 0:
            df = df.head(int(max_steps)).reset_index(drop=True)

        use_cols = [c for c in views if c in df.columns]
        if not use_cols:
            raise SystemExit(
                f"[ERR] none of requested columns {views} exist in {csv_path}"
            )

        for col in use_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                value_rows.append(
                    pd.DataFrame(
                        {
                            "source_csv": str(csv_path),
                            "view": col,
                            "angle": vals,
                        }
                    )
                )

        flat = (
            df[use_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1)
        )
        flat = flat[np.isfinite(flat)]
        source_summaries.append(
            {
                "source_csv": str(csv_path),
                "rows": int(len(df)),
                "views": ",".join(use_cols),
                "samples": int(flat.size),
                "min_angle": float(np.min(flat)) if flat.size else np.nan,
                "max_angle": float(np.max(flat)) if flat.size else np.nan,
                "mean_angle": float(np.mean(flat)) if flat.size else np.nan,
                "std_angle": float(np.std(flat)) if flat.size else np.nan,
            }
        )

    if not value_rows:
        raise SystemExit("[ERR] no finite angle values found")

    values_df = pd.concat(value_rows, ignore_index=True)
    return values_df, pd.DataFrame(source_summaries)


def assign_bins_like_torch_bucketize(values, edges):
    # learning_surprise.py uses torch.bucketize(values, edges) - 1 with
    # right=False, then clamps to [0, n_bins - 1]. np.searchsorted(...,
    # side="left") matches that boundary behavior.
    idx = np.searchsorted(edges, values, side="left") - 1
    return np.clip(idx, 0, len(edges) - 2)


def build_bin_table(values_df, edges, eps):
    n_bins = len(edges) - 1
    values_df = values_df.copy()
    values_df["bin"] = assign_bins_like_torch_bucketize(
        values_df["angle"].to_numpy(dtype=float), edges
    )

    counts = np.bincount(values_df["bin"].to_numpy(dtype=int), minlength=n_bins).astype(
        float
    )
    total = float(counts.sum())
    probs = counts / (total + eps)
    surprise = -np.log(probs + eps)

    rows = []
    for i in range(n_bins):
        rows.append(
            {
                "bin": i,
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "mid": float((edges[i] + edges[i + 1]) / 2.0),
                "count": int(counts[i]),
                "probability": float(probs[i]),
                "shannon_surprise_weight": float(surprise[i]),
                "relative_to_mean_weight": float(surprise[i] / np.mean(surprise))
                if np.isfinite(np.mean(surprise)) and np.mean(surprise) != 0
                else np.nan,
            }
        )

    bin_table = pd.DataFrame(rows)

    view_counts = (
        values_df.groupby(["bin", "view"], observed=False)
        .size()
        .reset_index(name="view_count")
    )
    return values_df, bin_table, view_counts


def write_plots(values_df, bin_table, view_counts, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        f"{r.left:.3f}\nto\n{r.right:.3f}" for _, r in bin_table.iterrows()
    ]
    x = np.arange(len(bin_table))
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x,
        bin_table["count"].to_numpy(),
        color="#4c78a8",
        edgecolor="#263238",
        linewidth=0.8,
    )
    ax.set_xlabel("angular velocity bin [rad/s]")
    ax.set_ylabel("sample count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        bin_table["shannon_surprise_weight"].to_numpy(),
        color="#d62728",
        marker="o",
        linewidth=2.0,
        label="-log(p)",
    )
    ax2.set_ylabel("Shannon surprise weight: -log(p)")

    for i, row in bin_table.iterrows():
        ax.text(
            i,
            row["count"],
            f'{row["probability"] * 100:.1f}%',
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(title)
    ax.set_title("Higher red line means rarer bin and larger training weight")
    fig.tight_layout()
    fig.savefig(out_dir / "shannon_surprise_bins.png", dpi=180)
    plt.close(fig)

    pivot = (
        view_counts.pivot(index="bin", columns="view", values="view_count")
        .reindex(index=bin_table["bin"].to_list())
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(bin_table), dtype=float)
    for j, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=col,
            color=colors[j % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += vals
    ax.set_xlabel("angular velocity bin [rad/s]")
    ax.set_ylabel("sample count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title + " by view")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "shannon_surprise_bins_by_view.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(values_df["angle"].to_numpy(dtype=float), bins=80, color="#59a14f")
    for edge in bin_table["left"].to_list() + [float(bin_table["right"].iloc[-1])]:
        ax.axvline(edge, color="#d62728", alpha=0.55, linewidth=1.0)
    ax.set_xlabel("angular velocity [rad/s]")
    ax.set_ylabel("sample count")
    ax.set_title(title + " raw histogram with surprise bin edges")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "angular_velocity_hist_with_bin_edges.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot angular velocity bins using the same Shannon surprise "
            "binning as nav_cloning/scripts/learning_surprise.py."
        )
    )
    ap.add_argument("--csv", action="append", help="CSV path. Repeatable.")
    ap.add_argument("--time", action="append", help="Dataset time under --data-root.")
    ap.add_argument(
        "--data-root",
        default="/home/shin/challenge_ws/nav_cloning_data",
        help="Root containing <time>/dataset/<vel-dir>/data.csv",
    )
    ap.add_argument("--vel-dir", default="vel")
    ap.add_argument("--views", default="center,left,right")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    if args.bins < 1:
        raise SystemExit("[ERR] --bins must be >= 1")

    views = parse_csv_list(args.views) or list(DEFAULT_VIEWS)
    csv_paths = collect_csv_paths(args)
    values_df, source_summary = load_angles(csv_paths, views, args.max_steps)

    values = values_df["angle"].to_numpy(dtype=float)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val == max_val:
        min_val -= 1e-6
        max_val += 1e-6
    edges = np.linspace(min_val, max_val, args.bins + 1)

    values_df, bin_table, view_counts = build_bin_table(values_df, edges, args.eps)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_table.to_csv(out_dir / "shannon_surprise_bins.csv", index=False)
    source_summary.to_csv(out_dir / "source_summary.csv", index=False)
    view_counts.to_csv(out_dir / "view_counts_by_bin.csv", index=False)

    if args.title:
        title = args.title
    elif args.time:
        title = "Angular velocity distribution: " + ", ".join(args.time)
    else:
        title = "Angular velocity distribution"

    write_plots(values_df, bin_table, view_counts, out_dir, title)

    frequent = bin_table.sort_values("count", ascending=False).iloc[0]
    rare = bin_table.sort_values("count", ascending=True).iloc[0]
    print(f"[DONE] out_dir: {out_dir}")
    print(
        "[INFO] most frequent bin: "
        f'{frequent["left"]:.4f} to {frequent["right"]:.4f} rad/s, '
        f'count={int(frequent["count"])}, '
        f'p={frequent["probability"]:.4f}, '
        f'weight={frequent["shannon_surprise_weight"]:.4f}'
    )
    print(
        "[INFO] rarest bin: "
        f'{rare["left"]:.4f} to {rare["right"]:.4f} rad/s, '
        f'count={int(rare["count"])}, '
        f'p={rare["probability"]:.4f}, '
        f'weight={rare["shannon_surprise_weight"]:.4f}'
    )


if __name__ == "__main__":
    main()
