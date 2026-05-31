#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch

from compare_models_on_test_times import infer_one_time, load_model


def extract_hour(time_id: str) -> int:
    try:
        return int(str(time_id).split("_", 1)[1][:2])
    except Exception as e:
        raise ValueError(f"failed to extract hour from time_id={time_id}") from e


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate baseline models trained per hour against all test hours and export summary CSVs."
    )
    ap.add_argument("--model-map", required=True, help="CSV with columns: train_time, model_path")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--test-times", nargs="+", required=True)
    ap.add_argument("--test-img-dir", default="test_rest_img")
    ap.add_argument("--test-vel-dir", default="test_rest_vel")
    ap.add_argument("--view", default="center", choices=["center", "left", "right"])
    ap.add_argument("--thr1", type=float, default=0.1)
    ap.add_argument("--thr2", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(args.model_map)
    req = {"train_time", "model_path"}
    miss = req - set(model_df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in model map: {sorted(miss)}")

    if "train_hour" not in model_df.columns:
        model_df["train_hour"] = model_df["train_time"].map(extract_hour)

    model_df = model_df.sort_values(["train_hour", "train_time"]).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    summary_rows = []
    detail_frames: List[pd.DataFrame] = []

    for _, row in model_df.iterrows():
        train_time = str(row["train_time"])
        train_hour = int(row["train_hour"])
        model_path = Path(row["model_path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")

        print(f"[RUN] train_time={train_time} model={model_path.name}")
        model = load_model(model_path, device)

        for test_time in args.test_times:
            test_hour = extract_hour(test_time)
            detail, metrics = infer_one_time(
                model=model,
                data_root=data_root,
                test_time=test_time,
                test_img_dir=args.test_img_dir,
                test_vel_dir=args.test_vel_dir,
                view=args.view,
                batch_size=args.batch_size,
                device=device,
                thr1=args.thr1,
                thr2=args.thr2,
            )
            detail["train_time"] = train_time
            detail["train_hour"] = train_hour
            detail["test_hour"] = test_hour
            detail["same_hour"] = int(train_hour == test_hour)
            detail["hour_gap"] = abs(train_hour - test_hour)
            detail["model_path"] = str(model_path)
            detail_frames.append(detail)

            metrics["train_time"] = train_time
            metrics["train_hour"] = train_hour
            metrics["test_hour"] = test_hour
            metrics["same_hour"] = int(train_hour == test_hour)
            metrics["hour_gap"] = abs(train_hour - test_hour)
            metrics["model_path"] = str(model_path)
            summary_rows.append(metrics)
            print(
                f"  [OK] test_time={test_time} "
                f"mae={metrics['mae']:.6f} "
                f"same_hour={metrics['same_hour']}"
            )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    summary_cols = [
        "train_time",
        "train_hour",
        "test_time",
        "test_hour",
        "same_hour",
        "hour_gap",
        "count",
        "mae",
        "mse",
        "rmse",
        "median",
        "p80",
        "p95",
        "sign_flip_rate",
        f"thr_disagree@{args.thr1}",
        f"thr_disagree@{args.thr2}",
        "total_rows",
        "kept_rows",
        "missing_images",
        "invalid_teacher",
        "model_path",
    ]
    summary_df = summary_df[summary_cols].sort_values(["train_hour", "test_hour", "train_time", "test_time"])
    summary_df.to_csv(out_dir / "summary_by_train_test.csv", index=False)

    detail_df = detail_df.sort_values(["train_hour", "test_hour", "train_time", "test_time", "episode"])
    detail_df.to_csv(out_dir / "detail_per_sample.csv", index=False)

    pivot_time = summary_df.pivot_table(index="train_time", columns="test_time", values="mae", aggfunc="first")
    pivot_time.to_csv(out_dir / "mae_matrix_by_time.csv")

    pivot_hour = summary_df.pivot_table(index="train_hour", columns="test_hour", values="mae", aggfunc="mean")
    pivot_hour.to_csv(out_dir / "mae_matrix_by_hour.csv")

    same_diff = (
        summary_df.groupby("same_hour", as_index=False)
        .agg(
            n_pairs=("mae", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            min_mae=("mae", "min"),
            max_mae=("mae", "max"),
        )
        .sort_values("same_hour")
    )
    same_diff["same_or_diff"] = same_diff["same_hour"].map({1: "same_hour", 0: "different_hour"})
    same_diff.to_csv(out_dir / "summary_same_vs_diff.csv", index=False)

    same_diff_by_train = (
        summary_df.groupby(["train_hour", "same_hour"], as_index=False)
        .agg(
            n_pairs=("mae", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
        )
        .sort_values(["train_hour", "same_hour"])
    )
    same_diff_by_train["same_or_diff"] = same_diff_by_train["same_hour"].map({1: "same_hour", 0: "different_hour"})
    same_diff_by_train.to_csv(out_dir / "summary_same_vs_diff_by_train_hour.csv", index=False)

    hour_gap = (
        summary_df.groupby("hour_gap", as_index=False)
        .agg(
            n_pairs=("mae", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
        )
        .sort_values("hour_gap")
    )
    hour_gap.to_csv(out_dir / "summary_by_hour_gap.csv", index=False)

    model_df.to_csv(out_dir / "models_used.csv", index=False)

    print(f"[DONE] summary_by_train_test: {out_dir / 'summary_by_train_test.csv'}")
    print(f"[DONE] detail_per_sample   : {out_dir / 'detail_per_sample.csv'}")
    print(f"[DONE] mae_matrix_by_time  : {out_dir / 'mae_matrix_by_time.csv'}")
    print(f"[DONE] mae_matrix_by_hour  : {out_dir / 'mae_matrix_by_hour.csv'}")
    print(f"[DONE] same_vs_diff        : {out_dir / 'summary_same_vs_diff.csv'}")
    print(f"[DONE] by_hour_gap         : {out_dir / 'summary_by_hour_gap.csv'}")


if __name__ == "__main__":
    main()
