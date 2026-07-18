#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A案: 事前学習 × 拡張 × データ量 × シード の因子計画を一括実行する。

各セルを train_cell.py で学習し、same条件 / gap条件のテストセットで評価して
results_long.csv に追記する。学習済みセル・評価済み行はスキップ (レジューム可)。

例 (屋内 hour-matrix データの場合):
  python run_factorial.py \
    --data_root /home/shin/challenge_ws/nav_cloning_data \
    --train_times 20260308_120428 \
    --test_times_same 20260308_130242 \
    --test_times_gap 20260308_100554 20260310_180327 \
    --rows_list 1000,2000,4000,all --encoders scratch,dinov2_vits14 \
    --augs none,taw3op --seeds 1,2,3 \
    --out_dir /home/shin/challenge_ws/nav_cloning_data/_factorial_A_202607
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from common import evaluate_model_on_time, pick_device

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_times", nargs="+", required=True)
    ap.add_argument("--test_times_same", nargs="+", default=[],
                    help="学習と同条件(同時間帯など)のテストデータ")
    ap.add_argument("--test_times_gap", nargs="+", default=[],
                    help="条件ギャップのあるテストデータ")
    ap.add_argument("--rows_list", default="1000,2000,4000,all")
    ap.add_argument("--encoders", default="scratch,dinov2_vits14")
    ap.add_argument("--augs", default="none,taw3op")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs_scratch", type=int, default=100)
    ap.add_argument("--epochs_head", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--eval_first_n", type=int, default=6100)
    ap.add_argument("--aug_k", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def cell_name(enc: str, aug: str, rows: str, seed: int) -> str:
    return "%s__%s__r%s__s%d" % (enc, aug, rows, seed)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    cache_dir = out_dir / "feature_cache"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows_list = [r.strip() for r in args.rows_list.split(",") if r.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    augs = [a.strip() for a in args.augs.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not (args.test_times_same or args.test_times_gap):
        raise SystemExit("[ERR] test_times_same / test_times_gap のどちらかは必要です")

    results_csv = out_dir / "results_long.csv"
    header = ["cell", "encoder", "aug", "rows", "seed", "test_time", "gap_type",
              "count", "mae", "rmse", "median_abs_error", "p95_abs_error",
              "bias_mean_error", "model_path"]
    done = set()
    if results_csv.is_file():
        with open(results_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["cell"], row["test_time"]))
    else:
        with open(results_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    test_sets = [(t, "same") for t in args.test_times_same] + \
                [(t, "gap") for t in args.test_times_gap]

    n_cells = len(encoders) * len(augs) * len(rows_list) * len(seeds)
    print("[INFO] cells=%d, test_sets=%d, out=%s" % (n_cells, len(test_sets), out_dir))

    device = pick_device(args.device)
    backbone_cache = {}
    idx = 0
    for enc in encoders:
        for aug in augs:
            for rows in rows_list:
                for seed in seeds:
                    idx += 1
                    name = cell_name(enc, aug, rows, seed)
                    model_path = models_dir / (name + ".pt")
                    rows_n = 0 if rows == "all" else int(rows)
                    epochs = args.epochs_scratch if enc == "scratch" else args.epochs_head

                    if model_path.is_file():
                        print("[%d/%d] skip train (exists): %s" % (idx, n_cells, name))
                    else:
                        cmd = [sys.executable, str(SCRIPT_DIR / "train_cell.py"),
                               "--data_root", args.data_root,
                               "--train_times"] + args.train_times + [
                               "--rows", str(rows_n),
                               "--encoder", enc,
                               "--aug", aug,
                               "--aug-k", str(args.aug_k),
                               "--seed", str(seed),
                               "--epochs", str(epochs),
                               "--batch_size", str(args.batch_size),
                               "--feature_cache", str(cache_dir),
                               "--out_model", str(model_path)]
                        if args.device:
                            cmd += ["--device", args.device]
                        print("[%d/%d] TRAIN %s" % (idx, n_cells, name))
                        if args.dry_run:
                            print("  " + " ".join(cmd))
                        else:
                            subprocess.run(cmd, check=True)

                    for test_time, gap_type in test_sets:
                        if (name, test_time) in done:
                            continue
                        if args.dry_run or not model_path.is_file():
                            print("  EVAL (pending) %s on %s [%s]" % (name, test_time, gap_type))
                            continue
                        m = evaluate_model_on_time(
                            str(model_path), args.data_root, test_time, device,
                            first_n=args.eval_first_n,
                            batch_size=args.eval_batch_size,
                            backbone_cache=backbone_cache,
                            feature_cache_dir=str(cache_dir))
                        with open(results_csv, "a", newline="") as f:
                            csv.writer(f).writerow([
                                name, enc, aug, rows, seed, test_time, gap_type,
                                m["count"], "%.6f" % m["mae"], "%.6f" % m["rmse"],
                                "%.6f" % m["median_abs_error"], "%.6f" % m["p95_abs_error"],
                                "%.6f" % m["bias_mean_error"], str(model_path)])
                        done.add((name, test_time))
                        print("  EVAL %s on %s [%s]: mae=%.4f" %
                              (name, test_time, gap_type, m["mae"]))

    print("[DONE] results: %s" % results_csv)
    print("       次: python plot_factorial.py --results %s --out_dir %s" %
          (results_csv, out_dir / "plots"))


if __name__ == "__main__":
    main()
