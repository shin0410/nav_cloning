#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from compare_models_on_test_times import (
    aggregate_detail_by_model,
    infer_one_time,
    load_model,
    load_yaml,
    public_data_dir,
)


DEFAULT_TEST_TIMES = [
    "20260315_11:06:52",
    "20260319_17:02:53",
    "20260316_13:45:21",
]


def parse_model_specs(raw_specs: List[str]) -> Dict[str, Path]:
    model_map: Dict[str, Path] = {}
    for spec in raw_specs:
        if "=" not in spec:
            raise SystemExit(f"[ERR] --model must be in name=/abs/path.pt format: {spec}")
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path = Path(path_str).expanduser().resolve()
        if not name:
            raise SystemExit(f"[ERR] empty model name in spec: {spec}")
        if name in model_map:
            raise SystemExit(f"[ERR] duplicate model name: {name}")
        if not path.is_file():
            raise SystemExit(f"[ERR] model file not found: {path}")
        model_map[name] = path
    return model_map


def read_models_csv(path: Path) -> Dict[str, Path]:
    if not path.is_file():
        raise SystemExit(f"[ERR] --models_csv not found: {path}")
    df = pd.read_csv(path)
    req = {"model", "model_path"}
    miss = req - set(df.columns)
    if miss:
        raise SystemExit(f"[ERR] missing columns in models_csv: {sorted(miss)}")
    model_map: Dict[str, Path] = {}
    for _, row in df.iterrows():
        name = str(row["model"]).strip()
        path = Path(str(row["model_path"])).expanduser().resolve()
        if not name:
            raise SystemExit("[ERR] models_csv has empty model name")
        if name in model_map:
            raise SystemExit(f"[ERR] duplicate model name in models_csv: {name}")
        if not path.is_file():
            raise SystemExit(f"[ERR] model file not found: {path}")
        model_map[name] = path
    return model_map


def load_model_map(args) -> Dict[str, Path]:
    model_map: Dict[str, Path] = {}
    if args.models_csv:
        model_map.update(read_models_csv(Path(args.models_csv).expanduser().resolve()))
    if args.model:
        for name, path in parse_model_specs(args.model).items():
            if name in model_map:
                raise SystemExit(f"[ERR] duplicate model name across inputs: {name}")
            model_map[name] = path
    if not model_map:
        raise SystemExit("[ERR] specify --models_csv and/or --model")
    return model_map


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate explicitly specified model files on test datasets and export CSV summaries."
    )
    ap.add_argument("--nav_dir", default=None, help="default: this script's nav_cloning dir")
    ap.add_argument("--config", default=None, help="default: <nav_dir>/config/config.yaml")
    ap.add_argument("--data_root", default=None, help="optional explicit data root")
    ap.add_argument("--models_csv", default=None, help="CSV with columns: model, model_path")
    ap.add_argument("--model", action="append", default=None, help="repeatable: name=/abs/path.pt")
    ap.add_argument("--test_times", nargs="*", default=DEFAULT_TEST_TIMES)
    ap.add_argument("--test-img-dir", default="img")
    ap.add_argument("--test-vel-dir", default="vel")
    ap.add_argument("--test-first-n", type=int, default=6100)
    ap.add_argument("--view", default="center", choices=["center", "left", "right"])
    ap.add_argument("--thr1", type=float, default=0.1)
    ap.add_argument("--thr2", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    nav_dir = Path(args.nav_dir) if args.nav_dir else Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else (nav_dir / "config" / "config.yaml")
    cfg = load_yaml(cfg_path)

    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        data_root = public_data_dir(cfg, nav_dir)
    if not data_root.is_dir():
        raise SystemExit(f"[ERR] data_root not found: {data_root}")

    model_map = load_model_map(args)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (data_root / f"_compare_named_models_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = pd.DataFrame(
        [{"model": name, "model_path": str(path)} for name, path in model_map.items()]
    ).sort_values("model")
    model_df.to_csv(out_dir / "models_used.csv", index=False)

    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] test_times: {', '.join(args.test_times)}")
    if args.test_first_n is not None:
        print(f"[INFO] test_first_n: {args.test_first_n}")
    print("[INFO] models:")
    for _, r in model_df.iterrows():
        print(f"  {r['model']}: {r['model_path']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    summary_rows = []
    detail_frames = []

    for model_name, model_path in model_map.items():
        print(f"[RUN] model={model_name}")
        model = load_model(model_path, device)
        for t in args.test_times:
            try:
                detail, met = infer_one_time(
                    model=model,
                    data_root=data_root,
                    test_time=t,
                    test_img_dir=args.test_img_dir,
                    test_vel_dir=args.test_vel_dir,
                    view=args.view,
                    batch_size=args.batch_size,
                    device=device,
                    thr1=args.thr1,
                    thr2=args.thr2,
                    test_first_n=args.test_first_n,
                )
            except Exception as e:
                print(f"[WARN] skip test_time={t} model={model_name}: {e}")
                continue

            detail["model"] = model_name
            detail["model_path"] = str(model_path)
            detail_frames.append(detail)

            met["model"] = model_name
            met["model_path"] = str(model_path)
            summary_rows.append(met)
            print(
                f"  [OK] test_time={t} count={met['count']} "
                f"mae={met['mae']:.6f} missing={met['missing_images']}"
            )

    if not summary_rows:
        raise SystemExit("[ERR] no evaluation results were produced")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    col_summary = [
        "test_time",
        "model",
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
        "original_total_rows",
        "eval_total_rows",
        "test_first_n",
        "kept_rows",
        "missing_images",
        "invalid_teacher",
        "model_path",
    ]
    summary_df = summary_df[col_summary].sort_values(["test_time", "model"])
    summary_df.to_csv(out_dir / "summary_by_time_model.csv", index=False)

    detail_df = detail_df.sort_values(["test_time", "model", "episode"])
    detail_df.to_csv(out_dir / "detail_per_sample.csv", index=False)

    model_agg = aggregate_detail_by_model(detail_df, args.thr1, args.thr2)
    model_agg = model_agg.merge(model_df, on="model", how="left")
    model_agg.to_csv(out_dir / "summary_by_model.csv", index=False)

    pivot = summary_df.pivot_table(index="test_time", columns="model", values="mae", aggfunc="first")
    pivot.to_csv(out_dir / "mae_pivot.csv")

    print(f"[DONE] summary_by_time_model: {out_dir / 'summary_by_time_model.csv'}")
    print(f"[DONE] summary_by_model     : {out_dir / 'summary_by_model.csv'}")
    print(f"[DONE] detail_per_sample   : {out_dir / 'detail_per_sample.csv'}")
    print(f"[DONE] models_used         : {out_dir / 'models_used.csv'}")
    print(f"[DONE] mae_pivot           : {out_dir / 'mae_pivot.csv'}")


if __name__ == "__main__":
    main()
