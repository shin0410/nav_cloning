#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


VIEWS = ("center", "left", "right")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_episode(ep) -> str:
    return str(ep).split(".")[0]


def materialize_split(df: pd.DataFrame, src_img_dir: Path, out_img_dir: Path, out_csv: Path) -> Tuple[int, int]:
    rows = []
    missing_rows = 0

    for _, row in df.iterrows():
        ep = safe_episode(row["episode"])
        srcs: List[Path] = []
        ok = True
        for view in VIEWS:
            src = src_img_dir / f"{ep}_{view}.npy"
            if not src.is_file():
                ok = False
                break
            srcs.append(src)

        if not ok:
            missing_rows += 1
            continue

        for view, src in zip(VIEWS, srcs):
            dst = out_img_dir / f"{ep}_{view}.npy"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)

        rec = row.to_dict()
        rec["episode"] = ep
        rows.append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return len(rows), missing_rows


def split_one_time(
    data_root: Path,
    time_id: str,
    train_rows: int,
    train_img_dir_name: str,
    train_vel_dir_name: str,
    test_img_dir_name: str,
    test_vel_dir_name: str,
) -> dict:
    dataset_dir = data_root / time_id / "dataset"
    src_img_dir = dataset_dir / "img"
    src_csv = dataset_dir / "vel" / "data.csv"

    if not src_img_dir.is_dir():
        raise FileNotFoundError(f"img dir not found: {src_img_dir}")
    if not src_csv.is_file():
        raise FileNotFoundError(f"csv not found: {src_csv}")

    df = pd.read_csv(src_csv)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found: {src_csv}")

    df_train = df.head(train_rows).copy()
    df_test = df.iloc[train_rows:].copy()

    train_img_dir = dataset_dir / train_img_dir_name
    train_vel_dir = dataset_dir / train_vel_dir_name
    test_img_dir = dataset_dir / test_img_dir_name
    test_vel_dir = dataset_dir / test_vel_dir_name

    for p in (train_img_dir, train_vel_dir, test_img_dir, test_vel_dir):
        ensure_clean_dir(p)

    train_kept, train_missing = materialize_split(
        df_train,
        src_img_dir=src_img_dir,
        out_img_dir=train_img_dir,
        out_csv=train_vel_dir / "data.csv",
    )
    test_kept, test_missing = materialize_split(
        df_test,
        src_img_dir=src_img_dir,
        out_img_dir=test_img_dir,
        out_csv=test_vel_dir / "data.csv",
    )

    return {
        "time": time_id,
        "source_rows": int(len(df)),
        "requested_train_rows": int(train_rows),
        "raw_train_rows": int(len(df_train)),
        "raw_test_rows": int(len(df_test)),
        "train_kept_rows": int(train_kept),
        "train_missing_rows": int(train_missing),
        "test_kept_rows": int(test_kept),
        "test_missing_rows": int(test_missing),
        "train_img_dir": str(train_img_dir),
        "train_csv": str(train_vel_dir / "data.csv"),
        "test_img_dir": str(test_img_dir),
        "test_csv": str(test_vel_dir / "data.csv"),
    }


def normalize_times(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        t = str(value).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split each nav_cloning dataset into first-N rows for train and the remainder for test."
    )
    ap.add_argument("--data-root", required=True, help="nav_cloning/data root")
    ap.add_argument("--times", nargs="+", required=True, help="time ids to split")
    ap.add_argument("--train-rows", type=int, default=4000)
    ap.add_argument("--train-img-dir", default="train4000_img")
    ap.add_argument("--train-vel-dir", default="train4000_vel")
    ap.add_argument("--test-img-dir", default="test_rest_img")
    ap.add_argument("--test-vel-dir", default="test_rest_vel")
    ap.add_argument("--summary-csv", default="", help="optional output CSV path")
    args = ap.parse_args()

    if args.train_rows <= 0:
        raise SystemExit("[ERR] --train-rows must be > 0")

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"[ERR] --data-root not found: {data_root}")

    rows = []
    for time_id in normalize_times(args.times):
        row = split_one_time(
            data_root=data_root,
            time_id=time_id,
            train_rows=args.train_rows,
            train_img_dir_name=args.train_img_dir,
            train_vel_dir_name=args.train_vel_dir,
            test_img_dir_name=args.test_img_dir,
            test_vel_dir_name=args.test_vel_dir,
        )
        rows.append(row)
        print(
            f"[DONE] {time_id}: "
            f"train={row['train_kept_rows']} (missing={row['train_missing_rows']}), "
            f"test={row['test_kept_rows']} (missing={row['test_missing_rows']})"
        )

    if args.summary_csv:
        out_csv = Path(args.summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[DONE] summary: {out_csv}")


if __name__ == "__main__":
    main()
