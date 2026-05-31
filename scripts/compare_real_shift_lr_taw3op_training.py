#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/evaluate real-vs-shift LR variants with 3-op TrivialAugmentWide.

Default comparison:
  - clean real_default / real_surprise / shift_default / shift_surprise
  - taw3op_real_default / taw3op_real_surprise / taw3op_shift_default / taw3op_shift_surprise

The TAW data is generated under the output directory, so the original
nav_cloning/data/<time>/dataset folders are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import compare_real_vs_shift_lr_training as base
from edit_dataset.aug_TrivialAugmentWide import (
    get_available_ops,
    make_trivial_transform,
    to_float01_bgr,
    trivialaug_variants_bgr,
)


TRAIN_TIMES_DEFAULT = (
    "20260428_12:31:05",
    "20260428_16:50:47",
    "20260428_19:29:22",
)
TEST_TIME_DEFAULT = "20260426_13_39_33"
DEFAULT_OPS = ("Equalize", "Brightness", "AutoContrast")
VIEWS = ("center", "left", "right")


@dataclass(frozen=True)
class Source:
    img_dir: Path
    csv_path: Path
    source_name: str


@dataclass(frozen=True)
class Sample:
    img_dir: Path
    episode: str
    view: str
    file_view: str
    transform: str
    target: float
    source_name: str


def parse_ops(raw: str | Sequence[str]) -> List[str]:
    parts = raw.split(",") if isinstance(raw, str) else raw
    out: List[str] = []
    seen = set()
    for p in parts:
        s = str(p).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    if not out:
        raise ValueError("ops is empty")
    return out


def safe_name(raw: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in raw)


def stable_seed(raw: str) -> int:
    h = hashlib.md5(str(raw).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def validate_ops(ops: Sequence[str], magnitude_bins: int) -> None:
    available = set(get_available_ops(magnitude_bins))
    invalid = [op for op in ops if op not in available]
    if invalid:
        raise ValueError(f"invalid TAW ops: {invalid}; available={sorted(available)}")


def clean_source(data_root: Path, time_id: str) -> Source:
    img_dir, csv_path = base.dataset_paths(data_root, time_id)
    return Source(img_dir=img_dir, csv_path=csv_path, source_name="clean")


def generate_taw_dataset(
    data_root: Path,
    time_id: str,
    out_root: Path,
    *,
    k: int,
    magnitude_bins: int,
    ops: Sequence[str],
    same3view: bool,
    force: bool,
) -> Source:
    src = clean_source(data_root, time_id)
    ops_tag = safe_name("_".join(ops))
    safe_time = time_id.replace(":", "-")
    out_base = out_root / safe_time / f"taw_K{k}_M{magnitude_bins}_OPS{ops_tag}"
    out_img_dir = out_base / "img"
    out_vel_dir = out_base / "vel"
    out_csv = out_vel_dir / "data.csv"

    if out_csv.is_file() and out_img_dir.is_dir() and not force:
        return Source(img_dir=out_img_dir, csv_path=out_csv, source_name="taw3op")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_vel_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src.csv_path)
    if "episode" not in df.columns:
        raise ValueError(f"missing episode column: {src.csv_path}")

    transform = make_trivial_transform(
        magnitude_bins,
        allowed_ops=list(ops),
        max_severity_ratio=1.0,
        max_severity_level=None,
    )

    records: List[dict] = []
    missing = 0
    for _, row in df.iterrows():
        episode = str(row["episode"]).replace(".0", "")
        imgs = {}
        ok = True
        for view in VIEWS:
            img_path = src.img_dir / f"{episode}_{view}.npy"
            if not img_path.is_file():
                ok = False
                break
            imgs[view] = to_float01_bgr(np.load(img_path, mmap_mode="r"))
        if not ok:
            missing += 1
            continue

        seed = stable_seed(episode) if same3view else None
        variants = {
            view: trivialaug_variants_bgr(imgs[view], k, transform, same_across_views_seed=seed)
            for view in VIEWS
        }
        for i in range(k):
            aug_episode = f"{episode}_{i}"
            for view in VIEWS:
                np.save(out_img_dir / f"{aug_episode}_{view}.npy", variants[view][i])
            rec = row.to_dict()
            rec["episode"] = aug_episode
            records.append(rec)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(
        f"[TAW] time={time_id} rows={len(records)} missing={missing} "
        f"img={out_img_dir} csv={out_csv}",
        flush=True,
    )
    return Source(img_dir=out_img_dir, csv_path=out_csv, source_name="taw3op")


class RealShiftDataset(Dataset):
    def __init__(
        self,
        clean_sources: Sequence[Source],
        *,
        image_mode: str,
        taw_sources: Sequence[Source] = (),
        include_clean: bool = True,
        max_steps: int = 0,
        camera_angle: float = 24.61,
        camera_fov: float = 150.0,
    ):
        if image_mode not in {"real", "shift"}:
            raise ValueError("image_mode must be real or shift")
        self.image_mode = image_mode
        self.camera_angle = float(camera_angle)
        self.camera_fov = float(camera_fov)
        self.samples: List[Sample] = []

        if include_clean:
            for source in clean_sources:
                self._add_source(source, max_steps=max_steps)
        for source in taw_sources:
            self._add_source(source, max_steps=0)

        if not self.samples:
            raise RuntimeError(f"no samples for image_mode={image_mode}")

    def _add_source(self, source: Source, max_steps: int) -> None:
        df = base.load_csv(source.csv_path, max_steps=max_steps)
        for _, row in df.iterrows():
            episode = str(row["episode"]).replace(".0", "")
            for view, file_view, transform, target in self._sample_specs(row):
                if not math.isfinite(target):
                    continue
                img_path = source.img_dir / f"{episode}_{file_view}.npy"
                if not img_path.is_file():
                    continue
                self.samples.append(
                    Sample(
                        img_dir=source.img_dir,
                        episode=episode,
                        view=view,
                        file_view=file_view,
                        transform=transform,
                        target=target,
                        source_name=source.source_name,
                    )
                )

    def _sample_specs(self, row: pd.Series) -> List[Tuple[str, str, str, float]]:
        if self.image_mode == "real":
            return [(view, view, "none", float(row[view])) for view in VIEWS]
        return [(view, "center", "shift", float(row[view])) for view in VIEWS]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        arr = base.normalize_img(np.load(s.img_dir / f"{s.episode}_{s.file_view}.npy", mmap_mode="r"))
        if s.transform == "shift":
            arr = base.shifted_view(arr, s.view, self.camera_angle, self.camera_fov)
        x = torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1).contiguous()
        y = torch.tensor([s.target], dtype=torch.float32)
        return x, y


def collect_targets(dataset: RealShiftDataset) -> np.ndarray:
    return np.asarray([s.target for s in dataset.samples], dtype=np.float32)


def condition_rows(taw_only: bool) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    if not taw_only:
        rows.extend(
            [
                ("real_default", "clean", "real", "default"),
                ("real_surprise", "clean", "real", "surprise"),
                ("shift_default", "clean", "shift", "default"),
                ("shift_surprise", "clean", "shift", "surprise"),
            ]
        )
    rows.extend(
        [
            ("taw3op_real_default", "taw3op", "real", "default"),
            ("taw3op_real_surprise", "taw3op", "real", "surprise"),
            ("taw3op_shift_default", "taw3op", "shift", "default"),
            ("taw3op_shift_surprise", "taw3op", "shift", "surprise"),
        ]
    )
    return rows


def write_clean_vs_taw_table(summary: pd.DataFrame, out_dir: Path) -> None:
    pivot = summary.pivot_table(
        index=["image_mode", "loss_mode"],
        columns="train_variant",
        values="mae",
        aggfunc="first",
    ).reset_index()
    if {"clean", "taw3op"} <= set(pivot.columns):
        pivot["delta_taw3op_minus_clean"] = pivot["taw3op"] - pivot["clean"]
        pivot["improvement_clean_minus_taw3op"] = pivot["clean"] - pivot["taw3op"]
    pivot.to_csv(out_dir / "mae_clean_vs_taw3op.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--train-times", default=",".join(TRAIN_TIMES_DEFAULT))
    parser.add_argument("--test-time", default=TEST_TIME_DEFAULT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bin", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-angle", type=float, default=24.61)
    parser.add_argument("--camera-fov", type=float, default=150.0)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--taw-k", type=int, default=3)
    parser.add_argument("--taw-mag", type=int, default=31)
    parser.add_argument("--taw-ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--no-same3view", action="store_true")
    parser.add_argument("--force-regen-taw", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--taw-only", action="store_true", help="train/evaluate only the 4 TAW conditions")
    parser.add_argument(
        "--reuse-clean-model-dir",
        default="",
        help=(
            "Directory containing existing clean models "
            "(real_default.pt, real_surprise.pt, shift_default.pt, shift_surprise.pt). "
            "When set, clean conditions are evaluated from these files and not retrained."
        ),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    train_times = base.parse_times(args.train_times)
    test_time = str(args.test_time).strip()
    ops = parse_ops(args.taw_ops)
    validate_ops(ops, args.taw_mag)

    if not train_times:
        raise ValueError("--train-times is empty")
    if not test_time:
        raise ValueError("--test-time is empty")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    default_out = (
        data_root
        / f"experiment_real_shift_lr_taw3op_train3_test_{test_time.replace(':', '-')}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] train_times={train_times}")
    print(f"[INFO] test_time={test_time}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] device={device} epochs={args.epochs} batch={args.batch_size}")
    print(f"[INFO] TAW K={args.taw_k} M={args.taw_mag} ops={','.join(ops)}")

    reuse_clean_model_dir = Path(args.reuse_clean_model_dir).expanduser().resolve() if args.reuse_clean_model_dir else None
    if reuse_clean_model_dir:
        if not reuse_clean_model_dir.is_dir():
            raise FileNotFoundError(f"--reuse-clean-model-dir not found: {reuse_clean_model_dir}")
        print(f"[INFO] reuse clean models from: {reuse_clean_model_dir}")

    clean_sources = [clean_source(data_root, t) for t in train_times]
    taw_root = out_dir / "taw_datasets"
    taw_sources = [
        generate_taw_dataset(
            data_root,
            t,
            taw_root,
            k=args.taw_k,
            magnitude_bins=args.taw_mag,
            ops=ops,
            same3view=(not args.no_same3view),
            force=args.force_regen_taw,
        )
        for t in train_times
    ]

    conditions = condition_rows(args.taw_only)
    histories: Dict[str, Dict[str, List[float]]] = {}
    condition_targets: Dict[str, np.ndarray] = {}
    model_paths: Dict[str, Path] = {}

    for condition, train_variant, image_mode, loss_mode in conditions:
        print(f"[TRAIN] {condition}", flush=True)
        train_ds = RealShiftDataset(
            clean_sources,
            image_mode=image_mode,
            taw_sources=taw_sources if train_variant == "taw3op" else (),
            include_clean=True,
            max_steps=args.max_steps,
            camera_angle=args.camera_angle,
            camera_fov=args.camera_fov,
        )
        condition_targets[condition] = collect_targets(train_ds)
        if train_variant == "clean" and reuse_clean_model_dir is not None:
            model_path = reuse_clean_model_dir / f"{condition}.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"reused clean model not found: {model_path}")
            model_paths[condition] = model_path
            print(f"  reuse existing clean model: {model_path}")
            continue

        model_path = model_dir / f"{condition}.pt"
        model_paths[condition] = model_path
        hist_path = model_dir / f"{condition}_history.json"
        if args.skip_existing and model_path.is_file() and hist_path.is_file():
            histories[condition] = json.loads(hist_path.read_text(encoding="utf-8"))
            print(f"  skip existing model: {model_path}")
            continue
        print(f"  samples={len(train_ds)}")
        history = base.train_one(
            train_ds=train_ds,
            loss_mode=loss_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_min=args.lr_min,
            device=device,
            out_model=model_path,
            bin_count=args.bin,
            num_workers=args.num_workers,
        )
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        histories[condition] = history

    print("[EVAL] center-camera test inference", flush=True)
    eval_ds = base.CenterEvalDataset(data_root=data_root, time_id=test_time, max_steps=args.max_steps)
    pred_df = eval_ds.df.copy()
    for condition, _, _, _ in conditions:
        model_path = model_paths[condition]
        preds = base.predict_center(
            model_path=model_path,
            eval_ds=eval_ds,
            batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        pred_df[f"pred_{condition}"] = preds
    pred_df.to_csv(out_dir / "test_predictions_center.csv", index=False)

    summary_rows = []
    metrics_by_condition = {}
    for condition, train_variant, image_mode, loss_mode in conditions:
        metrics = base.calc_metrics(pred_df, f"pred_{condition}", teacher_col="center")
        row = {
            "condition": condition,
            "train_variant": train_variant,
            "image_mode": image_mode,
            "loss_mode": loss_mode,
            **metrics,
        }
        summary_rows.append(row)
        metrics_by_condition[condition] = row
    summary = pd.DataFrame(summary_rows).sort_values("mae").reset_index(drop=True)
    summary.to_csv(out_dir / "mae_summary.csv", index=False)
    (out_dir / "mae_summary.json").write_text(
        json.dumps(metrics_by_condition, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_clean_vs_taw_table(summary, out_dir)

    base.save_history_plot(histories, out_dir)
    base.plot_angle_distributions(data_root, train_times, test_time, out_dir, max_steps=args.max_steps)
    base.plot_condition_target_distributions(condition_targets, out_dir)
    base.plot_teacher_by_dataset(data_root, train_times, test_time, out_dir, max_steps=args.max_steps)
    base.plot_predictions(pred_df, [c[0] for c in conditions], out_dir)
    base.plot_mae(summary, out_dir)

    shift_px = ((64 / 2.0) / math.tan(math.radians(args.camera_fov / 2.0))) * math.tan(
        math.radians(args.camera_angle)
    )
    manifest = {
        "data_root": str(data_root),
        "train_times": train_times,
        "test_time": test_time,
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "bin": args.bin,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "camera_angle": args.camera_angle,
        "camera_fov": args.camera_fov,
        "shift_px_at_width_64": shift_px,
        "device": str(device),
        "taw": {
            "k": args.taw_k,
            "magnitude_bins": args.taw_mag,
            "ops": ops,
            "same3view": not args.no_same3view,
            "source": "clean original + generated TAW data",
            "datasets_root": str(taw_root),
        },
        "reuse_clean_model_dir": str(reuse_clean_model_dir) if reuse_clean_model_dir else "",
        "model_paths": {k: str(v) for k, v in model_paths.items()},
        "conditions": [c[0] for c in conditions],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]")
    print(summary[["condition", "train_variant", "image_mode", "loss_mode", "mae", "rmse"]].to_string(index=False))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
