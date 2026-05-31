#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_train_compare_center14_gap_202603_ep300_resume_missing.sh"

DATA_DIR="${NAV_DATA_DIR:-/home/shin/challenge_ws/nav_cloning_data}"
OUTPUT_TIME="${NAV_OUTPUT_TIME:-_center14_gap_20260308_0310_ep1000}"
EPOCH="${NAV_EPOCH:-1000}"
REUSE_BASELINE=1

usage() {
  cat <<'EOF'
Usage:
  run_train_compare_center14_gap_202603_ep1000_resume_missing.sh [options]

This is a 1000-epoch wrapper for the center14_gap 13-model experiment.
It trains/evaluates/plots:
  baseline
  augmix_K3/K5/K7/K9
  TrivialAugmentWide_K3/K5/K7/K9
  rand_augment_K3/K5/K7/K9

Resume behavior:
  Existing target .pt files are skipped by default.
  The already trained baseline-1000 model is reused by symlink when available.

Options:
  --data-dir DIR             default: /home/shin/challenge_ws/nav_cloning_data
  --output-time NAME         default: _center14_gap_20260308_0310_ep1000
  --epoch N                  default: 1000
  --no-reuse-baseline        do not symlink the existing baseline-1000 model
  All other options are passed through to run_train_compare_center14_gap_202603_ep300_resume_missing.sh

Recommended:
  ./nav_cloning/scripts/run_train_compare_center14_gap_202603_ep1000_resume_missing.sh --skip-unzip --skip-split
EOF
}

PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
      shift 2
      ;;
    --output-time)
      OUTPUT_TIME="${2:?missing value for --output-time}"
      shift 2
      ;;
    --epoch)
      EPOCH="${2:?missing value for --epoch}"
      shift 2
      ;;
    --no-reuse-baseline)
      REUSE_BASELINE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -f "$BASE_SCRIPT" ]] || { echo "[ERR] missing base script: $BASE_SCRIPT" >&2; exit 1; }
[[ "$EPOCH" =~ ^[0-9]+$ ]] || { echo "[ERR] --epoch must be integer: $EPOCH" >&2; exit 1; }

if [[ "$REUSE_BASELINE" -eq 1 && "$EPOCH" == "1000" ]]; then
  SRC_BASELINE="$DATA_DIR/_center14_gap_20260308_0310_baseline_ep1000_full/model/1000/model_gpu_baseline_center14_allclean9_split4000_until_plateau.pt"
  DST_DIR="$DATA_DIR/$OUTPUT_TIME/model/1000"
  DST_BASELINE="$DST_DIR/model_gpu_baseline_center14_allclean9_split4000.pt"
  if [[ -f "$SRC_BASELINE" && ! -e "$DST_BASELINE" ]]; then
    mkdir -p "$DST_DIR"
    ln -s "$SRC_BASELINE" "$DST_BASELINE"
    echo "[INFO] reused existing baseline-1000 model:"
    echo "  $DST_BASELINE -> $SRC_BASELINE"
  fi
fi

"$BASE_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --output-time "$OUTPUT_TIME" \
  --epoch "$EPOCH" \
  "${PASS_ARGS[@]}"

COMPARE_DIR="$DATA_DIR/$OUTPUT_TIME/compare"
RUN_DIR="$DATA_DIR/$OUTPUT_TIME/result/$EPOCH/run"
BASELINE_RUN_DIR="$DATA_DIR/_center14_gap_20260308_0310_baseline_ep1000_full/result/1000/run"

echo "[STEP] plot 1000-epoch loss/lr curves"
DATA_DIR="$DATA_DIR" OUTPUT_TIME="$OUTPUT_TIME" EPOCH="$EPOCH" RUN_DIR="$RUN_DIR" BASELINE_RUN_DIR="$BASELINE_RUN_DIR" COMPARE_DIR="$COMPARE_DIR" python3 - <<'PY'
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

compare_dir = Path(os.environ["COMPARE_DIR"])
run_dir = Path(os.environ["RUN_DIR"])
baseline_run_dir = Path(os.environ["BASELINE_RUN_DIR"])
compare_dir.mkdir(parents=True, exist_ok=True)

labels = [
    "baseline",
    "augmix_K3", "TrivialAugmentWide_K3", "rand_augment_K3",
    "augmix_K5", "TrivialAugmentWide_K5", "rand_augment_K5",
    "augmix_K7", "TrivialAugmentWide_K7", "rand_augment_K7",
    "augmix_K9", "TrivialAugmentWide_K9", "rand_augment_K9",
]

def read_event(path: Path):
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    if "loss" not in tags:
        return None
    loss = ea.Scalars("loss")
    rows = {
        "epoch": [v.step + 1 for v in loss],
        "loss": [float(v.value) for v in loss],
    }
    if "lr" in tags:
        lr = ea.Scalars("lr")
        rows["lr"] = [float(v.value) for v in lr]
    return pd.DataFrame(rows)

series = []

baseline_events = []
if baseline_run_dir.is_dir():
    baseline_events = sorted(baseline_run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
if baseline_events:
    df = read_event(baseline_events[-1])
    if df is not None:
        df["model"] = "baseline"
        series.append(df)

events = []
if run_dir.is_dir():
    events = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)

start_label_idx = 1 if series else 0
for label, event in zip(labels[start_label_idx:], events):
    df = read_event(event)
    if df is None:
        continue
    df["model"] = label
    series.append(df)

if not series:
    print("[WARN] no TensorBoard scalar events found for loss/lr plots")
    raise SystemExit(0)

hist = pd.concat(series, ignore_index=True)
hist.to_csv(compare_dir / "loss_lr_history_1000epoch.csv", index=False)

summary = []
for model, g in hist.groupby("model", sort=False):
    g = g.sort_values("epoch")
    last = g.iloc[-1]
    best = g.loc[g["loss"].idxmin()]
    row = {
        "model": model,
        "epochs_recorded": int(g["epoch"].max()),
        "final_loss": float(last["loss"]),
        "best_loss": float(best["loss"]),
        "best_epoch": int(best["epoch"]),
    }
    if "lr" in g.columns and g["lr"].notna().any():
        row["final_lr"] = float(last["lr"])
    summary.append(row)
pd.DataFrame(summary).to_csv(compare_dir / "loss_lr_summary_1000epoch.csv", index=False)

plt.figure(figsize=(12, 7))
for model, g in hist.groupby("model", sort=False):
    plt.plot(g["epoch"], g["loss"], linewidth=1.8, label=model)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("1000-epoch training loss")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(compare_dir / "loss_all_models_1000epoch.png", dpi=180)
plt.close()

if "lr" in hist.columns and hist["lr"].notna().any():
    plt.figure(figsize=(10, 5))
    for model, g in hist.groupby("model", sort=False):
        if g["lr"].notna().any():
            plt.plot(g["epoch"], g["lr"], linewidth=1.5, label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("1000-epoch learning rate")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(compare_dir / "lr_all_models_1000epoch.png", dpi=180)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 7))
    for model, g in hist.groupby("model", sort=False):
        ax1.plot(g["epoch"], g["loss"], linewidth=1.6, label=model)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    lr_ref = hist[hist["lr"].notna()].drop_duplicates("epoch")
    ax2.plot(lr_ref["epoch"], lr_ref["lr"], color="black", linestyle="--", linewidth=2, label="lr")
    ax2.set_ylabel("Learning rate")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    plt.title("1000-epoch loss with learning-rate schedule")
    fig.tight_layout()
    plt.savefig(compare_dir / "loss_lr_all_models_1000epoch.png", dpi=180)
    plt.close()

print(f"[DONE] loss/lr history : {compare_dir / 'loss_lr_history_1000epoch.csv'}")
print(f"[DONE] loss/lr summary : {compare_dir / 'loss_lr_summary_1000epoch.csv'}")
print(f"[DONE] loss plot       : {compare_dir / 'loss_all_models_1000epoch.png'}")
PY

echo "[DONE] 1000-epoch resume script complete"
