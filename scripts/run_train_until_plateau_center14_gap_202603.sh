#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_PY="$SCRIPT_DIR/learning_surprise_until_plateau.py"

DATA_DIR="${NAV_DATA_DIR:-/home/shin/challenge_ws/nav_cloning_data}"
SOURCE_TIME="${NAV_SOURCE_TIME:-_center14_gap_20260308_0310_ep300}"
OUTPUT_TIME="${NAV_OUTPUT_TIME:-_center14_gap_20260308_0310_until_plateau}"
MAX_EPOCH="${NAV_EPOCH:-1000}"
MIN_EPOCH="${NAV_MIN_EPOCH:-800}"
PLATEAU_WINDOW="${NAV_PLATEAU_WINDOW:-50}"
PLATEAU_REL_DROP="${NAV_PLATEAU_REL_DROP:-0.01}"
CHECKPOINT_EVERY="${NAV_CHECKPOINT_EVERY:-50}"
MODEL="baseline"

usage() {
  cat <<'EOF'
Usage:
  run_train_until_plateau_center14_gap_202603.sh --model MODEL [options]

Models:
  baseline
  augmix_K3, augmix_K5, augmix_K7, augmix_K9
  taw_K3, taw_K5, taw_K7, taw_K9
  rand_augment_K3, rand_augment_K5, rand_augment_K7, rand_augment_K9

Options:
  --model NAME          model/dataset condition to train
  --data-dir DIR        default: /home/shin/challenge_ws/nav_cloning_data
  --source-time NAME    source mixed dataset time, default: _center14_gap_20260308_0310_ep300
  --output-time NAME    output time, default: _center14_gap_20260308_0310_until_plateau
  --max-epoch N         hard cap, default: 1000
  --min-epoch N         do not stop before this epoch, default: 800
  --window N            plateau check window, default: 50
  --rel-drop X          stop if loss drop over window is below X, default: 0.01
  --checkpoint-every N  save checkpoint every N epochs, default: 50

Example:
  ./nav_cloning/scripts/run_train_until_plateau_center14_gap_202603.sh --model baseline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
      shift 2
      ;;
    --source-time)
      SOURCE_TIME="${2:?missing value for --source-time}"
      shift 2
      ;;
    --output-time)
      OUTPUT_TIME="${2:?missing value for --output-time}"
      shift 2
      ;;
    --max-epoch)
      MAX_EPOCH="${2:?missing value for --max-epoch}"
      shift 2
      ;;
    --min-epoch)
      MIN_EPOCH="${2:?missing value for --min-epoch}"
      shift 2
      ;;
    --window)
      PLATEAU_WINDOW="${2:?missing value for --window}"
      shift 2
      ;;
    --rel-drop)
      PLATEAU_REL_DROP="${2:?missing value for --rel-drop}"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:?missing value for --checkpoint-every}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$MODEL" in
  baseline)
    LOAD_IMG="mix_baseline_center14_allclean9_img"
    LOAD_VEL="mix_baseline_center14_allclean9_vel"
    SAVE_MODEL="model_gpu_baseline_center14_allclean9_split4000_until_plateau.pt"
    ;;
  augmix_K3|augmix_K5|augmix_K7|augmix_K9)
    K="${MODEL#augmix_K}"
    LOAD_IMG="mix_augmix_center14_gap_K${K}_img"
    LOAD_VEL="mix_augmix_center14_gap_K${K}_vel"
    SAVE_MODEL="model_gpu_augmix_K${K}_center14_gap_split4000_until_plateau.pt"
    ;;
  taw_K3|taw_K5|taw_K7|taw_K9)
    K="${MODEL#taw_K}"
    LOAD_IMG="mix_taw_center14_gap_K${K}_img"
    LOAD_VEL="mix_taw_center14_gap_K${K}_vel"
    SAVE_MODEL="model_gpu_trivialaugwide_K${K}_center14_gap_split4000_until_plateau.pt"
    ;;
  rand_augment_K3|rand_augment_K5|rand_augment_K7|rand_augment_K9)
    K="${MODEL#rand_augment_K}"
    LOAD_IMG="mix_randaugment_center14_gap_K${K}_img"
    LOAD_VEL="mix_randaugment_center14_gap_K${K}_vel"
    SAVE_MODEL="model_gpu_randaugment_K${K}_center14_gap_split4000_until_plateau.pt"
    ;;
  *)
    echo "[ERR] unsupported model: $MODEL" >&2
    usage
    exit 1
    ;;
esac

SRC_DATASET="$DATA_DIR/$SOURCE_TIME/dataset"
[[ -d "$SRC_DATASET/$LOAD_IMG" ]] || { echo "[ERR] missing img dir: $SRC_DATASET/$LOAD_IMG" >&2; exit 1; }
[[ -f "$SRC_DATASET/$LOAD_VEL/data.csv" ]] || { echo "[ERR] missing csv: $SRC_DATASET/$LOAD_VEL/data.csv" >&2; exit 1; }

echo "[RUN] until plateau: $MODEL"
echo "[INFO] source_time=$SOURCE_TIME"
echo "[INFO] output_time=$OUTPUT_TIME"
echo "[INFO] max_epoch=$MAX_EPOCH min_epoch=$MIN_EPOCH window=$PLATEAU_WINDOW rel_drop=$PLATEAU_REL_DROP"
echo "[INFO] load_img=$LOAD_IMG"
echo "[INFO] load_vel=$LOAD_VEL"
echo "[INFO] save_model=$SAVE_MODEL"

NAV_DATA_DIR="$DATA_DIR" \
NAV_TIME="$SOURCE_TIME" \
NAV_TRAIN_TIMES="$SOURCE_TIME" \
NAV_OUTPUT_TIME="$OUTPUT_TIME" \
NAV_LOAD_DATASET_IMG="$LOAD_IMG" \
NAV_LOAD_DATASET_VEL="$LOAD_VEL" \
NAV_SAVE_MODEL="$SAVE_MODEL" \
NAV_EPOCH="$MAX_EPOCH" \
NAV_MIN_EPOCH="$MIN_EPOCH" \
NAV_PLATEAU_WINDOW="$PLATEAU_WINDOW" \
NAV_PLATEAU_REL_DROP="$PLATEAU_REL_DROP" \
NAV_CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
python3 "$LEARN_PY"
