#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_MAKE="$NAV_DIR/scripts/run_make_4models.sh"
COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"
PLOT_PY="$NAV_DIR/scripts/plot_compare_mae_lines.py"

TARGET_TIME="20260112_14:08:03"
KS="3,5,7,9"
RUN_TRAIN=1
ALLOW_MISSING_MODELS=0
OUT_DIR=""

TEST_TIMES=(
  "20260112_10:05:46"
  "20260112_11:09:30"
  "20260112_12:06:28"
  "20260112_13:09:38"
  "20260112_14:11:47"
  "20260112_15:07:57"
  "20260112_16:19:24"
  "20260112_17:15:12"
  "20260112_18:14:00"
)

usage() {
  cat <<'EOF'
Usage:
  run_train_test_plot_all.sh [options]

Options:
  --target-time TIME            default: 20260112_14:08:03
  --ks LIST                     default: 3,5,7,9
  --out-dir DIR                output root (default: auto under data/)
  --skip-train                 skip training and run compare/plot only
  --allow-missing-models       pass through to compare script
  -h, --help

This script runs all steps:
  1) Train models:
     - baseline(clean9) + augmix/taw/rand K=9 (scenario base4)
     - augmix/taw/rand K=7 + clean2 mix      (scenario k7clean2)
     - augmix/taw/rand K=5 + clean4 mix      (scenario k5clean4)
     - augmix/taw/rand K=3 + clean6 mix      (scenario k3clean6)
  2) Evaluate on test times and export CSV tables.
  3) Plot MAE line charts:
     - all models comparison
     - K comparison per augmentation method
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-time)
      TARGET_TIME="${2:?missing value for --target-time}"
      shift 2
      ;;
    --ks)
      KS="${2:?missing value for --ks}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --allow-missing-models)
      ALLOW_MISSING_MODELS=1
      shift
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

for f in "$RUN_MAKE" "$COMPARE_PY" "$PLOT_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

if [[ -z "$OUT_DIR" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_TARGET="${TARGET_TIME//:/-}"
  OUT_DIR="$NAV_DIR/data/_full_compare_${SAFE_TARGET}_${STAMP}"
fi
mkdir -p "$OUT_DIR"

echo "[INFO] target_time: $TARGET_TIME"
echo "[INFO] out_dir    : $OUT_DIR"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[STEP] training models (base4)"
  bash "$RUN_MAKE" --scenario base4 --time "$TARGET_TIME"

  echo "[STEP] training models (k7clean2)"
  bash "$RUN_MAKE" --scenario k7clean2 --time "$TARGET_TIME"

  echo "[STEP] training models (k5clean4)"
  bash "$RUN_MAKE" --scenario k5clean4 --time "$TARGET_TIME"

  echo "[STEP] training models (k3clean6)"
  bash "$RUN_MAKE" --scenario k3clean6 --time "$TARGET_TIME"
else
  echo "[STEP] skip training (--skip-train)"
fi

echo "[STEP] evaluating models on test times"
cmp_args=(
  "$COMPARE_PY"
  "--target_time" "$TARGET_TIME"
  "--ks" "$KS"
  "--out_dir" "$OUT_DIR"
  "--test_times"
)
for t in "${TEST_TIMES[@]}"; do
  cmp_args+=("$t")
done
if [[ "$ALLOW_MISSING_MODELS" -eq 1 ]]; then
  cmp_args+=("--allow-missing-models")
fi
"$PYTHON_BIN" "${cmp_args[@]}"

echo "[STEP] plotting MAE line charts"
"$PYTHON_BIN" "$PLOT_PY" \
  --summary_csv "$OUT_DIR/summary_by_time_model.csv" \
  --out_dir "$OUT_DIR"

echo "[DONE] all steps completed"
echo "  summary_by_time_model.csv : $OUT_DIR/summary_by_time_model.csv"
echo "  summary_by_model.csv      : $OUT_DIR/summary_by_model.csv"
echo "  detail_per_sample.csv     : $OUT_DIR/detail_per_sample.csv"
echo "  mae_table_all_models.csv  : $OUT_DIR/mae_table_all_models.csv"
echo "  all-models plot           : $OUT_DIR/01_mae_all_models.png"
