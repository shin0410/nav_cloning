#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLIT_PY="$NAV_DIR/scripts/split_dataset_first_n.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
EVAL_PY="$NAV_DIR/scripts/eval_baseline_hour_matrix.py"
PLOT_PY="$NAV_DIR/scripts/plot_baseline_hour_matrix.py"
ANALYZE_GAP_PY="$NAV_DIR/scripts/analyze_baseline_lux_brightness_gap.py"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_baseline_hour_matrix_20260308_0310"
OUT_DIR=""
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_PLOT=1

ALL_TIMES=(
  "20260308_100554"
  "20260308_111130"
  "20260308_120428"
  "20260308_130242"
  "20260308_140206"
  "20260308_150431"
  "20260308_160405"
  "20260310_170214"
  "20260310_180327"
)

usage() {
  cat <<'EOF'
Usage:
  run_hour_matrix_baseline_202603.sh [options]

Options:
  --train-rows N            default: 4000
  --output-time NAME        default: _baseline_hour_matrix_20260308_0310
  --out-dir DIR             default: data/<output-time>
  --skip-unzip              do not extract zip files
  --skip-split              do not rebuild train/test split dirs
  --skip-train              do not retrain baseline models
  --skip-plot               export CSV only
  -h, --help

Pipeline:
  1) unzip 9 datasets
  2) split each dataset: first N rows train / rest test
  3) train one baseline model per time slot
  4) evaluate all 9 models against all 9 test splits (9x9)
  5) export same-vs-different-hour summaries and heatmaps
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-rows)
      TRAIN_ROWS="${2:?missing value for --train-rows}"
      shift 2
      ;;
    --output-time)
      OUTPUT_TIME="${2:?missing value for --output-time}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --skip-unzip)
      RUN_UNZIP=0
      shift
      ;;
    --skip-split)
      RUN_SPLIT=0
      shift
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --skip-plot)
      RUN_PLOT=0
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

for f in "$SPLIT_PY" "$LEARN_PY" "$EVAL_PY" "$PLOT_PY" "$ANALYZE_GAP_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi
EVAL_OUT_DIR="$OUT_DIR/eval"
mkdir -p "$OUT_DIR" "$EVAL_OUT_DIR"

check_dataset_exists() {
  local time_id="$1"
  local ds="$DATA_DIR/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

check_split_exists() {
  local time_id="$1"
  local ds="$DATA_DIR/$time_id/dataset"
  [[ -d "$ds/$TRAIN_IMG_DIR" ]] || { echo "[ERR] missing: $ds/$TRAIN_IMG_DIR" >&2; exit 1; }
  [[ -f "$ds/$TRAIN_VEL_DIR/data.csv" ]] || { echo "[ERR] missing: $ds/$TRAIN_VEL_DIR/data.csv" >&2; exit 1; }
  [[ -d "$ds/$TEST_IMG_DIR" ]] || { echo "[ERR] missing: $ds/$TEST_IMG_DIR" >&2; exit 1; }
  [[ -f "$ds/$TEST_VEL_DIR/data.csv" ]] || { echo "[ERR] missing: $ds/$TEST_VEL_DIR/data.csv" >&2; exit 1; }
}

extract_hour() {
  local time_id="$1"
  echo "${time_id#*_}" | cut -c1-2
}

ZIP_FILES=()
for t in "${ALL_TIMES[@]}"; do
  ZIP_FILES+=("$DATA_DIR/$t.zip")
done

if [[ "$RUN_UNZIP" -eq 1 ]]; then
  echo "[STEP] unzip datasets"
  for z in "${ZIP_FILES[@]}"; do
    [[ -f "$z" ]] || { echo "[ERR] zip not found: $z" >&2; exit 1; }
    unzip -n -q "$z" -d "$DATA_DIR"
  done
else
  echo "[STEP] skip unzip (--skip-unzip)"
fi

for t in "${ALL_TIMES[@]}"; do
  check_dataset_exists "$t"
done

if [[ "$RUN_SPLIT" -eq 1 ]]; then
  echo "[STEP] build train/test split"
  "$PYTHON_BIN" "$SPLIT_PY" \
    --data-root "$DATA_DIR" \
    --times "${ALL_TIMES[@]}" \
    --train-rows "$TRAIN_ROWS" \
    --train-img-dir "$TRAIN_IMG_DIR" \
    --train-vel-dir "$TRAIN_VEL_DIR" \
    --test-img-dir "$TEST_IMG_DIR" \
    --test-vel-dir "$TEST_VEL_DIR" \
    --summary-csv "$OUT_DIR/split_summary.csv"
else
  echo "[STEP] skip split (--skip-split)"
fi

for t in "${ALL_TIMES[@]}"; do
  check_split_exists "$t"
done

EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
with open("$NAV_DIR/config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"
MODEL_DIR="$DATA_DIR/$OUTPUT_TIME/model/$EPOCH"
RESULT_DIR="$DATA_DIR/$OUTPUT_TIME/result/$EPOCH"
MODEL_MAP_CSV="$OUT_DIR/model_map.csv"
echo "train_time,train_hour,model_path" > "$MODEL_MAP_CSV"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[STEP] train one baseline per hour"
  for t in "${ALL_TIMES[@]}"; do
    hour="$(extract_hour "$t")"
    model_name="model_gpu_baseline_hour${hour}_${t}.pt"
    echo "[RUN] train_time=$t hour=$hour model=$model_name"
    NAV_TIME="$t" \
    NAV_TRAIN_TIMES="$t" \
    NAV_OUTPUT_TIME="$OUTPUT_TIME" \
    NAV_LOAD_DATASET_IMG="$TRAIN_IMG_DIR" \
    NAV_LOAD_DATASET_VEL="$TRAIN_VEL_DIR" \
    NAV_SAVE_MODEL="$model_name" \
    "$PYTHON_BIN" "$LEARN_PY"
    echo "$t,$hour,$MODEL_DIR/$model_name" >> "$MODEL_MAP_CSV"
  done
else
  echo "[STEP] skip train (--skip-train)"
  for t in "${ALL_TIMES[@]}"; do
    hour="$(extract_hour "$t")"
    model_name="model_gpu_baseline_hour${hour}_${t}.pt"
    [[ -f "$MODEL_DIR/$model_name" ]] || { echo "[ERR] missing model: $MODEL_DIR/$model_name" >&2; exit 1; }
    echo "$t,$hour,$MODEL_DIR/$model_name" >> "$MODEL_MAP_CSV"
  done
fi

echo "[STEP] evaluate 9x9 baseline matrix"
"$PYTHON_BIN" "$EVAL_PY" \
  --model-map "$MODEL_MAP_CSV" \
  --data-root "$DATA_DIR" \
  --test-times "${ALL_TIMES[@]}" \
  --test-img-dir "$TEST_IMG_DIR" \
  --test-vel-dir "$TEST_VEL_DIR" \
  --out-dir "$EVAL_OUT_DIR"

echo "[STEP] analyze lux and image brightness gap vs error"
"$PYTHON_BIN" "$ANALYZE_GAP_PY" \
  --summary-csv "$EVAL_OUT_DIR/summary_by_train_test.csv" \
  --data-root "$DATA_DIR" \
  --times "${ALL_TIMES[@]}" \
  --train-img-dir "$TRAIN_IMG_DIR" \
  --train-vel-dir "$TRAIN_VEL_DIR" \
  --test-img-dir "$TEST_IMG_DIR" \
  --test-vel-dir "$TEST_VEL_DIR" \
  --out-dir "$EVAL_OUT_DIR"

if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "[STEP] plot hour matrix summaries"
  "$PYTHON_BIN" "$PLOT_PY" --eval-dir "$EVAL_OUT_DIR"
else
  echo "[STEP] skip plot (--skip-plot)"
fi

echo "[DONE] output_time              : $OUTPUT_TIME"
echo "[DONE] model_dir                 : $MODEL_DIR"
echo "[DONE] result_dir                : $RESULT_DIR"
echo "[DONE] split_summary.csv         : $OUT_DIR/split_summary.csv"
echo "[DONE] model_map.csv             : $MODEL_MAP_CSV"
echo "[DONE] summary_by_train_test.csv : $EVAL_OUT_DIR/summary_by_train_test.csv"
echo "[DONE] same_vs_diff.csv          : $EVAL_OUT_DIR/summary_same_vs_diff.csv"
echo "[DONE] hour_gap.csv              : $EVAL_OUT_DIR/summary_by_hour_gap.csv"
echo "[DONE] lux_brightness_gap.csv    : $EVAL_OUT_DIR/mae_with_lux_brightness_gap.csv"
echo "[DONE] tolerance_candidates.csv  : $EVAL_OUT_DIR/tolerance_candidates.csv"
