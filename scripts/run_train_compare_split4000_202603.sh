#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
CFG="$NAV_DIR/config/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLIT_PY="$NAV_DIR/scripts/split_dataset_first_n.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"
PLOT_PY="$NAV_DIR/scripts/plot_compare_mae_lines.py"
AUGMIX_PY="$NAV_DIR/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
RANDAUG_PY="$NAV_DIR/scripts/edit_dataset/rand_augment.py"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_split4000_20260308_0310"
OUT_DIR=""
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_PLOT=1
KS_CSV="3,5,7,9"
RAND_N=2
RAND_M=9
RAND_P=0.5

TEST_TIMES=(
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
  run_train_compare_split4000_202603.sh [options]

Options:
  --train-rows N            default: 4000
  --ks LIST                 default: 3,5,7,9
  --output-time NAME        model/result save root under data/ (default: _split4000_20260308_0310)
  --out-dir DIR             compare/plot output root (default: data/<output-time>)
  --skip-unzip              do not extract zip files
  --skip-split              do not rebuild train/test split dirs
  --skip-train              do not retrain models
  --skip-plot               export CSV only
  --rand-n N                RandAugment N (default: 2)
  --rand-m N                RandAugment M (default: 9)
  --rand-p P                RandAugment p (default: 0.5)
  -h, --help

Pipeline:
  1) unzip the 20260308/20260310 datasets
  2) split each time: first N rows -> train, remaining rows -> test
  3) train 13 models on the train split:
     baseline + augmix/trivial/rand for K=3,5,7,9
  4) evaluate all models on the test split and export CSV/plots
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-rows)
      TRAIN_ROWS="${2:?missing value for --train-rows}"
      shift 2
      ;;
    --ks)
      KS_CSV="${2:?missing value for --ks}"
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
    --rand-n)
      RAND_N="${2:?missing value for --rand-n}"
      shift 2
      ;;
    --rand-m)
      RAND_M="${2:?missing value for --rand-m}"
      shift 2
      ;;
    --rand-p)
      RAND_P="${2:?missing value for --rand-p}"
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

for f in "$CFG" "$SPLIT_PY" "$LEARN_PY" "$COMPARE_PY" "$PLOT_PY" "$AUGMIX_PY" "$TAW_PY" "$RANDAUG_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi
COMPARE_OUT_DIR="$OUT_DIR/compare"
mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"

parse_ks() {
  local raw="$1"
  local item
  IFS=',' read -r -a KS <<< "$raw"
  if [[ ${#KS[@]} -eq 0 ]]; then
    echo "[ERR] empty --ks" >&2
    exit 1
  fi
  for item in "${KS[@]}"; do
    [[ "$item" =~ ^[0-9]+$ ]] || { echo "[ERR] invalid K: $item" >&2; exit 1; }
  done
}

parse_ks "$KS_CSV"

TIMES_CSV="$(IFS=,; echo "${TEST_TIMES[*]}")"

EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
with open("$CFG", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"

MODEL_DIR="$DATA_DIR/$OUTPUT_TIME/model/$EPOCH"
RESULT_DIR="$DATA_DIR/$OUTPUT_TIME/result/$EPOCH"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT

set_cfg_for_aug() {
  local time_id="$1"
  local in_img="$2"
  local in_vel="$3"
  local out_img="$4"
  local out_vel="$5"

  CFG_PATH="$CFG" TIME_ID="$time_id" IN_IMG="$in_img" IN_VEL="$in_vel" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import yaml

p = os.environ["CFG_PATH"]
with open(p, "r") as f:
    cfg = yaml.safe_load(f)

cfg["time"] = os.environ["TIME_ID"]
cfg["train_times"] = []
cfg["input_aug_dataset_img"] = os.environ["IN_IMG"]
cfg["input_aug_dataset_vel"] = os.environ["IN_VEL"]
cfg["output_aug_dataset_img"] = os.environ["OUT_IMG"]
cfg["output_aug_dataset_vel"] = os.environ["OUT_VEL"]

with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

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

run_aug_for_time() {
  local time_id="$1"
  local name="$2"
  local py="$3"
  local out_img="$4"
  local out_vel="$5"
  local k="$6"
  shift 6
  echo "[RUN] $name time=$time_id K=$k"
  set_cfg_for_aug "$time_id" "$TRAIN_IMG_DIR" "$TRAIN_VEL_DIR" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$py" --K "$k" "$@"
}

run_train_model() {
  local label="$1"
  local load_img="$2"
  local load_vel="$3"
  local save_model="$4"
  echo "[RUN] train $label"
  NAV_OUTPUT_TIME="$OUTPUT_TIME" \
  NAV_TIME="${TEST_TIMES[0]}" \
  NAV_TRAIN_TIMES="$TIMES_CSV" \
  NAV_LOAD_DATASET_IMG="$load_img" \
  NAV_LOAD_DATASET_VEL="$load_vel" \
  NAV_SAVE_MODEL="$save_model" \
  "$PYTHON_BIN" "$LEARN_PY"
}

ZIP_FILES=()
for t in "${TEST_TIMES[@]}"; do
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

for t in "${TEST_TIMES[@]}"; do
  check_dataset_exists "$t"
done

if [[ "$RUN_SPLIT" -eq 1 ]]; then
  echo "[STEP] build train/test split"
  "$PYTHON_BIN" "$SPLIT_PY" \
    --data-root "$DATA_DIR" \
    --times "${TEST_TIMES[@]}" \
    --train-rows "$TRAIN_ROWS" \
    --train-img-dir "$TRAIN_IMG_DIR" \
    --train-vel-dir "$TRAIN_VEL_DIR" \
    --test-img-dir "$TEST_IMG_DIR" \
    --test-vel-dir "$TEST_VEL_DIR" \
    --summary-csv "$OUT_DIR/split_summary.csv"
else
  echo "[STEP] skip split (--skip-split)"
fi

for t in "${TEST_TIMES[@]}"; do
  check_split_exists "$t"
done

printf "%s\n" "${TEST_TIMES[@]}" > "$OUT_DIR/train_times_used.txt"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[STEP] train baseline"
  run_train_model \
    "baseline(clean train split)" \
    "$TRAIN_IMG_DIR" \
    "$TRAIN_VEL_DIR" \
    "model_gpu_baseline_split4000_20260308_0310.pt"

  for k in "${KS[@]}"; do
    augmix_img="augmix_${TRAIN_IMG_DIR}_K${k}"
    augmix_vel="augmix_${TRAIN_VEL_DIR}_K${k}"
    taw_img="TrivialAugmentWide_${TRAIN_IMG_DIR}_K${k}"
    taw_vel="TrivialAugmentWide_${TRAIN_VEL_DIR}_K${k}"
    rand_img="randaugment_${TRAIN_IMG_DIR}_K${k}"
    rand_vel="randaugment_${TRAIN_VEL_DIR}_K${k}"

    echo "[STEP] build augmented datasets for K=$k"
    for t in "${TEST_TIMES[@]}"; do
      run_aug_for_time "$t" "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$k"
      run_aug_for_time "$t" "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$k" --same3view
      run_aug_for_time "$t" "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$k" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views
    done

    echo "[STEP] train augmented models for K=$k"
    run_train_model \
      "augmix(K=$k)" \
      "$augmix_img" \
      "$augmix_vel" \
      "model_gpu_augmix_K${k}_split4000_20260308_0310.pt"

    run_train_model \
      "TrivialAugmentWide(K=$k)" \
      "$taw_img" \
      "$taw_vel" \
      "model_gpu_trivialaugwide_K${k}_split4000_20260308_0310.pt"

    run_train_model \
      "rand_augment(K=$k)" \
      "$rand_img" \
      "$rand_vel" \
      "model_gpu_randaugment_K${k}_split4000_20260308_0310.pt"
  done
else
  echo "[STEP] skip train (--skip-train)"
fi

[[ -d "$MODEL_DIR" ]] || { echo "[ERR] model dir not found: $MODEL_DIR" >&2; exit 1; }

echo "[STEP] evaluate models on test split"
"$PYTHON_BIN" "$COMPARE_PY" \
  --model_dir "$MODEL_DIR" \
  --ks "$KS_CSV" \
  --out_dir "$COMPARE_OUT_DIR" \
  --test-img-dir "$TEST_IMG_DIR" \
  --test-vel-dir "$TEST_VEL_DIR" \
  --test_times "${TEST_TIMES[@]}"

if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "[STEP] plot MAE charts"
  "$PYTHON_BIN" "$PLOT_PY" \
    --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
    --out_dir "$COMPARE_OUT_DIR"
else
  echo "[STEP] skip plot (--skip-plot)"
fi

echo "[DONE] output_time        : $OUTPUT_TIME"
echo "[DONE] model_dir           : $MODEL_DIR"
echo "[DONE] tensorboard result  : $RESULT_DIR"
echo "[DONE] split_summary.csv   : $OUT_DIR/split_summary.csv"
echo "[DONE] summary_by_time     : $COMPARE_OUT_DIR/summary_by_time_model.csv"
echo "[DONE] summary_by_model    : $COMPARE_OUT_DIR/summary_by_model.csv"
echo "[DONE] detail_per_sample   : $COMPARE_OUT_DIR/detail_per_sample.csv"
