#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
PYTHON_BIN="${PYTHON_BIN:-python3}"

COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"
PLOT_PY="$NAV_DIR/scripts/plot_compare_mae_lines.py"

TARGET_TIME="20260112_14:08:03"
KS="3,5,7,9"
RUN_UNZIP=1
RUN_PLOT=1
ALLOW_MISSING_MODELS=0
OUT_DIR=""

K7_CLEAN_TIME_1="20260112_10:01:45"
K7_CLEAN_TIME_2="20260112_18:10:41"
K5_CLEAN_TIME_1="20260112_10:01:45"
K5_CLEAN_TIME_2="20260112_11:02:15"
K5_CLEAN_TIME_3="20260112_17:11:48"
K5_CLEAN_TIME_4="20260112_18:10:41"
K3_CLEAN_TIME_1="20260112_10:01:45"
K3_CLEAN_TIME_2="20260112_11:02:15"
K3_CLEAN_TIME_3="20260112_12:03:40"
K3_CLEAN_TIME_4="20260112_16:15:51"
K3_CLEAN_TIME_5="20260112_17:11:48"
K3_CLEAN_TIME_6="20260112_18:10:41"

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
  run_compare_20260308_0310_testsets.sh [options]

Options:
  --target-time TIME          model source time (default: 20260112_14:08:03)
  --ks LIST                   default: 3,5,7,9
  --out-dir DIR               output root (default: auto under data/)
  --skip-unzip                do not extract zip files before compare
  --skip-plot                 export CSV only
  --allow-missing-models      pass through to compare script
  -h, --help

This script runs:
  1) Extract 20260308/20260310 zip datasets into nav_cloning/data/
  2) Compare the standard 13 models on those test times:
     baseline(clean9) + augmix/taw/rand K=3,5,7,9
  3) Export CSV summaries (and plot if not skipped)
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
    --skip-unzip)
      RUN_UNZIP=0
      shift
      ;;
    --skip-plot)
      RUN_PLOT=0
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

for f in "$COMPARE_PY" "$PLOT_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

ZIP_FILES=()
for t in "${TEST_TIMES[@]}"; do
  ZIP_FILES+=("$DATA_DIR/$t.zip")
done

EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
p = Path("$NAV_DIR") / "config" / "config.yaml"
with p.open("r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"

check_dataset_exists() {
  local time_id="$1"
  local ds="$DATA_DIR/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

if [[ "$RUN_UNZIP" -eq 1 ]]; then
  echo "[STEP] extracting zip datasets"
  for z in "${ZIP_FILES[@]}"; do
    [[ -f "$z" ]] || { echo "[ERR] zip not found: $z" >&2; exit 1; }
    echo "  unzip -n $(basename "$z")"
    unzip -n -q "$z" -d "$DATA_DIR"
  done
else
  echo "[STEP] skip unzip (--skip-unzip)"
fi

for t in "${TEST_TIMES[@]}"; do
  check_dataset_exists "$t"
done

if [[ -z "$OUT_DIR" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_TARGET="${TARGET_TIME//:/-}"
  OUT_DIR="$DATA_DIR/_compare_20260308_0310_${SAFE_TARGET}_${STAMP}"
fi
mkdir -p "$OUT_DIR"

MODEL_DIR="$DATA_DIR/$TARGET_TIME/model/$EPOCH"
[[ -d "$MODEL_DIR" ]] || { echo "[ERR] model dir not found: $MODEL_DIR" >&2; exit 1; }

SAFE_TARGET="${TARGET_TIME//:/-}"
SAFE_K7_C1="${K7_CLEAN_TIME_1//:/-}"
SAFE_K7_C2="${K7_CLEAN_TIME_2//:/-}"
SAFE_K5_C1="${K5_CLEAN_TIME_1//:/-}"
SAFE_K5_C2="${K5_CLEAN_TIME_2//:/-}"
SAFE_K5_C3="${K5_CLEAN_TIME_3//:/-}"
SAFE_K5_C4="${K5_CLEAN_TIME_4//:/-}"
SAFE_K3_C1="${K3_CLEAN_TIME_1//:/-}"
SAFE_K3_C2="${K3_CLEAN_TIME_2//:/-}"
SAFE_K3_C3="${K3_CLEAN_TIME_3//:/-}"
SAFE_K3_C4="${K3_CLEAN_TIME_4//:/-}"
SAFE_K3_C5="${K3_CLEAN_TIME_5//:/-}"
SAFE_K3_C6="${K3_CLEAN_TIME_6//:/-}"

resolve_standard_model() {
  local key="$1"
  case "$key" in
    baseline)
      echo "$MODEL_DIR/model_gpu_baseline_allclean9_${SAFE_TARGET}.pt"
      ;;
    augmix_K9)
      echo "$MODEL_DIR/model_gpu_augmix_K9_${SAFE_TARGET}.pt"
      ;;
    TrivialAugmentWide_K9)
      echo "$MODEL_DIR/model_gpu_trivialaugwide_K9_${SAFE_TARGET}.pt"
      ;;
    rand_augment_K9)
      echo "$MODEL_DIR/model_gpu_randaugment_K9_${SAFE_TARGET}.pt"
      ;;
    augmix_K7)
      echo "$MODEL_DIR/model_gpu_augmix_K7_plusclean2_${SAFE_TARGET}_${SAFE_K7_C1}_${SAFE_K7_C2}.pt"
      ;;
    TrivialAugmentWide_K7)
      echo "$MODEL_DIR/model_gpu_trivialaugwide_K7_plusclean2_${SAFE_TARGET}_${SAFE_K7_C1}_${SAFE_K7_C2}.pt"
      ;;
    rand_augment_K7)
      echo "$MODEL_DIR/model_gpu_randaugment_K7_plusclean2_${SAFE_TARGET}_${SAFE_K7_C1}_${SAFE_K7_C2}.pt"
      ;;
    augmix_K5)
      echo "$MODEL_DIR/model_gpu_augmix_K5_plusclean4_${SAFE_TARGET}_${SAFE_K5_C1}_${SAFE_K5_C2}_${SAFE_K5_C3}_${SAFE_K5_C4}.pt"
      ;;
    TrivialAugmentWide_K5)
      echo "$MODEL_DIR/model_gpu_trivialaugwide_K5_plusclean4_${SAFE_TARGET}_${SAFE_K5_C1}_${SAFE_K5_C2}_${SAFE_K5_C3}_${SAFE_K5_C4}.pt"
      ;;
    rand_augment_K5)
      echo "$MODEL_DIR/model_gpu_randaugment_K5_plusclean4_${SAFE_TARGET}_${SAFE_K5_C1}_${SAFE_K5_C2}_${SAFE_K5_C3}_${SAFE_K5_C4}.pt"
      ;;
    augmix_K3)
      echo "$MODEL_DIR/model_gpu_augmix_K3_plusclean6_${SAFE_TARGET}_${SAFE_K3_C1}_${SAFE_K3_C2}_${SAFE_K3_C3}_${SAFE_K3_C4}_${SAFE_K3_C5}_${SAFE_K3_C6}.pt"
      ;;
    TrivialAugmentWide_K3)
      echo "$MODEL_DIR/model_gpu_trivialaugwide_K3_plusclean6_${SAFE_TARGET}_${SAFE_K3_C1}_${SAFE_K3_C2}_${SAFE_K3_C3}_${SAFE_K3_C4}_${SAFE_K3_C5}_${SAFE_K3_C6}.pt"
      ;;
    rand_augment_K3)
      echo "$MODEL_DIR/model_gpu_randaugment_K3_plusclean6_${SAFE_TARGET}_${SAFE_K3_C1}_${SAFE_K3_C2}_${SAFE_K3_C3}_${SAFE_K3_C4}_${SAFE_K3_C5}_${SAFE_K3_C6}.pt"
      ;;
    *)
      echo "[ERR] unknown model key: $key" >&2
      exit 1
      ;;
  esac
}

MODEL_ALIAS_DIR="$OUT_DIR/model_alias_standard"
rm -rf "$MODEL_ALIAS_DIR"
mkdir -p "$MODEL_ALIAS_DIR"

MODEL_MAP_CSV="$OUT_DIR/models_selected.csv"
echo "label,real_model_path,alias_model_path" > "$MODEL_MAP_CSV"

link_model() {
  local label="$1"
  local alias_name="$2"
  local real_path
  real_path="$(resolve_standard_model "$label")"
  if [[ ! -f "$real_path" ]]; then
    if [[ "$ALLOW_MISSING_MODELS" -eq 1 ]]; then
      echo "[WARN] missing model: $real_path"
      return 0
    fi
    echo "[ERR] missing model: $real_path" >&2
    exit 1
  fi
  local alias_path="$MODEL_ALIAS_DIR/$alias_name"
  ln -sfn "$real_path" "$alias_path"
  echo "$label,$real_path,$alias_path" >> "$MODEL_MAP_CSV"
}

echo "[STEP] selecting standard model set"
link_model baseline model_gpu_baseline_selected.pt
link_model augmix_K3 model_gpu_augmix_K3_selected.pt
link_model augmix_K5 model_gpu_augmix_K5_selected.pt
link_model augmix_K7 model_gpu_augmix_K7_selected.pt
link_model augmix_K9 model_gpu_augmix_K9_selected.pt
link_model TrivialAugmentWide_K3 model_gpu_trivialaugwide_K3_selected.pt
link_model TrivialAugmentWide_K5 model_gpu_trivialaugwide_K5_selected.pt
link_model TrivialAugmentWide_K7 model_gpu_trivialaugwide_K7_selected.pt
link_model TrivialAugmentWide_K9 model_gpu_trivialaugwide_K9_selected.pt
link_model rand_augment_K3 model_gpu_randaugment_K3_selected.pt
link_model rand_augment_K5 model_gpu_randaugment_K5_selected.pt
link_model rand_augment_K7 model_gpu_randaugment_K7_selected.pt
link_model rand_augment_K9 model_gpu_randaugment_K9_selected.pt

echo "[STEP] evaluating models on 20260308/20260310 test times"
cmp_args=(
  "$COMPARE_PY"
  "--target_time" "$TARGET_TIME"
  "--model_dir" "$MODEL_ALIAS_DIR"
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

if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "[STEP] plotting MAE line charts"
  "$PYTHON_BIN" "$PLOT_PY" \
    --summary_csv "$OUT_DIR/summary_by_time_model.csv" \
    --out_dir "$OUT_DIR"
else
  echo "[STEP] skip plot (--skip-plot)"
fi

echo "[DONE] compare completed"
echo "  summary_by_time_model.csv : $OUT_DIR/summary_by_time_model.csv"
echo "  summary_by_model.csv      : $OUT_DIR/summary_by_model.csv"
echo "  detail_per_sample.csv     : $OUT_DIR/detail_per_sample.csv"
echo "  models_selected.csv       : $MODEL_MAP_CSV"
echo "  models_used.csv           : $OUT_DIR/models_used.csv"
if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "  mae_table_all_models.csv  : $OUT_DIR/mae_table_all_models.csv"
  echo "  all-models plot           : $OUT_DIR/01_mae_all_models.png"
fi
