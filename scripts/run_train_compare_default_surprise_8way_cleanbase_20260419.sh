#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
CFG="$NAV_DIR/config/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

AUGMIX_PY="$NAV_DIR/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
RANDAUG_PY="$NAV_DIR/scripts/edit_dataset/rand_augment.py"
MERGE_PY="$NAV_DIR/pipeline_scripts/merge_dataset.py"
LEARN_DEFAULT_PY="$NAV_DIR/scripts/learning_default.py"
LEARN_SURPRISE_PY="$NAV_DIR/scripts/learning_surprise.py"
COMPARE_NAMED_PY="$NAV_DIR/scripts/compare_named_models_on_test_times.py"
PLOT_PY="$NAV_DIR/scripts/plot_compare_mae_lines.py"
SUMMARY_PY="$NAV_DIR/scripts/summarize_default_surprise_8way.py"

SOURCE_TIME="20260419_merged_clean_6200_6150_6200"
EPOCHS=100
BATCH_SIZE=8
AUG_K=3
TAW_M_BINS=20
RAND_N=2
RAND_M=9
RAND_P=0.5
TEST_FIRST_N=6100
RUN_PREP=1
RUN_TRAIN=1
RUN_COMPARE=1
DRY_RUN=0

SRC1_TIME="20260315_13:41:50"
SRC1_ROWS=6200
SRC2_TIME="20260316_13:02:58"
SRC2_ROWS=6150
SRC3_TIME="20260319_16:18:42"
SRC3_ROWS=6200

TEST_TIMES=(
  "20260315_11:06:52"
  "20260319_17:02:53"
  "20260316_13:45:21"
)

usage() {
  cat <<'EOF'
Usage:
  run_train_compare_default_surprise_8way_cleanbase_20260419.sh [options]

Options:
  --source-time TIME        default: 20260419_merged_clean_6200_6150_6200
  --epochs N                default: 100
  --batch-size N            default: 8
  --aug-k N                 default: 3
  --taw-m-bins N            default: 20
  --rand-n N                default: 2
  --rand-m N                default: 9
  --rand-p P                default: 0.5
  --test-first-n N          default: 6100
  --skip-prep               do not rebuild augmentation datasets / mix datasets
  --skip-train              do not retrain the 8 models
  --skip-compare            do not run evaluation/summary
  --dry-run                 print commands only
  -h, --help

This script first builds a clean-only merged training dataset from:
  20260315_13:41:50 first 6200 rows
  20260316_13:02:58 first 6150 rows
  20260319_16:18:42 first 6200 rows

Then it compares 8 runs on that clean base:
  clean / AugMix / TrivialAugmentWide / RandAugment
  x default / surprise
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-time)
      SOURCE_TIME="${2:?missing value for --source-time}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?missing value for --epochs}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --aug-k)
      AUG_K="${2:?missing value for --aug-k}"
      shift 2
      ;;
    --taw-m-bins)
      TAW_M_BINS="${2:?missing value for --taw-m-bins}"
      shift 2
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
    --test-first-n)
      TEST_FIRST_N="${2:?missing value for --test-first-n}"
      shift 2
      ;;
    --skip-prep)
      RUN_PREP=0
      shift
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --skip-compare)
      RUN_COMPARE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

for f in "$CFG" "$AUGMIX_PY" "$TAW_PY" "$RANDAUG_PY" "$MERGE_PY" "$LEARN_DEFAULT_PY" "$LEARN_SURPRISE_PY" "$COMPARE_NAMED_PY" "$PLOT_PY" "$SUMMARY_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

for t in "$SRC1_TIME" "$SRC2_TIME" "$SRC3_TIME"; do
  [[ -d "$DATA_DIR/$t/dataset/img" ]] || { echo "[ERR] missing raw img dir: $DATA_DIR/$t/dataset/img" >&2; exit 1; }
  [[ -f "$DATA_DIR/$t/dataset/vel/data.csv" ]] || { echo "[ERR] missing raw csv: $DATA_DIR/$t/dataset/vel/data.csv" >&2; exit 1; }
done

BASE_DATASET_DIR="$DATA_DIR/$SOURCE_TIME/dataset"

RAND_P_TAG="$(printf '%s' "$RAND_P" | tr -d '.')"
AUGMIX_IMG="AugMix_img_K${AUG_K}"
AUGMIX_VEL="AugMix_vel_K${AUG_K}"
MIX_AUGMIX_IMG="mix_augmix_img_K${AUG_K}"
MIX_AUGMIX_VEL="mix_augmix_vel_K${AUG_K}"
TAW_IMG="TrivialAugmentWide_img_K${AUG_K}_M${TAW_M_BINS}"
TAW_VEL="TrivialAugmentWide_vel_K${AUG_K}_M${TAW_M_BINS}"
MIX_TAW_IMG="mix_taw_img_K${AUG_K}_M${TAW_M_BINS}"
MIX_TAW_VEL="mix_taw_vel_K${AUG_K}_M${TAW_M_BINS}"
RANDAUG_IMG="RandAugment_img_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}"
RANDAUG_VEL="RandAugment_vel_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}"
MIX_RANDAUG_IMG="mix_randaugment_img_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}"
MIX_RANDAUG_VEL="mix_randaugment_vel_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}"

DEFAULT_MODEL_DIR="$DATA_DIR/$SOURCE_TIME/model/default/$EPOCHS"
SURPRISE_MODEL_DIR="$DATA_DIR/$SOURCE_TIME/model/$EPOCHS"

DEFAULT_CLEAN_MODEL="model_gpu_default_clean_rows6200_6150_6200.pt"
SURPRISE_CLEAN_MODEL="model_gpu_surprise_clean_rows6200_6150_6200.pt"
DEFAULT_AUGMIX_MODEL="model_gpu_default_augmix_K${AUG_K}_rows6200_6150_6200.pt"
SURPRISE_AUGMIX_MODEL="model_gpu_surprise_augmix_K${AUG_K}_rows6200_6150_6200.pt"
DEFAULT_TAW_MODEL="model_gpu_default_taw_K${AUG_K}_M${TAW_M_BINS}_rows6200_6150_6200.pt"
SURPRISE_TAW_MODEL="model_gpu_surprise_taw_K${AUG_K}_M${TAW_M_BINS}_rows6200_6150_6200.pt"
DEFAULT_RANDAUG_MODEL="model_gpu_default_randaugment_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_rows6200_6150_6200.pt"
SURPRISE_RANDAUG_MODEL="model_gpu_surprise_randaugment_K${AUG_K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_rows6200_6150_6200.pt"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"
restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT

run_cmd() {
  echo "[RUN] $*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

ensure_clean_merged_dataset() {
  if [[ -d "$BASE_DATASET_DIR/img" && -f "$BASE_DATASET_DIR/vel/data.csv" ]]; then
    echo "[INFO] reuse existing clean merged dataset: $SOURCE_TIME"
    return 0
  fi

  echo "[STEP] build clean merged dataset: $SOURCE_TIME"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "      build from $SRC1_TIME:$SRC1_ROWS, $SRC2_TIME:$SRC2_ROWS, $SRC3_TIME:$SRC3_ROWS"
    return 0
  fi

  DATA_DIR_ENV="$DATA_DIR" SOURCE_TIME_ENV="$SOURCE_TIME" \
  SRC1_TIME_ENV="$SRC1_TIME" SRC1_ROWS_ENV="$SRC1_ROWS" \
  SRC2_TIME_ENV="$SRC2_TIME" SRC2_ROWS_ENV="$SRC2_ROWS" \
  SRC3_TIME_ENV="$SRC3_TIME" SRC3_ROWS_ENV="$SRC3_ROWS" \
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path
import pandas as pd

data_dir = Path(os.environ['DATA_DIR_ENV'])
out_time = os.environ['SOURCE_TIME_ENV']
out_ds = data_dir / out_time / 'dataset'
out_img = out_ds / 'img'
out_vel = out_ds / 'vel'
out_csv = out_vel / 'data.csv'

specs = [
    ('d1', os.environ['SRC1_TIME_ENV'], int(os.environ['SRC1_ROWS_ENV'])),
    ('d2', os.environ['SRC2_TIME_ENV'], int(os.environ['SRC2_ROWS_ENV'])),
    ('d3', os.environ['SRC3_TIME_ENV'], int(os.environ['SRC3_ROWS_ENV'])),
]

if out_ds.exists():
    shutil.rmtree(out_ds)
out_img.mkdir(parents=True, exist_ok=True)
out_vel.mkdir(parents=True, exist_ok=True)

rows = []
missing = 0
for label, time_id, limit in specs:
    src_ds = data_dir / time_id / 'dataset'
    src_img = src_ds / 'img'
    src_csv = src_ds / 'vel' / 'data.csv'
    df = pd.read_csv(src_csv).head(limit).copy()
    safe_time = time_id.replace(':', '-')
    for _, r in df.iterrows():
        ep = str(r['episode']).split('.')[0]
        new_ep = f'{label}_{safe_time}_{ep}'
        ok = True
        for view in ('center', 'left', 'right'):
            src = src_img / f'{ep}_{view}.npy'
            dst = out_img / f'{new_ep}_{view}.npy'
            if not src.is_file():
                ok = False
                break
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        if not ok:
            missing += 1
            continue
        rec = r.to_dict()
        rec['episode'] = new_ep
        rows.append(rec)

pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f'[DONE] clean merged rows={len(rows)} missing={missing}')
print(f'[DONE] out dataset: {out_ds}')
PY
}

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

ensure_aug_dataset() {
  local kind="$1"
  local img_dirname="$2"
  local vel_dirname="$3"

  local img_path="$BASE_DATASET_DIR/$img_dirname"
  local csv_path="$BASE_DATASET_DIR/$vel_dirname/data.csv"
  if [[ -d "$img_path" && -f "$csv_path" ]]; then
    echo "[INFO] reuse existing $kind dataset: $img_dirname / $vel_dirname"
    return 0
  fi

  case "$kind" in
    augmix)
      set_cfg_for_aug "$SOURCE_TIME" "img" "vel" "$img_dirname" "$vel_dirname"
      run_cmd "$PYTHON_BIN" "$AUGMIX_PY" --K "$AUG_K"
      ;;
    taw)
      set_cfg_for_aug "$SOURCE_TIME" "img" "vel" "$img_dirname" "$vel_dirname"
      run_cmd "$PYTHON_BIN" "$TAW_PY" --K "$AUG_K" --same3view --num-magnitude-bins "$TAW_M_BINS"
      ;;
    randaugment)
      set_cfg_for_aug "$SOURCE_TIME" "img" "vel" "$img_dirname" "$vel_dirname"
      run_cmd "$PYTHON_BIN" "$RANDAUG_PY" --K "$AUG_K" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views
      ;;
    *)
      echo "[ERR] unknown augmentation kind: $kind" >&2
      exit 1
      ;;
  esac
}

ensure_mix_dataset() {
  local kind="$1"
  local aug_img="$2"
  local aug_vel="$3"
  local out_img="$4"
  local out_vel="$5"

  local out_img_path="$BASE_DATASET_DIR/$out_img"
  local out_csv_path="$BASE_DATASET_DIR/$out_vel/data.csv"
  if [[ -d "$out_img_path" && -f "$out_csv_path" ]]; then
    echo "[INFO] reuse existing mixed $kind dataset: $out_img / $out_vel"
    return 0
  fi

  run_cmd "$PYTHON_BIN" "$MERGE_PY" \
    --orig_img "$BASE_DATASET_DIR/img" \
    --orig_csv "$BASE_DATASET_DIR/vel/data.csv" \
    --aug_img "$BASE_DATASET_DIR/$aug_img" \
    --aug_csv "$BASE_DATASET_DIR/$aug_vel/data.csv" \
    --out_img "$out_img_path" \
    --out_csv "$out_csv_path" \
    --mode symlink
}

train_default_model() {
  local label="$1"
  local load_img="$2"
  local load_vel="$3"
  local save_model="$4"
  local out_path="$DEFAULT_MODEL_DIR/$save_model"
  if [[ -f "$out_path" ]]; then
    echo "[INFO] skip default $label (already exists): $out_path"
    return 0
  fi
  echo "[RUN] train default $label"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "      NAV_TRAIN_TIMES=$SOURCE_TIME $PYTHON_BIN $LEARN_DEFAULT_PY $SOURCE_TIME $EPOCHS --data_root $DATA_DIR --batch_size $BATCH_SIZE --load_dataset_img $load_img --load_dataset_vel $load_vel --save_model $save_model"
    return 0
  fi
  NAV_TRAIN_TIMES="$SOURCE_TIME" \
  "$PYTHON_BIN" "$LEARN_DEFAULT_PY" "$SOURCE_TIME" "$EPOCHS" \
    --data_root "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --load_dataset_img "$load_img" \
    --load_dataset_vel "$load_vel" \
    --save_model "$save_model"
}

train_surprise_model() {
  local label="$1"
  local load_img="$2"
  local load_vel="$3"
  local save_model="$4"
  local out_path="$SURPRISE_MODEL_DIR/$save_model"
  if [[ -f "$out_path" ]]; then
    echo "[INFO] skip surprise $label (already exists): $out_path"
    return 0
  fi
  echo "[RUN] train surprise $label"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "      NAV_TIME=$SOURCE_TIME NAV_OUTPUT_TIME=$SOURCE_TIME NAV_TRAIN_TIMES=$SOURCE_TIME NAV_LOAD_DATASET_IMG=$load_img NAV_LOAD_DATASET_VEL=$load_vel NAV_EPOCH=$EPOCHS NAV_BATCH_SIZE=$BATCH_SIZE NAV_SAVE_MODEL=$save_model $PYTHON_BIN $LEARN_SURPRISE_PY"
    return 0
  fi
  NAV_TIME="$SOURCE_TIME" \
  NAV_OUTPUT_TIME="$SOURCE_TIME" \
  NAV_TRAIN_TIMES="$SOURCE_TIME" \
  NAV_LOAD_DATASET_IMG="$load_img" \
  NAV_LOAD_DATASET_VEL="$load_vel" \
  NAV_EPOCH="$EPOCHS" \
  NAV_BATCH_SIZE="$BATCH_SIZE" \
  NAV_SAVE_MODEL="$save_model" \
  "$PYTHON_BIN" "$LEARN_SURPRISE_PY"
}

ensure_clean_merged_dataset

if [[ "$RUN_PREP" -eq 1 ]]; then
  echo "[STEP] prepare augmentation datasets"
  ensure_aug_dataset augmix "$AUGMIX_IMG" "$AUGMIX_VEL"
  ensure_aug_dataset taw "$TAW_IMG" "$TAW_VEL"
  ensure_aug_dataset randaugment "$RANDAUG_IMG" "$RANDAUG_VEL"

  echo "[STEP] prepare mixed datasets"
  ensure_mix_dataset augmix "$AUGMIX_IMG" "$AUGMIX_VEL" "$MIX_AUGMIX_IMG" "$MIX_AUGMIX_VEL"
  ensure_mix_dataset taw "$TAW_IMG" "$TAW_VEL" "$MIX_TAW_IMG" "$MIX_TAW_VEL"
  ensure_mix_dataset randaugment "$RANDAUG_IMG" "$RANDAUG_VEL" "$MIX_RANDAUG_IMG" "$MIX_RANDAUG_VEL"
fi

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[STEP] train 8 models"
  train_default_model clean img vel "$DEFAULT_CLEAN_MODEL"
  train_surprise_model clean img vel "$SURPRISE_CLEAN_MODEL"

  train_default_model augmix "$MIX_AUGMIX_IMG" "$MIX_AUGMIX_VEL" "$DEFAULT_AUGMIX_MODEL"
  train_surprise_model augmix "$MIX_AUGMIX_IMG" "$MIX_AUGMIX_VEL" "$SURPRISE_AUGMIX_MODEL"

  train_default_model taw "$MIX_TAW_IMG" "$MIX_TAW_VEL" "$DEFAULT_TAW_MODEL"
  train_surprise_model taw "$MIX_TAW_IMG" "$MIX_TAW_VEL" "$SURPRISE_TAW_MODEL"

  train_default_model randaugment "$MIX_RANDAUG_IMG" "$MIX_RANDAUG_VEL" "$DEFAULT_RANDAUG_MODEL"
  train_surprise_model randaugment "$MIX_RANDAUG_IMG" "$MIX_RANDAUG_VEL" "$SURPRISE_RANDAUG_MODEL"
fi

if [[ "$RUN_COMPARE" -eq 1 ]]; then
  echo "[STEP] compare 8 models"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  COMPARE_OUT_DIR="$DATA_DIR/_compare_8way_default_surprise_${SOURCE_TIME//:/-}_e${EPOCHS}_${STAMP}"
  mkdir -p "$COMPARE_OUT_DIR"

  MODELS_CSV="$COMPARE_OUT_DIR/models_used.csv"
  cat > "$MODELS_CSV" <<EOF
model,model_path
default_clean,$DEFAULT_MODEL_DIR/$DEFAULT_CLEAN_MODEL
surprise_clean,$SURPRISE_MODEL_DIR/$SURPRISE_CLEAN_MODEL
default_augmix_k${AUG_K},$DEFAULT_MODEL_DIR/$DEFAULT_AUGMIX_MODEL
surprise_augmix_k${AUG_K},$SURPRISE_MODEL_DIR/$SURPRISE_AUGMIX_MODEL
default_taw_k${AUG_K}_m${TAW_M_BINS},$DEFAULT_MODEL_DIR/$DEFAULT_TAW_MODEL
surprise_taw_k${AUG_K}_m${TAW_M_BINS},$SURPRISE_MODEL_DIR/$SURPRISE_TAW_MODEL
default_randaugment_k${AUG_K}_n${RAND_N}_m${RAND_M}_p${RAND_P_TAG},$DEFAULT_MODEL_DIR/$DEFAULT_RANDAUG_MODEL
surprise_randaugment_k${AUG_K}_n${RAND_N}_m${RAND_M}_p${RAND_P_TAG},$SURPRISE_MODEL_DIR/$SURPRISE_RANDAUG_MODEL
EOF

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[RUN] $PYTHON_BIN $COMPARE_NAMED_PY --nav_dir $NAV_DIR --config $CFG --models_csv $MODELS_CSV --test-first-n $TEST_FIRST_N --out_dir $COMPARE_OUT_DIR --test_times ${TEST_TIMES[*]}"
    echo "[RUN] $PYTHON_BIN $PLOT_PY --summary_csv $COMPARE_OUT_DIR/summary_by_time_model.csv --out_dir $COMPARE_OUT_DIR/plots"
    echo "[RUN] $PYTHON_BIN $SUMMARY_PY --summary_csv $COMPARE_OUT_DIR/summary_by_time_model.csv --out_dir $COMPARE_OUT_DIR/summary_default_surprise"
  else
    "$PYTHON_BIN" "$COMPARE_NAMED_PY" \
      --nav_dir "$NAV_DIR" \
      --config "$CFG" \
      --models_csv "$MODELS_CSV" \
      --test-first-n "$TEST_FIRST_N" \
      --out_dir "$COMPARE_OUT_DIR" \
      --test_times "${TEST_TIMES[@]}"

    "$PYTHON_BIN" "$PLOT_PY" \
      --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
      --out_dir "$COMPARE_OUT_DIR/plots"

    "$PYTHON_BIN" "$SUMMARY_PY" \
      --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
      --out_dir "$COMPARE_OUT_DIR/summary_default_surprise"
  fi

  echo "[DONE] compare_out_dir: $COMPARE_OUT_DIR"
fi
