#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$NAV_DIR/config/config.yaml"

AUGMIX_PY="$NAV_DIR/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
RANDAUG_PY="$NAV_DIR/scripts/edit_dataset/rand_augment.py"
MERGE_PY="$NAV_DIR/pipeline_scripts/merge_dataset.py"
LEARN_DEFAULT_PY="$NAV_DIR/scripts/learning_default.py"
COMPARE_NAMED_PY="$NAV_DIR/scripts/compare_named_models_on_test_times.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TRAIN_TIME="20260426_13_39_33"
TEST_TIMES_CSV="20260428_12:31:05,20260428_16:50:47,20260428_19:29:22"
OPS_CSV="Equalize,Brightness,AutoContrast"
K=3
TAW_MAG=31
RAND_N=2
RAND_M=9
RAND_P=0.5
SEED=1234
TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=128
EPOCH=""
VIEW="center"
TEST_FIRST_N="6100"
OUT_ID=""
RUN_PREP=1
RUN_TRAIN=1
RUN_EVAL=1

usage() {
  cat <<'EOF'
Usage:
  run_3op_aug_compare_outdoor_20260426.sh [options]

Default flow:
  train data : 20260426_13_39_33
  test data  : 20260428_12:31:05, 20260428_16:50:47, 20260428_19:29:22
  ops        : Equalize,Brightness,AutoContrast only
  methods    : Clean, AugMix, TrivialAugmentWide, RandAugment
  K          : 3
  training   : learning_default.py (MSE), using center/left/right cameras
  train set  : clean original, plus clean original + augmented data for each augmentation method
  eval view  : center

Options:
  --train-time TIME          default: 20260426_13_39_33
  --test-times CSV           default: 20260428_12:31:05,20260428_16:50:47,20260428_19:29:22
  --ops CSV                  default: Equalize,Brightness,AutoContrast
  --k N                      default: 3
  --taw-mag N                default: 31
  --rand-n N                 default: 2
  --rand-m N                 default: 9
  --rand-p P                 default: 0.5
  --seed N                   default: 1234
  --epoch N                  default: read nav_cloning/config/config.yaml
  --batch-size N             default: read nav_cloning/config/config.yaml
  --view center|left|right   default: center
  --test-first-n N|all       default: 6100
  --eval-batch-size N        default: 128
  --out-id NAME              output folder name under nav_cloning/data/
  --skip-prep                skip augmentation and clean+aug dataset creation
  --skip-train               skip training and evaluate existing models
  --skip-eval                skip evaluation
  -h, --help

Outputs:
  nav_cloning/data/<out-id>/
    dataset/clean_* and dataset/mix_*
    model/default/<epoch>/*.pt
    eval/summary_by_time_model.csv
    eval/summary_by_model.csv
    eval/mae_table_aug_method_compare.csv
    eval/summary_aug_method_ranked.csv
    eval/mae_aug_method_compare.png
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-time)
      TRAIN_TIME="${2:?missing value for --train-time}"
      shift 2
      ;;
    --test-times)
      TEST_TIMES_CSV="${2:?missing value for --test-times}"
      shift 2
      ;;
    --ops)
      OPS_CSV="${2:?missing value for --ops}"
      shift 2
      ;;
    --k)
      K="${2:?missing value for --k}"
      shift 2
      ;;
    --taw-mag)
      TAW_MAG="${2:?missing value for --taw-mag}"
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
    --seed)
      SEED="${2:?missing value for --seed}"
      shift 2
      ;;
    --epoch)
      EPOCH="${2:?missing value for --epoch}"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --view)
      VIEW="${2:?missing value for --view}"
      shift 2
      ;;
    --test-first-n)
      TEST_FIRST_N="${2:?missing value for --test-first-n}"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="${2:?missing value for --eval-batch-size}"
      shift 2
      ;;
    --out-id)
      OUT_ID="${2:?missing value for --out-id}"
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
    --skip-eval)
      RUN_EVAL=0
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

case "$VIEW" in
  center|left|right) ;;
  *)
    echo "[ERR] --view must be center, left, or right: $VIEW" >&2
    exit 1
    ;;
esac

for f in "$CFG" "$AUGMIX_PY" "$TAW_PY" "$RANDAUG_PY" "$MERGE_PY" "$LEARN_DEFAULT_PY" "$COMPARE_NAMED_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

read_cfg() {
  local key="$1"
  "$PYTHON_BIN" - "$CFG" "$key" <<'PY'
import sys
import yaml

cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get(key, ""))
PY
}

DATA_ROOT="$("$PYTHON_BIN" - "$CFG" "$NAV_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path, nav_dir = sys.argv[1], Path(sys.argv[2])
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
pc = cfg.get("pc_user_name")
ws = cfg.get("ws_name")
if pc and ws:
    print(Path(f"/home/{pc}/{ws}/src/nav_cloning/data"))
else:
    print(nav_dir / "data")
PY
)"

if [[ -z "$EPOCH" ]]; then
  EPOCH="$(read_cfg epoch)"
fi
[[ -n "$EPOCH" ]] || { echo "[ERR] epoch is empty" >&2; exit 1; }
if [[ -z "$TRAIN_BATCH_SIZE" ]]; then
  TRAIN_BATCH_SIZE="$(read_cfg batch_size)"
fi
[[ -n "$TRAIN_BATCH_SIZE" ]] || { echo "[ERR] batch_size is empty" >&2; exit 1; }

trim_token() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

parse_csv() {
  local csv="$1"
  local -n out_ref="$2"
  local -a parsed=()
  local token trimmed
  IFS=',' read -r -a tokens <<< "$csv"
  for token in "${tokens[@]}"; do
    trimmed="$(trim_token "$token")"
    [[ -n "$trimmed" ]] || continue
    parsed+=("$trimmed")
  done
  [[ ${#parsed[@]} -gt 0 ]] || return 1
  out_ref=("${parsed[@]}")
}

declare -a TEST_TIMES=()
parse_csv "$TEST_TIMES_CSV" TEST_TIMES || { echo "[ERR] no valid --test-times" >&2; exit 1; }

check_dataset_exists() {
  local time_id="$1"
  local ds="$DATA_ROOT/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

check_dataset_exists "$TRAIN_TIME"
for t in "${TEST_TIMES[@]}"; do
  check_dataset_exists "$t"
done

if [[ -z "$OUT_ID" ]]; then
  SAFE_TRAIN_FOR_OUT="${TRAIN_TIME//:/-}"
  OUT_ID="_aug3op_compare_${SAFE_TRAIN_FOR_OUT}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "$OUT_ID" == */* ]]; then
  echo "[ERR] --out-id must be a folder name under nav_cloning/data, not a path: $OUT_ID" >&2
  exit 1
fi

OUT_DIR="$DATA_ROOT/$OUT_ID"
OUT_DATASET_DIR="$OUT_DIR/dataset"
EVAL_DIR="$OUT_DIR/eval"
MODEL_DIR="$OUT_DIR/model/default/$EPOCH"
mkdir -p "$OUT_DATASET_DIR" "$EVAL_DIR" "$MODEL_DIR"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
  echo "[CFG] restored: $CFG"
}
trap restore_config EXIT

set_cfg_for_aug() {
  local out_img="$1"
  local out_vel="$2"

  CFG_PATH="$CFG" TIME_ID="$TRAIN_TIME" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import yaml

p = os.environ["CFG_PATH"]
with open(p, "r") as f:
    cfg = yaml.safe_load(f)

cfg["time"] = os.environ["TIME_ID"]
cfg["train_times"] = []
cfg["input_aug_dataset_img"] = "img"
cfg["input_aug_dataset_vel"] = "vel"
cfg["output_aug_dataset_img"] = os.environ["OUT_IMG"]
cfg["output_aug_dataset_vel"] = os.environ["OUT_VEL"]

with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_' '_'
}

OPS_SAFE="$(safe_name "$OPS_CSV")"
RAND_P_TAG="$(printf '%s' "$RAND_P" | tr -d '.')"

AUGMIX_IMG="AugMix_img_K${K}_OPS${OPS_SAFE}"
AUGMIX_VEL="AugMix_vel_K${K}_OPS${OPS_SAFE}"
TAW_IMG="TrivialAugmentWide_img_K${K}_M${TAW_MAG}_OPS${OPS_SAFE}"
TAW_VEL="TrivialAugmentWide_vel_K${K}_M${TAW_MAG}_OPS${OPS_SAFE}"
RANDAUG_IMG="RandAugment_img_K${K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_OPS${OPS_SAFE}"
RANDAUG_VEL="RandAugment_vel_K${K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_OPS${OPS_SAFE}"

MIX_AUGMIX_IMG="mix_augmix_img_K${K}_OPS${OPS_SAFE}"
MIX_AUGMIX_VEL="mix_augmix_vel_K${K}_OPS${OPS_SAFE}"
MIX_TAW_IMG="mix_taw_img_K${K}_M${TAW_MAG}_OPS${OPS_SAFE}"
MIX_TAW_VEL="mix_taw_vel_K${K}_M${TAW_MAG}_OPS${OPS_SAFE}"
MIX_RANDAUG_IMG="mix_randaugment_img_K${K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_OPS${OPS_SAFE}"
MIX_RANDAUG_VEL="mix_randaugment_vel_K${K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_OPS${OPS_SAFE}"
CLEAN_IMG="clean_img"
CLEAN_VEL="clean_vel"

MODEL_CLEAN="model_gpu_default_clean.pt"
MODEL_AUGMIX="model_gpu_default_augmix_K${K}_ops3.pt"
MODEL_TAW="model_gpu_default_taw_K${K}_M${TAW_MAG}_ops3.pt"
MODEL_RANDAUG="model_gpu_default_randaugment_K${K}_N${RAND_N}_M${RAND_M}_p${RAND_P_TAG}_ops3.pt"

dataset_ready() {
  local img_dir="$1"
  local vel_dir="$2"
  [[ -d "$img_dir" && -f "$vel_dir/data.csv" ]]
}

run_augmix() {
  if dataset_ready "$DATA_ROOT/$TRAIN_TIME/dataset/$AUGMIX_IMG" "$DATA_ROOT/$TRAIN_TIME/dataset/$AUGMIX_VEL"; then
    echo "[INFO] reuse AugMix dataset: $AUGMIX_IMG / $AUGMIX_VEL"
    return 0
  fi
  echo "[RUN] AugMix K=$K ops=$OPS_CSV"
  set_cfg_for_aug "$AUGMIX_IMG" "$AUGMIX_VEL"
  "$PYTHON_BIN" "$AUGMIX_PY" --K "$K" --seed "$SEED" --allowed-ops "$OPS_CSV"
}

run_taw() {
  if dataset_ready "$DATA_ROOT/$TRAIN_TIME/dataset/$TAW_IMG" "$DATA_ROOT/$TRAIN_TIME/dataset/$TAW_VEL"; then
    echo "[INFO] reuse TrivialAugmentWide dataset: $TAW_IMG / $TAW_VEL"
    return 0
  fi
  echo "[RUN] TrivialAugmentWide K=$K M=$TAW_MAG ops=$OPS_CSV"
  set_cfg_for_aug "$TAW_IMG" "$TAW_VEL"
  "$PYTHON_BIN" "$TAW_PY" --K "$K" --same3view --num-magnitude-bins "$TAW_MAG" --allowed-ops "$OPS_CSV"
}

run_randaugment() {
  if dataset_ready "$DATA_ROOT/$TRAIN_TIME/dataset/$RANDAUG_IMG" "$DATA_ROOT/$TRAIN_TIME/dataset/$RANDAUG_VEL"; then
    echo "[INFO] reuse RandAugment dataset: $RANDAUG_IMG / $RANDAUG_VEL"
    return 0
  fi
  echo "[RUN] RandAugment K=$K N=$RAND_N M=$RAND_M p=$RAND_P ops=$OPS_CSV"
  set_cfg_for_aug "$RANDAUG_IMG" "$RANDAUG_VEL"
  "$PYTHON_BIN" "$RANDAUG_PY" \
    --K "$K" \
    --N "$RAND_N" \
    --M "$RAND_M" \
    --p "$RAND_P" \
    --seed "$SEED" \
    --sync_views \
    --allowed-ops "$OPS_CSV"
}

merge_clean_only() {
  echo "[RUN] build clean dataset: $CLEAN_IMG / $CLEAN_VEL"
  "$PYTHON_BIN" "$MERGE_PY" \
    --orig_img "$DATA_ROOT/$TRAIN_TIME/dataset/img" \
    --orig_csv "$DATA_ROOT/$TRAIN_TIME/dataset/vel/data.csv" \
    --out_img "$OUT_DATASET_DIR/$CLEAN_IMG" \
    --out_csv "$OUT_DATASET_DIR/$CLEAN_VEL/data.csv" \
    --mode symlink
}

merge_clean_aug() {
  local aug_img="$1"
  local aug_vel="$2"
  local out_img="$3"
  local out_vel="$4"

  echo "[RUN] build clean+aug dataset: $out_img / $out_vel"
  "$PYTHON_BIN" "$MERGE_PY" \
    --orig_img "$DATA_ROOT/$TRAIN_TIME/dataset/img" \
    --orig_csv "$DATA_ROOT/$TRAIN_TIME/dataset/vel/data.csv" \
    --aug_img "$DATA_ROOT/$TRAIN_TIME/dataset/$aug_img" \
    --aug_csv "$DATA_ROOT/$TRAIN_TIME/dataset/$aug_vel/data.csv" \
    --out_img "$OUT_DATASET_DIR/$out_img" \
    --out_csv "$OUT_DATASET_DIR/$out_vel/data.csv" \
    --mode symlink
}

train_default() {
  local label="$1"
  local load_img="$2"
  local load_vel="$3"
  local save_model="$4"
  local model_path="$MODEL_DIR/$save_model"

  if [[ -f "$model_path" ]]; then
    echo "[INFO] skip existing default $label model: $model_path"
    return 0
  fi

  echo "[RUN] train default $label (uses center/left/right)"
  "$PYTHON_BIN" "$LEARN_DEFAULT_PY" "$OUT_ID" "$EPOCH" \
    --data_root "$DATA_ROOT" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --load_dataset_img "$load_img" \
    --load_dataset_vel "$load_vel" \
    --save_model "$save_model" \
    --no_phase_training
}

MANIFEST="$OUT_DIR/run_manifest.txt"
{
  echo "train_time=$TRAIN_TIME"
  echo "test_times=$(IFS=,; echo "${TEST_TIMES[*]}")"
  echo "ops=$OPS_CSV"
  echo "k=$K"
  echo "taw_mag=$TAW_MAG"
  echo "rand_n=$RAND_N"
  echo "rand_m=$RAND_M"
  echo "rand_p=$RAND_P"
  echo "seed=$SEED"
  echo "epoch=$EPOCH"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "train_method=default_mse"
  echo "train_views=center,left,right"
  echo "eval_view=$VIEW"
  echo "test_first_n=$TEST_FIRST_N"
  echo "out_id=$OUT_ID"
} > "$MANIFEST"

echo "[INFO] output: $OUT_DIR"
echo "[INFO] ops: $OPS_CSV"

if [[ "$RUN_PREP" -eq 1 ]]; then
  merge_clean_only
  run_augmix
  run_taw
  run_randaugment

  merge_clean_aug "$AUGMIX_IMG" "$AUGMIX_VEL" "$MIX_AUGMIX_IMG" "$MIX_AUGMIX_VEL"
  merge_clean_aug "$TAW_IMG" "$TAW_VEL" "$MIX_TAW_IMG" "$MIX_TAW_VEL"
  merge_clean_aug "$RANDAUG_IMG" "$RANDAUG_VEL" "$MIX_RANDAUG_IMG" "$MIX_RANDAUG_VEL"
else
  echo "[STEP] skip prep (--skip-prep)"
fi

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  train_default "Clean" "$CLEAN_IMG" "$CLEAN_VEL" "$MODEL_CLEAN"
  train_default "AugMix" "$MIX_AUGMIX_IMG" "$MIX_AUGMIX_VEL" "$MODEL_AUGMIX"
  train_default "TrivialAugmentWide" "$MIX_TAW_IMG" "$MIX_TAW_VEL" "$MODEL_TAW"
  train_default "RandAugment" "$MIX_RANDAUG_IMG" "$MIX_RANDAUG_VEL" "$MODEL_RANDAUG"
else
  echo "[STEP] skip train (--skip-train)"
fi

if [[ "$RUN_EVAL" -eq 1 ]]; then
  for model_path in "$MODEL_DIR/$MODEL_CLEAN" "$MODEL_DIR/$MODEL_AUGMIX" "$MODEL_DIR/$MODEL_TAW" "$MODEL_DIR/$MODEL_RANDAUG"; do
    [[ -f "$model_path" ]] || { echo "[ERR] missing model: $model_path" >&2; exit 1; }
  done

  MODELS_CSV="$EVAL_DIR/models_for_compare.csv"
  cat > "$MODELS_CSV" <<EOF
model,model_path
default_clean,$MODEL_DIR/$MODEL_CLEAN
default_augmix_ops3_k${K},$MODEL_DIR/$MODEL_AUGMIX
default_taw_ops3_k${K},$MODEL_DIR/$MODEL_TAW
default_randaugment_ops3_k${K},$MODEL_DIR/$MODEL_RANDAUG
EOF

  compare_args=(
    "$COMPARE_NAMED_PY"
    "--models_csv" "$MODELS_CSV"
    "--test_times"
  )
  for t in "${TEST_TIMES[@]}"; do
    compare_args+=("$t")
  done
  compare_args+=(
    "--view" "$VIEW"
    "--batch_size" "$EVAL_BATCH_SIZE"
    "--out_dir" "$EVAL_DIR"
  )
  if [[ "$TEST_FIRST_N" == "all" ]]; then
    compare_args+=("--test-first-n" "1000000000")
  else
    compare_args+=("--test-first-n" "$TEST_FIRST_N")
  fi

  echo "[RUN] evaluate models"
  "$PYTHON_BIN" "${compare_args[@]}"

  "$PYTHON_BIN" - "$EVAL_DIR/summary_by_time_model.csv" "$EVAL_DIR" <<'PY'
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

summary_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
df = pd.read_csv(summary_csv)

method_map = {
    "default_clean": "Clean",
    "default_augmix": "AugMix",
    "default_taw": "TrivialAugmentWide",
    "default_randaugment": "RandAugment",
}

def method_name(model: str) -> str:
    for prefix, name in method_map.items():
        if str(model).startswith(prefix):
            return name
    return str(model)

df["method"] = df["model"].map(method_name)
df = df.sort_values(["test_time", "method"]).copy()
df.to_csv(out_dir / "summary_aug_method_by_time.csv", index=False)

wide = df.pivot_table(index="test_time", columns="method", values="mae", aggfunc="first")
wide = wide.reindex(columns=[c for c in ["Clean", "AugMix", "TrivialAugmentWide", "RandAugment"] if c in wide.columns])
wide.to_csv(out_dir / "mae_table_aug_method_compare.csv")

overall = (
    df.groupby("method")["mae"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
    .rename(columns={"count": "n_test_times", "mean": "mae_mean", "median": "mae_median", "std": "mae_std"})
)
overall = overall.sort_values(["mae_mean", "mae_median", "method"]).reset_index(drop=True)
overall.insert(0, "rank", range(1, len(overall) + 1))
overall.to_csv(out_dir / "summary_aug_method_ranked.csv", index=False)

plt.figure(figsize=(10, 5))
x = wide.index.tolist()
for c in wide.columns:
    plt.plot(x, wide[c].to_numpy(), marker="o", label=c)
plt.xlabel("test dataset time")
plt.ylabel("MAE")
plt.title("MAE over outdoor test datasets (clean vs 3-op augmentation)")
plt.xticks(rotation=25, ha="right")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(out_dir / "mae_aug_method_compare.png", dpi=180)
plt.close()

best = overall.iloc[0]
print(f"[BEST] method={best['method']} mae_mean={best['mae_mean']:.6f} mae_median={best['mae_median']:.6f}")
print(f"[DONE] ranked summary: {out_dir / 'summary_aug_method_ranked.csv'}")
print(f"[DONE] plot: {out_dir / 'mae_aug_method_compare.png'}")
PY
else
  echo "[STEP] skip eval (--skip-eval)"
fi

echo "[DONE] finished"
echo "  manifest                  : $MANIFEST"
echo "  models                    : $MODEL_DIR"
echo "  summary by time/model     : $EVAL_DIR/summary_by_time_model.csv"
echo "  summary by model          : $EVAL_DIR/summary_by_model.csv"
echo "  method ranking            : $EVAL_DIR/summary_aug_method_ranked.csv"
echo "  method MAE table          : $EVAL_DIR/mae_table_aug_method_compare.csv"
echo "  plot                      : $EVAL_DIR/mae_aug_method_compare.png"
