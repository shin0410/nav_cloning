#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$NAV_DIR/config/config.yaml"

TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
LEARN_DEFAULT_PY="$NAV_DIR/scripts/learning_default.py"
COMPARE_NAMED_PY="$NAV_DIR/scripts/compare_named_models_on_test_times.py"
PLOT_OP_PY="$NAV_DIR/scripts/plot_taw_op_compare.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TRAIN_TIME="20260426_13_39_33"
TEST_TIMES_CSV="20260428_12:31:05,20260428_16:50:47,20260428_19:29:22"
K=3
MAG=31
OPS_CSV="AutoContrast,Brightness,Color,Contrast,Equalize,Identity,Posterize,Sharpness,Solarize"
WITH_CLEAN=0
RUN_TRAIN=1
VIEW="center"
TEST_FIRST_N="6100"
TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=128
EPOCH=""
OUT_ID=""

usage() {
  cat <<'EOF'
Usage:
  run_taw_singleop_outdoor_20260426.sh [options]

Default flow:
  train data : 20260426_13_39_33
  test data  : 20260428_12:31:05, 20260428_16:50:47, 20260428_19:29:22
  TAW        : one operation at a time, K=3, num-magnitude-bins=31
  training   : default MSE learning; each model trains on that TAW-augmented dataset only
  eval view  : center

Options:
  --train-time TIME          default: 20260426_13_39_33
  --test-times CSV           default: 20260428_12:31:05,20260428_16:50:47,20260428_19:29:22
  --ops CSV|all              default: AutoContrast,Brightness,Color,Contrast,Equalize,Identity,Posterize,Sharpness,Solarize
  --k N                      default: 3
  --mag N                    default: 31
  --epoch N                  default: read nav_cloning/config/config.yaml
  --batch-size N             default: read nav_cloning/config/config.yaml
  --view center|left|right   default: center
  --test-first-n N|all       default: 6100
  --eval-batch-size N        default: 128
  --with-clean               train on clean original + TAW augmented data
  --skip-train               skip augmentation/training and evaluate existing models
  --out-id NAME              output folder name under nav_cloning/data/
  -h, --help

Outputs:
  nav_cloning/data/<out-id>/
    model/default/<epoch>/*.pt
    eval/summary_by_time_model.csv
    eval/summary_by_model.csv
    eval/summary_taw_op_overall.csv
    eval/summary_taw_op_ranked.csv
    eval/mae_taw_op_compare.png
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
    --mag)
      MAG="${2:?missing value for --mag}"
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
    --with-clean)
      WITH_CLEAN=1
      shift
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --out-id)
      OUT_ID="${2:?missing value for --out-id}"
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

case "$VIEW" in
  center|left|right) ;;
  *)
    echo "[ERR] --view must be center, left, or right: $VIEW" >&2
    exit 1
    ;;
esac

for f in "$CFG" "$TAW_PY" "$LEARN_DEFAULT_PY" "$COMPARE_NAMED_PY" "$PLOT_OP_PY"; do
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

if [[ -z "$OUT_ID" ]]; then
  SAFE_TRAIN_FOR_OUT="${TRAIN_TIME//:/-}"
  OUT_ID="_taw_singleop_outdoor_${SAFE_TRAIN_FOR_OUT}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "$OUT_ID" == */* ]]; then
  echo "[ERR] --out-id must be a folder name under nav_cloning/data, not a path: $OUT_ID" >&2
  exit 1
fi

OUT_DIR="$DATA_ROOT/$OUT_ID"
EVAL_DIR="$OUT_DIR/eval"
MODEL_DIR="$OUT_DIR/model/default/$EPOCH"
mkdir -p "$EVAL_DIR" "$MODEL_DIR"

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

AVAILABLE_OPS_RAW="$("$PYTHON_BIN" "$TAW_PY" --num-magnitude-bins "$MAG" --list-ops)"
if [[ "$OPS_CSV" == "all" ]]; then
  OPS_CSV="$AVAILABLE_OPS_RAW"
fi

declare -a OPS=()
parse_csv "$OPS_CSV" OPS || { echo "[ERR] no valid --ops" >&2; exit 1; }

declare -A AVAIL=()
declare -a AVAILABLE_OPS=()
parse_csv "$AVAILABLE_OPS_RAW" AVAILABLE_OPS || { echo "[ERR] failed to list TAW ops" >&2; exit 1; }
for op in "${AVAILABLE_OPS[@]}"; do
  AVAIL["$op"]=1
done
for op in "${OPS[@]}"; do
  [[ -n "${AVAIL[$op]+x}" ]] || {
    echo "[ERR] unsupported op '$op' for --mag=$MAG" >&2
    echo "      available: $AVAILABLE_OPS_RAW" >&2
    exit 1
  }
done

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

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
  echo "[CFG] restored: $CFG"
}
trap restore_config EXIT

set_cfg_for_taw() {
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

op_safe_name() {
  local op="$1"
  printf '%s' "$op" | tr -c 'A-Za-z0-9_' '_'
}

safe_time_name() {
  local time_id="$1"
  printf '%s' "$time_id" | tr ':' '-'
}

build_clean_aug_mix_dataset() {
  local aug_img="$1"
  local aug_vel="$2"
  local out_img="$3"
  local out_vel="$4"

  NAV_DATA_ROOT="$DATA_ROOT" TRAIN_TIME="$TRAIN_TIME" AUG_IMG="$aug_img" AUG_VEL="$aug_vel" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

import pandas as pd

data_root = Path(os.environ["NAV_DATA_ROOT"])
train_time = os.environ["TRAIN_TIME"]
aug_img = os.environ["AUG_IMG"]
aug_vel = os.environ["AUG_VEL"]
out_img = os.environ["OUT_IMG"]
out_vel = os.environ["OUT_VEL"]

base = data_root / train_time / "dataset"
out_img_dir = base / out_img
out_vel_dir = base / out_vel
out_csv = out_vel_dir / "data.csv"

for p in (out_img_dir, out_vel_dir):
    if p.is_dir():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

sources = [
    ("clean", base / "img", base / "vel" / "data.csv"),
    ("taw", base / aug_img, base / aug_vel / "data.csv"),
]

rows = []
missing_rows = 0
for prefix, img_dir, csv_path in sources:
    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing img dir: {img_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing csv: {csv_path}")
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' not found: {csv_path}")
    for _, row in df.iterrows():
        ep = str(row["episode"]).split(".")[0]
        new_ep = f"{prefix}_{ep}"
        ok = True
        for view in ("center", "left", "right"):
            src = img_dir / f"{ep}_{view}.npy"
            dst = out_img_dir / f"{new_ep}_{view}.npy"
            if not src.is_file():
                ok = False
                break
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        if not ok:
            missing_rows += 1
            continue
        rec = row.to_dict()
        rec["episode"] = new_ep
        rows.append(rec)

pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"[DONE] clean+TAW rows={len(rows)} missing_rows={missing_rows}")
print(f"[DONE] mix img: {out_img_dir}")
print(f"[DONE] mix csv: {out_csv}")
PY
}

model_name_for_op() {
  local op="$1"
  local safe_op train_safe mix_tag
  safe_op="$(op_safe_name "$op")"
  train_safe="$(safe_time_name "$TRAIN_TIME")"
  if [[ "$WITH_CLEAN" -eq 1 ]]; then
    mix_tag="plusclean"
  else
    mix_tag="augonly"
  fi
  echo "model_gpu_default_taw_singleop_K${K}_M${MAG}_OP${safe_op}_${mix_tag}_${train_safe}.pt"
}

link_one_path() {
  local src="$1"
  local dst="$2"

  [[ -e "$src" ]] || { echo "[ERR] source does not exist: $src" >&2; exit 1; }
  if [[ -L "$dst" || -f "$dst" ]]; then
    rm -f "$dst"
  elif [[ -e "$dst" ]]; then
    echo "[ERR] destination already exists and is not a symlink/file: $dst" >&2
    echo "      choose a new --out-id or remove that directory manually" >&2
    exit 1
  fi
  ln -s "$src" "$dst"
}

link_train_dataset_for_default() {
  local train_img="$1"
  local train_vel="$2"
  local out_dataset="$OUT_DIR/dataset"

  mkdir -p "$out_dataset"
  link_one_path "$DATA_ROOT/$TRAIN_TIME/dataset/$train_img" "$out_dataset/$train_img"
  link_one_path "$DATA_ROOT/$TRAIN_TIME/dataset/$train_vel" "$out_dataset/$train_vel"
}

run_taw_aug() {
  local out_img="$1"
  local out_vel="$2"
  local op="$3"

  echo "[RUN] TAW augmentation time=$TRAIN_TIME K=$K M=$MAG op=$op"
  set_cfg_for_taw "$out_img" "$out_vel"
  "$PYTHON_BIN" "$TAW_PY" \
    --K "$K" \
    --same3view \
    --num-magnitude-bins "$MAG" \
    --allowed-ops "$op"
}

run_train_one() {
  local op="$1"
  local train_img="$2"
  local train_vel="$3"
  local model_name="$4"

  echo "[RUN] train op=$op img=$train_img vel=$train_vel save=$MODEL_DIR/$model_name"
  link_train_dataset_for_default "$train_img" "$train_vel"
  "$PYTHON_BIN" "$LEARN_DEFAULT_PY" "$OUT_ID" "$EPOCH" \
    --data_root "$DATA_ROOT" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --load_dataset_img "$train_img" \
    --load_dataset_vel "$train_vel" \
    --save_model "$model_name" \
    --no_phase_training
}

MANIFEST="$OUT_DIR/run_manifest.txt"
{
  echo "train_time=$TRAIN_TIME"
  echo "test_times=$(IFS=,; echo "${TEST_TIMES[*]}")"
  echo "k=$K"
  echo "mag=$MAG"
  echo "ops=$(IFS=,; echo "${OPS[*]}")"
  echo "with_clean=$WITH_CLEAN"
  echo "view=$VIEW"
  echo "test_first_n=$TEST_FIRST_N"
  echo "epoch=$EPOCH"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "train_method=default_mse"
  echo "out_id=$OUT_ID"
} > "$MANIFEST"

echo "[INFO] output: $OUT_DIR"
echo "[INFO] ops: $(IFS=,; echo "${OPS[*]}")"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  for op in "${OPS[@]}"; do
    safe_op="$(op_safe_name "$op")"
    aug_img="TrivialAugmentWide_img_K${K}_M${MAG}_OP${safe_op}"
    aug_vel="TrivialAugmentWide_vel_K${K}_M${MAG}_OP${safe_op}"
    model_name="$(model_name_for_op "$op")"

    run_taw_aug "$aug_img" "$aug_vel" "$op"

    if [[ "$WITH_CLEAN" -eq 1 ]]; then
      train_img="mix_clean_taw_img_K${K}_M${MAG}_OP${safe_op}"
      train_vel="mix_clean_taw_vel_K${K}_M${MAG}_OP${safe_op}"
      build_clean_aug_mix_dataset "$aug_img" "$aug_vel" "$train_img" "$train_vel"
    else
      train_img="$aug_img"
      train_vel="$aug_vel"
    fi

    run_train_one "$op" "$train_img" "$train_vel" "$model_name"
  done
else
  echo "[STEP] skip augmentation/training (--skip-train)"
fi

MODELS_CSV="$EVAL_DIR/models_for_compare.csv"
MAP_CSV="$EVAL_DIR/taw_singleop_model_map.csv"
echo "model,model_path" > "$MODELS_CSV"
echo "k_alias,op_name,k,magnitude_bins,real_model_path,alias_model_path" > "$MAP_CSV"

k_alias=101
for op in "${OPS[@]}"; do
  model_path="$MODEL_DIR/$(model_name_for_op "$op")"
  [[ -f "$model_path" ]] || {
    echo "[ERR] model not found for op=$op: $model_path" >&2
    exit 1
  }
  echo "TrivialAugmentWide_K${k_alias},${model_path}" >> "$MODELS_CSV"
  echo "${k_alias},${op},${K},${MAG},${model_path},${model_path}" >> "$MAP_CSV"
  k_alias=$((k_alias + 1))
done

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

echo "[RUN] evaluate models on test times"
"$PYTHON_BIN" "${compare_args[@]}"

echo "[RUN] plot and summarize TAW op comparison"
"$PYTHON_BIN" "$PLOT_OP_PY" \
  --summary_csv "$EVAL_DIR/summary_by_time_model.csv" \
  --op_map_csv "$MAP_CSV" \
  --out_dir "$EVAL_DIR" \
  --title "MAE over outdoor test datasets (TAW single-op, K=$K, M=$MAG)"

"$PYTHON_BIN" - "$EVAL_DIR/summary_taw_op_overall.csv" "$EVAL_DIR/summary_taw_op_ranked.csv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
df = pd.read_csv(src)
ranked = df.sort_values(["mae_mean", "mae_median", "op_name"]).reset_index(drop=True)
ranked.insert(0, "rank", range(1, len(ranked) + 1))
ranked.to_csv(dst, index=False)
best = ranked.iloc[0]
print(f"[BEST] rank=1 op={best['op_name']} mae_mean={best['mae_mean']:.6f} mae_median={best['mae_median']:.6f}")
print(f"[DONE] ranked summary: {dst}")
PY

echo "[DONE] finished"
echo "  manifest                   : $MANIFEST"
echo "  models csv                 : $MODELS_CSV"
echo "  model map                  : $MAP_CSV"
echo "  summary by time/model      : $EVAL_DIR/summary_by_time_model.csv"
echo "  summary by model           : $EVAL_DIR/summary_by_model.csv"
echo "  op ranking                 : $EVAL_DIR/summary_taw_op_ranked.csv"
echo "  plot                       : $EVAL_DIR/mae_taw_op_compare.png"
